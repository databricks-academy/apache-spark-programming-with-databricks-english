# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="1fa7a9c8-3dad-454e-b7ac-555020a4bda8"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Hyperopt
# MAGIC
# MAGIC Hyperopt is a Python library for "serial and parallel optimization over awkward search spaces, which may include real-valued, discrete, and conditional dimensions".
# MAGIC
# MAGIC In the machine learning workflow, hyperopt can be used to distribute/parallelize the hyperparameter optimization process with more advanced optimization strategies than are available in other libraries.
# MAGIC
# MAGIC There are two ways to scale hyperopt with Apache Spark:
# MAGIC * Use single-machine hyperopt with a distributed training algorithm (e.g. MLlib)
# MAGIC * Use distributed hyperopt with single-machine training algorithms (e.g. scikit-learn) with the SparkTrials class. 
# MAGIC
# MAGIC In this lesson, we will use single-machine hyperopt with MLlib, but in the lab, you will see how to use hyperopt to distribute the hyperparameter tuning of single node models. 
# MAGIC
# MAGIC Unfortunately you can’t use hyperopt to distribute the hyperparameter optimization for distributed training algorithms at this time. However, you do still get the benefit of using more advanced hyperparameter search algorthims (random search, TPE, etc.) with Spark ML.
# MAGIC
# MAGIC
# MAGIC Resources:
# MAGIC
# MAGIC 0. <a href="http://hyperopt.github.io/hyperopt/scaleout/spark/" target="_blank">Documentation</a>
# MAGIC 0. <a href="https://docs.databricks.com/applications/machine-learning/automl/hyperopt/index.html" target="_blank">Hyperopt on Databricks</a>
# MAGIC 0. <a href="https://databricks.com/blog/2019/06/07/hyperparameter-tuning-with-mlflow-apache-spark-mllib-and-hyperopt.html" target="_blank">Hyperparameter Tuning with MLflow, Apache Spark MLlib and Hyperopt</a>
# MAGIC 0. <a href="https://databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html" target="_blank">How (Not) to Tune Your Model With Hyperopt</a>
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Use hyperopt to find the optimal parameters for an MLlib model using TPE

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="2340cdf4-9753-41b4-a613-043b90f0f472"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's start by loading in our SF Airbnb Dataset.

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, val_df, test_df = airbnb_df.randomSplit([.6, .2, .2], seed=42)

# COMMAND ----------

# MAGIC %md <i18n value="37bbd5bd-f330-4d02-8af6-1b185612cdf8"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC We will then create our random forest pipeline and regression evaluator.

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price"))]
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

rf = RandomForestRegressor(labelCol="price", maxBins=40, seed=42)
pipeline = Pipeline(stages=[string_indexer, vec_assembler, rf])
regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price")

# COMMAND ----------

# MAGIC %md <i18n value="e4627900-f2a5-4f65-881e-1374187dd4f9"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Next, we get to the hyperopt-specific part of the workflow.
# MAGIC
# MAGIC First, we define our **objective function**. The objective function has two primary requirements:
# MAGIC
# MAGIC 1. An **input** **`params`** including hyperparameter values to use when training the model
# MAGIC 2. An **output** containing a loss metric on which to optimize
# MAGIC
# MAGIC In this case, we are specifying values of **`max_depth`** and **`num_trees`** and returning the RMSE as our loss metric.
# MAGIC
# MAGIC We are reconstructing our pipeline for the **`RandomForestRegressor`** to use the specified hyperparameter values.

# COMMAND ----------

def objective_function(params):    
    # set the hyperparameters that we want to tune
    max_depth = params["max_depth"]
    num_trees = params["num_trees"]

    with mlflow.start_run():
        estimator = pipeline.copy({rf.maxDepth: max_depth, rf.numTrees: num_trees})
        model = estimator.fit(train_df)

        preds = model.transform(val_df)
        rmse = regression_evaluator.evaluate(preds)
        mlflow.log_metric("rmse", rmse)

    return rmse

# COMMAND ----------

# MAGIC %md <i18n value="d4f9dd2b-060b-4eef-8164-442b2be242f4"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Next, we define our search space. 
# MAGIC
# MAGIC This is similar to the parameter grid in a grid search process. However, we are only specifying the range of values rather than the individual, specific values to be tested. It's up to hyperopt's optimization algorithm to choose the actual values.
# MAGIC
# MAGIC See the <a href="https://github.com/hyperopt/hyperopt/wiki/FMin" target="_blank">documentation</a> for helpful tips on defining your search space.

# COMMAND ----------

from hyperopt import hp

search_space = {
    "max_depth": hp.quniform("max_depth", 2, 5, 1),
    "num_trees": hp.quniform("num_trees", 10, 100, 1)
}

# COMMAND ----------

# MAGIC %md <i18n value="27891521-e481-4734-b21c-b2c5fe1f01fe"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC **`fmin()`** generates new hyperparameter configurations to use for your **`objective_function`**. It will evaluate 4 models in total, using the information from the previous models to make a more informative decision for the the next hyperparameter to try. 
# MAGIC
# MAGIC Hyperopt allows for parallel hyperparameter tuning using either random search or Tree of Parzen Estimators (TPE). Note that in the cell below, we are importing **`tpe`**. According to the <a href="http://hyperopt.github.io/hyperopt/scaleout/spark/" target="_blank">documentation</a>, TPE is an adaptive algorithm that 
# MAGIC
# MAGIC > iteratively explores the hyperparameter space. Each new hyperparameter setting tested will be chosen based on previous results. 
# MAGIC
# MAGIC Hence, **`tpe.suggest`** is a Bayesian method.
# MAGIC
# MAGIC MLflow also integrates with Hyperopt, so you can track the results of all the models you’ve trained and their results as part of your hyperparameter tuning. Notice you can track the MLflow experiment in this notebook, but you can also specify an external experiment.

# COMMAND ----------

from hyperopt import fmin, tpe, Trials
import numpy as np
import mlflow
import mlflow.spark
mlflow.pyspark.ml.autolog(log_models=False)

num_evals = 4
trials = Trials()
best_hyperparam = fmin(fn=objective_function, 
                       space=search_space,
                       algo=tpe.suggest, 
                       max_evals=num_evals,
                       trials=trials,
                       rstate=np.random.default_rng(42))

# Retrain model on train & validation dataset and evaluate on test dataset
with mlflow.start_run():
    best_max_depth = best_hyperparam["max_depth"]
    best_num_trees = best_hyperparam["num_trees"]
    estimator = pipeline.copy({rf.maxDepth: best_max_depth, rf.numTrees: best_num_trees})
    combined_df = train_df.union(val_df) # Combine train & validation together

    pipeline_model = estimator.fit(combined_df)
    pred_df = pipeline_model.transform(test_df)
    rmse = regression_evaluator.evaluate(pred_df)

    # Log param and metrics for the final model
    mlflow.log_param("maxDepth", best_max_depth)
    mlflow.log_param("numTrees", best_num_trees)
    mlflow.log_metric("rmse", rmse)
    mlflow.spark.log_model(pipeline_model, "model")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
