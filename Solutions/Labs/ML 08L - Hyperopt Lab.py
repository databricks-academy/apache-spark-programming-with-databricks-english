# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="be8397b6-c087-4d7b-8302-5652eec27caf"/>
# MAGIC
# MAGIC
# MAGIC  
# MAGIC # Hyperopt Lab
# MAGIC
# MAGIC The <a href="https://github.com/hyperopt/hyperopt" target="_blank">Hyperopt library</a> allows for parallel hyperparameter tuning using either random search or Tree of Parzen Estimators (TPE). With MLflow, we can record the hyperparameters and corresponding metrics for each hyperparameter combination. You can read more on <a href="https://github.com/hyperopt/hyperopt/blob/master/docs/templates/scaleout/spark.md" target="_blank">SparkTrials w/ Hyperopt</a>.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC - Learn how to distribute tuning tasks when training a single-node machine learning model by using **`SparkTrials`** class, rather than the default **`Trials`** class. 
# MAGIC
# MAGIC > SparkTrials fits and evaluates each model on one Spark executor, allowing massive scale-out for tuning. To use SparkTrials with Hyperopt, simply pass the SparkTrials object to Hyperopt's fmin() function.

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="13b0389c-cbd8-4b31-9f15-a6a9f18e8f60"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Read in a cleaned version of the Airbnb dataset with just numeric features.

# COMMAND ----------

from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv(f"{DA.paths.datasets}/airbnb/sf-listings/airbnb-cleaned-mlflow.csv".replace("dbfs:/", "/dbfs/")).drop(["zipcode"], axis=1)

# split 80/20 train-test
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1),
                                                    df[["price"]].values.ravel(),
                                                    test_size = 0.2,
                                                    random_state = 42)

# COMMAND ----------

# MAGIC %md <i18n value="b84062c7-9fb2-4d34-a196-98e5074c7ad4"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now we need to define an **`objective_function`** where you evaluate the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html" target="_blank">random forest's</a> predictions using R2.
# MAGIC
# MAGIC In the code below, compute the **`r2`** and return it (remember we are trying to maximize R2, so we need to return it as a negative value).

# COMMAND ----------

# ANSWER
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, r2_score
from numpy import mean
  
def objective_function(params):
    # set the hyperparameters that we want to tune:
    max_depth = params["max_depth"]
    max_features = params["max_features"]

    regressor = RandomForestRegressor(max_depth=max_depth, max_features=max_features, random_state=42)

    # Evaluate predictions
    r2 = mean(cross_val_score(regressor, X_train, y_train, cv=3))

    # Note: since we aim to maximize r2, we need to return it as a negative value ("loss": -metric)
    return -r2

# COMMAND ----------

# MAGIC %md <i18n value="7b10a96d-d868-4603-ab84-50388a8f50fc"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC We need to define a search space for HyperOpt. Let the **`max_depth`** vary between 2-10, and **`max_features`** be one of: "auto", "sqrt", or "log2".

# COMMAND ----------

# ANSWER
from hyperopt import hp

max_features_choices =  ["auto", "sqrt", "log2"]
search_space = {
    "max_depth": hp.quniform("max_depth", 2, 10, 1),
    "max_features": hp.choice("max_features", max_features_choices)
}

# COMMAND ----------

# MAGIC %md <i18n value="6db6a36a-e1ca-400d-81fc-20ad5a794a01"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Instead of using the default **`Trials`** class, you can leverage the **`SparkTrials`** class to trigger the distribution of tuning tasks across Spark executors. On Databricks, SparkTrials are automatically logged with MLflow.
# MAGIC
# MAGIC **`SparkTrials`** takes 3 optional arguments, namely **`parallelism`**, **`timeout`**, and **`spark_session`**. You can refer to this <a href="http://hyperopt.github.io/hyperopt/scaleout/spark/" target="_blank">page</a> to read more.
# MAGIC
# MAGIC In the code below, fill in the **`fmin`** function.

# COMMAND ----------

# ANSWER
from hyperopt import fmin, tpe, SparkTrials
import mlflow
import numpy as np

# Number of models to evaluate
num_evals = 8
# Number of models to train concurrently
spark_trials = SparkTrials(parallelism=2)
# Automatically logs to MLflow
best_hyperparam = fmin(fn=objective_function, 
                       space=search_space,
                       algo=tpe.suggest, 
                       trials=spark_trials,
                       max_evals=num_evals,
                       rstate=np.random.default_rng(42))

# Re-train best model and log metrics on test dataset
with mlflow.start_run(run_name="best_model"):
    # get optimal hyperparameter values
    best_max_depth = best_hyperparam["max_depth"]
    best_max_features = max_features_choices[best_hyperparam["max_features"]]

    # train model on entire training data
    regressor = RandomForestRegressor(max_depth=best_max_depth, max_features=best_max_features, random_state=42)
    regressor.fit(X_train, y_train)

    # evaluate on holdout/test data
    r2 = regressor.score(X_test, y_test)

    # Log param and metric for the final model
    mlflow.log_param("max_depth", best_max_depth)
    mlflow.log_param("max_features", best_max_features)
    mlflow.log_metric("loss", r2)

# COMMAND ----------

# MAGIC %md <i18n value="398681fb-0ab4-4886-bb08-58117da3b7af"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now you can compare all of the models using the MLflow UI. 
# MAGIC
# MAGIC To understand the effect of tuning a hyperparameter:
# MAGIC
# MAGIC 0. Select the resulting runs and click Compare.
# MAGIC 0. In the Scatter Plot, select a hyperparameter for the X-axis and loss for the Y-axis.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
