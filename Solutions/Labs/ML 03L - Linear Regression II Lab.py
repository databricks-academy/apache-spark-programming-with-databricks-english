# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="4e1b9835-762c-42f2-9ff8-75164cb1a702"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Linear Regression II Lab
# MAGIC
# MAGIC Alright! We're making progress. Still not a great RMSE or R2, but better than the baseline or just using a single feature.
# MAGIC
# MAGIC In the lab, you will see how to improve our performance even more.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Use RFormula to simplify the process of using StringIndexer, OneHotEncoder, and VectorAssembler
# MAGIC  - Transform the price into log(price), predict, and exponentiate the result for a lower RMSE

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md <i18n value="a427d25c-591f-4899-866a-14064eff40e3"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## RFormula
# MAGIC
# MAGIC Instead of manually specifying which columns are categorical to the StringIndexer and OneHotEncoder, <a href="(https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.RFormula.html?highlight=rformula#pyspark.ml.feature.RFormula" target="_blank">RFormula</a> can do that automatically for you.
# MAGIC
# MAGIC With RFormula, if you have any columns of type String, it treats it as a categorical feature and string indexes & one hot encodes it for us. Otherwise, it leaves as it is. Then it combines all of one-hot encoded features and numeric features into a single vector, called **`features`**.
# MAGIC
# MAGIC You can see a detailed example of how to use RFormula <a href="https://spark.apache.org/docs/latest/ml-features.html#rformula" target="_blank">here</a>.

# COMMAND ----------

# ANSWER
from pyspark.ml import Pipeline
from pyspark.ml.feature import RFormula
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

r_formula = RFormula(formula="price ~ .", featuresCol="features", labelCol="price", handleInvalid="skip") # Look at handleInvalid

lr = LinearRegression(labelCol="price", featuresCol="features")
pipeline = Pipeline(stages=[r_formula, lr])
pipeline_model = pipeline.fit(train_df)
pred_df = pipeline_model.transform(test_df)

regression_evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")
rmse = regression_evaluator.setMetricName("rmse").evaluate(pred_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# MAGIC %md <i18n value="c9898a31-90e4-4a6d-87e6-731b95c764bd"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Log Scale
# MAGIC
# MAGIC Now that we have verified we get the same result using RFormula as above, we are going to improve upon our model. If you recall, our price dependent variable appears to be log-normally distributed, so we are going to try to predict it on the log scale.
# MAGIC
# MAGIC Let's convert our price to be on log scale, and have the linear regression model predict the log price

# COMMAND ----------

from pyspark.sql.functions import log

display(train_df.select(log("price")))

# COMMAND ----------

# ANSWER
from pyspark.sql.functions import col, log

log_train_df = train_df.withColumn("log_price", log(col("price")))
log_test_df = test_df.withColumn("log_price", log(col("price")))

r_formula = RFormula(formula="log_price ~ . - price", featuresCol="features", labelCol="log_price", handleInvalid="skip") 

lr.setLabelCol("log_price").setPredictionCol("log_pred")
pipeline = Pipeline(stages=[r_formula, lr])
pipeline_model = pipeline.fit(log_train_df)
pred_df = pipeline_model.transform(log_test_df)

# COMMAND ----------

# MAGIC %md <i18n value="51b5e35f-e527-438a-ab56-2d4d0d389d29"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Exponentiate
# MAGIC
# MAGIC In order to interpret our RMSE, we need to convert our predictions back from logarithmic scale.

# COMMAND ----------

# ANSWER
from pyspark.sql.functions import col, exp

exp_df = pred_df.withColumn("prediction", exp(col("log_pred")))

rmse = regression_evaluator.setMetricName("rmse").evaluate(exp_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(exp_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# MAGIC %md <i18n value="05d3baa6-bb71-4c31-984b-a2daabc35f97"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Nice job! You have increased the R2 and dropped the RMSE significantly in comparison to the previous model.
# MAGIC
# MAGIC In the next few notebooks, we will see how we can reduce the RMSE even more.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
