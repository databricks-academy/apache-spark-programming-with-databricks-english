# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="45bb1181-9fe0-4255-b0f0-b42637fc9591"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC #Linear Regression Lab
# MAGIC
# MAGIC In the previous lesson, we predicted price using just one variable: bedrooms. Now, we want to predict price given a few other features.
# MAGIC
# MAGIC Steps:
# MAGIC 1. Use the features: **`bedrooms`**, **`bathrooms`**, **`bathrooms_na`**, **`minimum_nights`**, and **`number_of_reviews`** as input to your VectorAssembler.
# MAGIC 1. Build a Linear Regression Model
# MAGIC 1. Evaluate the **`RMSE`** and the **`R2`**.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Build a linear regression model with multiple features
# MAGIC  - Compute various metrics to evaluate goodness of fit

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# ANSWER
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression

vec_assembler = VectorAssembler(inputCols=["bedrooms", "bathrooms", "bathrooms_na", "minimum_nights", "number_of_reviews"], outputCol="features")

vec_train_df = vec_assembler.transform(train_df)
vec_test_df = vec_assembler.transform(test_df)

lr_model = LinearRegression(featuresCol="features", labelCol="price").fit(vec_train_df)

pred_df = lr_model.transform(vec_test_df)

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")
rmse = regression_evaluator.evaluate(pred_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# MAGIC %md <i18n value="25a260af-8d6e-4897-8228-80074c4f1d64"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Examine the coefficients for each of the variables.

# COMMAND ----------

for col, coef in zip(vec_assembler.getInputCols(), lr_model.coefficients):
    print(col, coef)
  
print(f"intercept: {lr_model.intercept}")

# COMMAND ----------

# MAGIC %md <i18n value="218d51b8-7453-4f6a-8965-5a60e8c80eaf"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Distributed Setting
# MAGIC
# MAGIC Although we can quickly solve for the parameters when the data is small, the closed form solution doesn't scale well to large datasets. 
# MAGIC
# MAGIC Spark uses the following approach to solve a linear regression problem:
# MAGIC
# MAGIC * First, Spark tries to use matrix decomposition to solve the linear regression problem. 
# MAGIC * If it fails, Spark then uses <a href="https://spark.apache.org/docs/latest/ml-advanced.html#limited-memory-bfgs-l-bfgs" target="_blank">L-BFGS</a> to solve for the parameters. L-BFGS is a limited-memory version of BFGS that is particularly suited to problems with very large numbers of variables. The <a href="https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm" target="_blank">BFGS</a> method belongs to <a href="https://en.wikipedia.org/wiki/Quasi-Newton_method" target="_blank">quasi-Newton methods</a>, which are used to either find zeroes or local maxima and minima of functions iteratively. 
# MAGIC
# MAGIC If you are interested in how linear regression is implemented in the distributed setting and bottlenecks, check out these lecture slides:
# MAGIC * <a href="https://files.training.databricks.com/static/docs/distributed-linear-regression-1.pdf" target="_blank">distributed-linear-regression-1</a>
# MAGIC * <a href="https://files.training.databricks.com/static/docs/distributed-linear-regression-2.pdf" target="_blank">distributed-linear-regression-2</a>

# COMMAND ----------

# MAGIC %md <i18n value="f3e00d9e-3b02-44cf-87b7-20b54ba350c9"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC Yikes! We built a pretty bad model. In the next notebook, we will see how we can further improve upon our model.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
