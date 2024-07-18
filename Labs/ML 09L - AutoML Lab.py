# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="0f0f211a-70ba-4432-ab87-19bf7c8fc6cc"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # AutoML Lab
# MAGIC
# MAGIC <a href="https://docs.databricks.com/applications/machine-learning/automl.html" target="_blank">Databricks AutoML</a> helps you automatically build machine learning models both through a UI and programmatically. It prepares the dataset for model training and then performs and records a set of trials (using HyperOpt), creating, tuning, and evaluating multiple models. 
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC  - Use AutoML to automatically train and tune your models
# MAGIC  - Run AutoML in Python and through the UI
# MAGIC  - Interpret the results of an AutoML run

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="9178a13f-aeaf-49cb-ac27-600e7cea0077"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Currently, AutoML uses a combination of XGBoost and sklearn (only single node models) but optimizes the hyperparameters within each.

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md <i18n value="af913436-4be4-4a26-8381-d40d4e1af9d2"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Use the UI
# MAGIC
# MAGIC Instead of programmatically building our models, we can also use the UI. But first we need to register our dataset as a table.

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {DA.cleaned_username}")
train_df.write.mode("overwrite").saveAsTable(f"{DA.cleaned_username}.autoMLTable")

# COMMAND ----------

# MAGIC %md <i18n value="2f854d06-800c-428c-8add-aece6c9a91b6"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC First, make sure that you have selected the Machine Learning role on the left, before selecting start AutoML on the workspace homepage.
# MAGIC
# MAGIC <img src="http://files.training.databricks.com/images/301/AutoML_1_2.png" alt="step12" width="750"/>

# COMMAND ----------

# MAGIC %md <i18n value="98f64ede-5b15-442b-8346-874e0fdea6b5"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Select **`regression`** as the problem type, as well as the table we created in the previous cell. Then, select **`price`** as the column to predict.
# MAGIC
# MAGIC <img src="http://files.training.databricks.com/images/301/AutoML_UI.png" alt="ui" width="750"/>

# COMMAND ----------

# MAGIC %md <i18n value="4e561687-2509-4084-bd33-4221cb047eba"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC In the advanced configuration dropdown, change the evaluation metric to rmse, timeout to 5 minutes, and the maximum number of runs to 20.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/301/AutoML_Advanced.png" alt="advanced" width="500"/>

# COMMAND ----------

# MAGIC %md <i18n value="b15305f8-04cd-422f-a1da-ad7640b3846b"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Finally, we can start our run. Once completed, we can view the tuned model by clicking on the edit best model button.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/301/AutoMLResultsUpdated.png" alt="results" width="1000"/>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
