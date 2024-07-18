# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="c311be95-77f9-477b-93a5-c9289b3dedb6"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Pandas API on Spark
# MAGIC
# MAGIC The pandas API on Spark project makes data scientists more productive when interacting with big data, by implementing the pandas DataFrame API on top of Apache Spark. By unifying the two ecosystems with a familiar API, pandas API on Spark offers a seamless transition between small and large data. 
# MAGIC
# MAGIC Some of you might be familiar with the <a href="https://github.com/databricks/koalas" target="_blank">Koalas</a> project, which has been merged into PySpark in 3.2. For Apache Spark 3.2 and above, please use PySpark directly as the standalone Koalas project is now in maintenance mode. See this <a href="https://databricks.com/blog/2021/10/04/pandas-api-on-upcoming-apache-spark-3-2.html" target="_blank">blog post</a>.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC - Demonstrate the similarities of the pandas API on Spark API with the pandas API
# MAGIC - Understand the differences in syntax for the same DataFrame operations in pandas API on Spark vs PySpark

# COMMAND ----------

# MAGIC %md <i18n value="d711990a-af32-4357-b710-d2db434e4f15"/>
# MAGIC
# MAGIC
# MAGIC  
# MAGIC
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://files.training.databricks.com/images/301/31gb.png" width="900"/>
# MAGIC </div>
# MAGIC
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://files.training.databricks.com/images/301/95gb.png" width="900"/>
# MAGIC </div>
# MAGIC
# MAGIC **Pandas** DataFrames are mutable, eagerily evaluated, and maintain row order. They are restricted to a single machine, and are very performant when the data sets are small, as shown in a).
# MAGIC
# MAGIC **Spark** DataFrames are distributed, lazily evaluated, immutable, and do not maintain row order. They are very performant when working at scale, as shown in b) and c).
# MAGIC
# MAGIC **pandas API on Spark** provides the best of both worlds: pandas API with the performance benefits of Spark. However, it is not as fast as implementing your solution natively in Spark, and let's see why below.

# COMMAND ----------

# MAGIC %md <i18n value="c3080510-c8d9-4020-9910-37199f0ad5de"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## InternalFrame
# MAGIC
# MAGIC The InternalFrame holds the current Spark DataFrame and internal immutable metadata.
# MAGIC
# MAGIC It manages mappings from pandas API on Spark column names to Spark column names, as well as from pandas API on Spark index names to Spark column names. 
# MAGIC
# MAGIC If a user calls some API, the pandas API on Spark DataFrame updates the Spark DataFrame and metadata in InternalFrame. It creates or copies the current InternalFrame with the new states, and returns a new pandas API on Spark DataFrame.
# MAGIC
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://files.training.databricks.com/images/301/InternalFramePs.png" width="900"/>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="785ed714-6726-40d5-b7fb-c63c094e568e"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## InternalFrame Metadata Updates Only
# MAGIC
# MAGIC Sometimes the update of Spark DataFrame is not needed but of metadata only, then new structure will be like this.
# MAGIC
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://files.training.databricks.com/images/301/InternalFrameMetadataPs.png" width="900"/>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="e6d7a47f-a4c8-4178-bc70-62c2ac6764d5"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## InternalFrame Inplace Updates
# MAGIC
# MAGIC On the other hand, sometimes pandas API on Spark DataFrame updates internal state instead of returning a new DataFrame, for example, the argument  inplace=True is provided, then new structure will be like this.
# MAGIC
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://files.training.databricks.com/images/301/InternalFrameUpdate.png" width="900"/>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="23a2fc6d-1360-4e41-beab-b1fe8e23aac3"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Read in the dataset
# MAGIC
# MAGIC * PySpark
# MAGIC * pandas
# MAGIC * pandas API on Spark

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="1be64dea-9d63-476d-a7d6-9f6fa4ccd784"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Read in Parquet with PySpark

# COMMAND ----------

spark_df = spark.read.parquet(f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/")
display(spark_df)

# COMMAND ----------

# MAGIC %md <i18n value="00b99bdc-e4d1-44d2-b117-ae2cd97d0490"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Read in Parquet with pandas

# COMMAND ----------

import pandas as pd

pandas_df = pd.read_parquet(f"{DA.paths.datasets.replace('dbfs:/', '/dbfs/')}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/")
pandas_df.head()

# COMMAND ----------

# MAGIC %md <i18n value="e75a3ba6-98f6-4b39-aecb-345109cb2ce9"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Read in Parquet with pandas API on Spark. You'll notice pandas API on Spark generates an index column for you, like in pandas.
# MAGIC
# MAGIC Pandas API on Spark also supports reading from Delta (**`read_delta`**), but pandas does not support that yet.

# COMMAND ----------

import pyspark.pandas as ps

df = ps.read_parquet(f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/")
df.head()

# COMMAND ----------

# MAGIC %md <i18n value="f099c73b-0bd8-4ff1-a12e-578ffb0cb152"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### <a href="https://koalas.readthedocs.io/en/latest/user_guide/options.html#default-index-type" target="_blank">Index Types</a>
# MAGIC
# MAGIC ![](https://files.training.databricks.com/images/301/koalas_index.png)

# COMMAND ----------

ps.set_option("compute.default_index_type", "distributed-sequence")
df_dist_sequence = ps.read_parquet(f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/")
df_dist_sequence.head()

# COMMAND ----------

# MAGIC %md <i18n value="07b3f029-f81b-442f-8cdd-cb2d29033a35"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Converting to pandas API on Spark DataFrame to/from Spark DataFrame

# COMMAND ----------

# MAGIC %md <i18n value="ed25204e-2822-4694-b3b3-968ea8ef7343"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Creating a pandas API on Spark DataFrame from PySpark DataFrame

# COMMAND ----------

df = ps.DataFrame(spark_df)
display(df)

# COMMAND ----------

# MAGIC %md <i18n value="a41480c7-1787-4bd6-a4c3-c85552a5f762"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Alternative way of creating a pandas API on Spark DataFrame from PySpark DataFrame

# COMMAND ----------

df = spark_df.to_pandas_on_spark()
display(df)

# COMMAND ----------

# MAGIC %md <i18n value="5abf965b-2f69-469e-a0cf-ba8ffd714764"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Go from a pandas API on Spark DataFrame to a Spark DataFrame

# COMMAND ----------

display(df.to_spark())

# COMMAND ----------

# MAGIC %md <i18n value="480e9e60-9286-4f4c-9db3-b650b32cb7ce"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Value Counts

# COMMAND ----------

# MAGIC %md <i18n value="99f93d32-d09d-4fea-9ac9-57099eb2c819"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Get value counts of the different property types with PySpark

# COMMAND ----------

display(spark_df.groupby("property_type").count().orderBy("count", ascending=False))

# COMMAND ----------

# MAGIC %md <i18n value="150b6a18-123d-431a-84b1-ad2d2b7beae2"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Get value counts of the different property types with pandas API on Spark

# COMMAND ----------

df["property_type"].value_counts()

# COMMAND ----------

# MAGIC %md <i18n value="767f19b5-137f-4b33-9ef4-e5bb48603299"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Visualizations
# MAGIC
# MAGIC Based on the type of visualization, the pandas API on Spark has optimized ways to execute the plotting.
# MAGIC <br><br>
# MAGIC
# MAGIC ![](https://files.training.databricks.com/images/301/ps_plotting.png)

# COMMAND ----------

df.plot(kind="hist", x="bedrooms", y="price", bins=200)

# COMMAND ----------

# MAGIC %md <i18n value="6b70f1df-dfe1-43de-aeec-5541b036927c"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### SQL on pandas API on Spark DataFrames

# COMMAND ----------

ps.sql("SELECT distinct(property_type) FROM {df}", df=df)

# COMMAND ----------

# MAGIC %md <i18n value="7345361b-e6c4-4ce3-9ba4-8f132c8c8df2"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Interesting Facts
# MAGIC
# MAGIC * With pandas API on Spark you can read from Delta Tables and read in a directory of files
# MAGIC * If you use apply on a pandas API on Spark DF and that DF is <1000 (by default), pandas API on Spark will use pandas as a shortcut - this can be adjusted using **`compute.shortcut_limit`**
# MAGIC * When you create bar plots, the top n rows are only used - this can be adjusted using **`plotting.max_rows`**
# MAGIC * How to utilize **`.apply`** <a href="https://koalas.readthedocs.io/en/latest/reference/api/databricks.koalas.DataFrame.apply.html#databricks.koalas.DataFrame.apply" target="_blank">docs</a> with its use of return type hints similar to pandas UDFs
# MAGIC * How to check the execution plan, as well as caching a pandas API on Spark DF (which aren't immediately intuitive)
# MAGIC * Koalas are marsupials whose max speed is 30 kph (20 mph)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
