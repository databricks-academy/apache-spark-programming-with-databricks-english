# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="2b5dc285-0d50-4ea7-a71b-8a7aa355ad7c"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Inference with Pandas UDFs
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC - Build a scikit-learn model, track it with MLflow, and apply it at scale using the Pandas Scalar Iterator UDFs and **`mapInPandas()`**
# MAGIC
# MAGIC To learn more about Pandas UDFs, you can refer to this <a href="https://databricks.com/blog/2020/05/20/new-pandas-udfs-and-python-type-hints-in-the-upcoming-release-of-apache-spark-3-0.html" target="_blank">blog post</a> to see what's new in Spark 3.0.

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md <i18n value="8b52bca0-45f0-4ada-be31-c2c473fb8e77"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Train sklearn model and log it with MLflow

# COMMAND ----------

import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

with mlflow.start_run(run_name="sklearn-random-forest") as run:
    # Enable autologging 
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, log_models=True)
    # Import the data
    df = pd.read_csv(f"{DA.paths.datasets}/airbnb/sf-listings/airbnb-cleaned-mlflow.csv".replace("dbfs:/", "/dbfs/")).drop(["zipcode"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)

    # Create model
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md <i18n value="7ebcaaf9-c6f5-4c92-865a-c7f2c7afb555"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Create Spark DataFrame

# COMMAND ----------

spark_df = spark.createDataFrame(X_test)

# COMMAND ----------

# MAGIC %md <i18n value="1cdc4475-f55f-4126-9d38-dedb19577f4e"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Pandas/Vectorized UDFs
# MAGIC
# MAGIC As of Spark 2.3, there are Pandas UDFs available in Python to improve the efficiency of UDFs. Pandas UDFs utilize Apache Arrow to speed up computation. Let's see how that helps improve our processing time.
# MAGIC
# MAGIC * <a href="https://databricks.com/blog/2017/10/30/introducing-vectorized-udfs-for-pyspark.html" target="_blank">Blog post</a>
# MAGIC * <a href="https://spark.apache.org/docs/latest/sql-programming-guide.html#pyspark-usage-guide-for-pandas-with-apache-arrow" target="_blank">Documentation</a>
# MAGIC
# MAGIC <img src="https://databricks.com/wp-content/uploads/2017/10/image1-4.png" alt="Benchmark" width ="500" height="1500">
# MAGIC
# MAGIC The user-defined functions are executed by: 
# MAGIC * <a href="https://arrow.apache.org/" target="_blank">Apache Arrow</a>, is an in-memory columnar data format that is used in Spark to efficiently transfer data between JVM and Python processes with near-zero (de)serialization cost. See more <a href="https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html" target="_blank">here</a>.
# MAGIC * pandas inside the function, to work with pandas instances and APIs.
# MAGIC
# MAGIC **NOTE**: In Spark 3.0, you should define your Pandas UDF using Python type hints.

# COMMAND ----------

from pyspark.sql.functions import pandas_udf

@pandas_udf("double")
def predict(*args: pd.Series) -> pd.Series:
    model_path = f"runs:/{run.info.run_id}/model" 
    model = mlflow.sklearn.load_model(model_path) # Load model
    pdf = pd.concat(args, axis=1)
    return pd.Series(model.predict(pdf))

prediction_df = spark_df.withColumn("prediction", predict(*spark_df.columns))
display(prediction_df)

# COMMAND ----------

# MAGIC %md <i18n value="e97526c6-ef40-4d55-9763-ee3ebe846096"/>
# MAGIC
# MAGIC
# MAGIC  
# MAGIC ### Pandas Scalar Iterator UDF
# MAGIC
# MAGIC If your model is very large, then there is high overhead for the Pandas UDF to repeatedly load the same model for every batch in the same Python worker process. In Spark 3.0, Pandas UDFs can accept an iterator of pandas.Series or pandas.DataFrame so that you can load the model only once instead of loading it for every series in the iterator.
# MAGIC
# MAGIC This way the cost of any set-up needed will be incurred fewer times. When the number of records you’re working with is greater than **`spark.conf.get('spark.sql.execution.arrow.maxRecordsPerBatch')`**, which is 10,000 by default, you should see speed ups over a pandas scalar UDF because it iterates through batches of pd.Series.
# MAGIC
# MAGIC It has the general syntax of: 
# MAGIC
# MAGIC ```
# MAGIC @pandas_udf(...)
# MAGIC def predict(iterator):
# MAGIC     model = ... # load model
# MAGIC     for features in iterator:
# MAGIC         yield model.predict(features)
# MAGIC ```

# COMMAND ----------

from typing import Iterator, Tuple

@pandas_udf("double")
def predict(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
    model_path = f"runs:/{run.info.run_id}/model" 
    model = mlflow.sklearn.load_model(model_path) # Load model
    for features in iterator:
        pdf = pd.concat(features, axis=1)
        yield pd.Series(model.predict(pdf))

prediction_df = spark_df.withColumn("prediction", predict(*spark_df.columns))
display(prediction_df)

# COMMAND ----------

# MAGIC %md <i18n value="23b8296e-e0bc-481e-bd35-4048d532c71d"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Pandas Function API
# MAGIC
# MAGIC Instead of using a Pandas UDF, we can use a Pandas Function API. This new category in Apache Spark 3.0 enables you to directly apply a Python native function, which takes and outputs Pandas instances against a PySpark DataFrame. Pandas Functions APIs supported in Apache Spark 3.0 are: grouped map, map, and co-grouped map.
# MAGIC
# MAGIC **`mapInPandas()`** takes an iterator of pandas.DataFrame as input, and outputs another iterator of pandas.DataFrame. It's flexible and easy to use if your model requires all of your columns as input, but it requires serialization/deserialization of the whole DataFrame (as it is passed to its input). You can control the size of each pandas.DataFrame with the **`spark.sql.execution.arrow.maxRecordsPerBatch`** config.

# COMMAND ----------

def predict(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    model_path = f"runs:/{run.info.run_id}/model" 
    model = mlflow.sklearn.load_model(model_path) # Load model
    for features in iterator:
        yield pd.concat([features, pd.Series(model.predict(features), name="prediction")], axis=1)
    
display(spark_df.mapInPandas(predict, """`host_total_listings_count` DOUBLE,`neighbourhood_cleansed` BIGINT,`latitude` DOUBLE,`longitude` DOUBLE,`property_type` BIGINT,`room_type` BIGINT,`accommodates` DOUBLE,`bathrooms` DOUBLE,`bedrooms` DOUBLE,`beds` DOUBLE,`bed_type` BIGINT,`minimum_nights` DOUBLE,`number_of_reviews` DOUBLE,`review_scores_rating` DOUBLE,`review_scores_accuracy` DOUBLE,`review_scores_cleanliness` DOUBLE,`review_scores_checkin` DOUBLE,`review_scores_communication` DOUBLE,`review_scores_location` DOUBLE,`review_scores_value` DOUBLE, `prediction` DOUBLE""")) 

# COMMAND ----------

# MAGIC %md <i18n value="d13b87a7-0625-4acc-88dc-438cf06e18bd"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Or you can define the schema like this below.

# COMMAND ----------

from pyspark.sql.functions import lit
from pyspark.sql.types import DoubleType

schema = spark_df.withColumn("prediction", lit(None).cast(DoubleType())).schema
display(spark_df.mapInPandas(predict, schema)) 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
