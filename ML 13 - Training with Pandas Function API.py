# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="d6718279-32b1-490e-8a38-f1d6e3578184"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Training with Pandas Function API
# MAGIC
# MAGIC This notebook demonstrates how to use Pandas Function API to manage and scale machine learning models for IoT devices. 
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Use <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.GroupedData.applyInPandas.html" target="_blank"> **.groupBy().applyInPandas()** </a> to build many models in parallel for each IoT Device

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md <i18n value="35af29dc-0fc5-4e37-963d-3fbe86f4ba59"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Create dummy data with:
# MAGIC - **`device_id`**: 10 different devices
# MAGIC - **`record_id`**: 10k unique records
# MAGIC - **`feature_1`**: a feature for model training
# MAGIC - **`feature_2`**: a feature for model training
# MAGIC - **`feature_3`**: a feature for model training
# MAGIC - **`label`**: the variable we're trying to predict

# COMMAND ----------

import pyspark.sql.functions as f

df = (spark
      .range(1000*100)
      .select(f.col("id").alias("record_id"), (f.col("id")%10).alias("device_id"))
      .withColumn("feature_1", f.rand() * 1)
      .withColumn("feature_2", f.rand() * 2)
      .withColumn("feature_3", f.rand() * 3)
      .withColumn("label", (f.col("feature_1") + f.col("feature_2") + f.col("feature_3")) + f.rand())
     )

display(df)

# COMMAND ----------

# MAGIC %md <i18n value="b5f90a62-80fd-4173-adf0-6e73d0e31309"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Define the return schema

# COMMAND ----------

train_return_schema = "device_id integer, n_used integer, model_path string, mse float"

# COMMAND ----------

# MAGIC %md <i18n value="e2ac315f-e950-48c6-9bb8-9ceede8f93dd"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Define a pandas function that takes all the data for a given device, train a model, saves it as a nested run, and returns a spark object with the above schema

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Trains an sklearn model on grouped instances
    """
    # Pull metadata
    device_id = df_pandas["device_id"].iloc[0]
    n_used = df_pandas.shape[0]
    run_id = df_pandas["run_id"].iloc[0] # Pulls run ID to do a nested run

    # Train the model
    X = df_pandas[["feature_1", "feature_2", "feature_3"]]
    y = df_pandas["label"]
    rf = RandomForestRegressor()
    rf.fit(X, y)

    # Evaluate the model
    predictions = rf.predict(X)
    mse = mean_squared_error(y, predictions) # Note we could add a train/test split

    # Resume the top-level training
    with mlflow.start_run(run_id=run_id) as outer_run:
        # Small hack for running as a job
        experiment_id = outer_run.info.experiment_id
        print(f"Current experiment_id = {experiment_id}")

        # Create a nested run for the specific device
        with mlflow.start_run(run_name=str(device_id), nested=True, experiment_id=experiment_id) as run:
            mlflow.sklearn.log_model(rf, str(device_id))
            mlflow.log_metric("mse", mse)
            mlflow.set_tag("device", str(device_id))

            artifact_uri = f"runs:/{run.info.run_id}/{device_id}"
            # Create a return pandas DataFrame that matches the schema above
            return_df = pd.DataFrame([[device_id, n_used, artifact_uri, mse]], 
                                    columns=["device_id", "n_used", "model_path", "mse"])

    return return_df 


# COMMAND ----------

# MAGIC %md <i18n value="2b6bf899-de7c-4ab9-b343-a11a832ddd77"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Apply the pandas function to grouped data. 
# MAGIC
# MAGIC Note that the way you would apply this in practice depends largely on where the data for inference is located. In this example, we'll reuse the training data which contains our device and run id's.

# COMMAND ----------

with mlflow.start_run(run_name="Training session for all devices") as run:
    run_id = run.info.run_id

    model_directories_df = (df
        .withColumn("run_id", f.lit(run_id)) # Add run_id
        .groupby("device_id")
        .applyInPandas(train_model, schema=train_return_schema)
        .cache()
    )

combined_df = df.join(model_directories_df, on="device_id", how="left")
display(combined_df)

# COMMAND ----------

# MAGIC %md <i18n value="3f660cc6-4979-48dd-beea-9dab9b536230"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Define a pandas function and return schema to apply the model.  *This needs only one read from DBFS per device.*

# COMMAND ----------

apply_return_schema = "record_id integer, prediction float"

def apply_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Applies model to data for a particular device, represented as a pandas DataFrame
    """
    model_path = df_pandas["model_path"].iloc[0]

    input_columns = ["feature_1", "feature_2", "feature_3"]
    X = df_pandas[input_columns]

    model = mlflow.sklearn.load_model(model_path)
    prediction = model.predict(X)

    return_df = pd.DataFrame({
        "record_id": df_pandas["record_id"],
        "prediction": prediction
    })
    return return_df

prediction_df = combined_df.groupby("device_id").applyInPandas(apply_model, schema=apply_return_schema)
display(prediction_df)

# COMMAND ----------

# MAGIC %md <i18n value="d760694c-8be7-4cbb-8825-8b8aa0d740db"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Serving Multiple Models from a Registered Model
# MAGIC
# MAGIC MLflow allows models to deploy as real-time REST APIs. At the moment, a single MLflow model serves from one instance (typically one VM). However, sometimes multiple models need to be served from a single endpoint. Imagine 1000 similar models that need to be served with different inputs. Running 1000 endpoints could waste resources, especially if certain models are underutilized.
# MAGIC
# MAGIC One way around this is to package many models into a single custom model, which internally routes to one of the models based on the input and deploys that 'bundle' of models as a single 'model.'
# MAGIC
# MAGIC Below we demonstrate creating such a custom model that bundles all of the models we trained for each device. For every row of data fed to this model, the model will determine the device id and then use the appropriate model trained on that device id to make predictions for a given row. 
# MAGIC
# MAGIC First, we need to access the models for each device id.

# COMMAND ----------

experiment_id = run.info.experiment_id

model_df = (spark.read.format("mlflow-experiment")
            .load(experiment_id)
            .filter("tags.device IS NOT NULL")
            .orderBy("end_time", ascending=False)
            .select("tags.device", "run_id")
            .limit(10))

display(model_df)

# COMMAND ----------

# MAGIC %md <i18n value="b9b38048-397b-4eb3-a7c7-541aef502d4a"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC We create a dictionary mapping device ids to the model trained on that device id.

# COMMAND ----------

device_to_model = {row["device"]: mlflow.sklearn.load_model(f"runs:/{row['run_id']}/{row['device']}") for row in model_df.collect()}
                                                          
device_to_model

# COMMAND ----------

# MAGIC %md <i18n value="f1081d85-677f-4a55-a3f5-a7e3a6710d3a"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC We create a custom model that takes the device id to model mappings as an attribute and delegates input to the appropriate model based on the device id.

# COMMAND ----------

from mlflow.pyfunc import PythonModel

class OriginDelegatingModel(PythonModel):
    
    def __init__(self, device_to_model_map):
        self.device_to_model_map = device_to_model_map
        
    def predict_for_device(self, row):
        '''
        This method applies to a single row of data by
        fetching the appropriate model and generating predictions
        '''
        model = self.device_to_model_map.get(str(row["device_id"]))
        data = row[["feature_1", "feature_2", "feature_3"]].to_frame().T
        return model.predict(data)[0]
    
    def predict(self, model_input):
        return model_input.apply(self.predict_for_device, axis=1)

# COMMAND ----------

# MAGIC %md <i18n value="da424f95-113f-4feb-a20c-6d0178d03bdb"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Here we demonstrate the use of this model.

# COMMAND ----------

example_model = OriginDelegatingModel(device_to_model)
example_model.predict(combined_df.toPandas().head(20))

# COMMAND ----------

# MAGIC %md <i18n value="624309e5-7ba8-4968-92d4-3fe71e36375b"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC From here we can log and then register the model to be used for serving models for all the device ids from one instance.

# COMMAND ----------

with mlflow.start_run():
    model = OriginDelegatingModel(device_to_model)
    mlflow.pyfunc.log_model("model", python_model=model)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
