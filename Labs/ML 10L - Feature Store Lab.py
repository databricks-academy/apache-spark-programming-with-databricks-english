# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="c7e3123c-9ace-4d9a-89a1-10307b238964"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Feature Store Lab
# MAGIC
# MAGIC Now that you are familiar with the <a href="https://docs.databricks.com/applications/machine-learning/feature-store.html" target="_blank">Databricks Feature Store</a>, try applying the concepts we learned to a new dataset below.
# MAGIC
# MAGIC The Feature Store Python API documentation can be found <a href="https://docs.databricks.com/dev-tools/api/python/latest/index.html#feature-store-python-api-reference" target="_blank">here</a>.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you will:<br>
# MAGIC  - Create a feature store 
# MAGIC  - Update existing feature tables
# MAGIC  - Register a MLflow model with feature tables
# MAGIC  - Perform batch scoring

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="2861b87d-095f-44c7-848e-9e4ea539ed2d"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Load the data
# MAGIC For this example, we will use a new COVID-19 dataset. Run the cell below to create our dataframe **`covid_df`**.

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

file_path = f"{DA.paths.datasets}/COVID/coronavirusdataset/Time.csv"
covid_df = (spark.read
            .format("csv")
            .option("header",True)
            .option("inferSchema", True)
            .load(file_path)
            .withColumn("index", monotonically_increasing_id()))

display(covid_df)

# COMMAND ----------

# MAGIC %md <i18n value="e0cab81f-4d2b-486b-9af7-01f6fb212168"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Run the cell below to set up a database and unique table name **`table_name`** for the lab.

# COMMAND ----------

import uuid

spark.sql(f"CREATE DATABASE IF NOT EXISTS {DA.cleaned_username}")
table_name = f"{DA.cleaned_username}.airbnb_{str(uuid.uuid4())[:6]}"

print(table_name)

# COMMAND ----------

# MAGIC %md <i18n value="9db69933-359c-46df-88df-523c20aec03e"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's set up our FeatureStoreClient **`fs`**. 
# MAGIC
# MAGIC To create a feature store client, initialize a **`FeatureStoreClient`** object from the **`feature_store`** module.

# COMMAND ----------

# TODO
from databricks import feature_store
 
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %md <i18n value="961b8810-5fce-4b34-b066-75f33a12e4f8"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Extract Features
# MAGIC
# MAGIC In this simple example we want to predict the number of daily deceased using the other information from from day. 
# MAGIC
# MAGIC Before we write to our feature table, we will need to write a feature computation function that separates our features from the label. 
# MAGIC
# MAGIC Fill in the feature computation function below to select only the feature columns, not **`deceased`**.

# COMMAND ----------

# TODO
columns = covid_df.columns
columns.remove('deceased')
@feature_store.feature_table
def select_features(dataframe):
    return dataframe.select(columns)

covid_features_df = select_features(covid_df)
display(covid_features_df)

# COMMAND ----------

# MAGIC %md <i18n value="d3892ba8-77ca-403f-869f-a6e0b0706555"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Create Feature Table
# MAGIC
# MAGIC Now that we have our features ready, complete the cell below to create our feature table.
# MAGIC
# MAGIC Make sure to set the name to the **`table_name`** we defined above.
# MAGIC
# MAGIC **NOTE:** The primary key needs to be defined in a list as follows: ["primary key name"]

# COMMAND ----------

# TODO
fs.create_table(
    name=table_name,
    primary_keys=['index'],
    df=covid_features_df,
    schema=covid_features_df.schema,
    description='feature store for ML10 Lab'
)

# COMMAND ----------

# MAGIC %md <i18n value="a7a6fb53-a77d-465b-b9bd-1671e3afbbd6"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Update Feature Table
# MAGIC
# MAGIC Imagine now that we wanted to add separate columns for the month and day of the date for each entry. 
# MAGIC
# MAGIC Rather than recompute the table with these values, we just want to append these new columns to the existing table. 
# MAGIC
# MAGIC First, let's create columns for the month and day.

# COMMAND ----------

from pyspark.sql.functions import month, dayofmonth

add_df = (covid_features_df
  .select("date", "index")
  .withColumn("month", month("date"))
  .withColumn("day", dayofmonth("date"))
)

display(add_df)

# COMMAND ----------

# MAGIC %md <i18n value="c040ac8a-f24d-4951-82bc-af1f509a90c2"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now we want to add this information to our feature table using **`write_table`**. 
# MAGIC
# MAGIC **NOTE:** Remember, we can use either **`"overwrite"`** or **`"merge"`** mode. Which one should we use here?

# COMMAND ----------

# TODO
fs.write_table(
    name=table_name,
    df=add_df,
    mode='merge'
)

# COMMAND ----------

# MAGIC %md <i18n value="67d3eb45-0083-4a4c-a391-92ce31628b42"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now try using **`fs.read_table`**, specifying the **`table_name`** to see our updated feature table.

# COMMAND ----------

# TODO
updated_df = fs.read_table(table_name)

display(updated_df)

# COMMAND ----------

# MAGIC %md <i18n value="529e3acd-adc0-45e3-b2a7-eed04a3911b2"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Training 
# MAGIC
# MAGIC Now that we have our feature table, we are ready to use it for model training. We'll need our target variable **`deceased`** in addition to our features, so let's get that first.

# COMMAND ----------

target_df = covid_df.select(["index", "deceased"])

display(target_df)

# COMMAND ----------

# MAGIC %md <i18n value="386a9f38-3c89-41ba-9ce2-3645aa727411"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now let's create our training and test datasets.

# COMMAND ----------

from sklearn.model_selection import train_test_split

def load_data(table_name, lookup_key):
    model_feature_lookups = [feature_store.FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

    # fs.create_training_set will look up features in model_feature_lookups with matched key from inference_data_df
    training_set = fs.create_training_set(target_df, model_feature_lookups, label="deceased", exclude_columns=["index","date"])
    training_pd = training_set.load_df().toPandas()

    # Create train and test datasets
    X = training_pd.drop("deceased", axis=1)
    y = training_pd["deceased"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, training_set

X_train, X_test, y_train, y_test, training_set = load_data(table_name, "index")
X_train.head()

# COMMAND ----------

# MAGIC %md <i18n value="3d1f0e1f-5ead-4f3d-81d5-d400231d0e43"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now we can train a model and register it to the feature store.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
try:
    client.delete_registered_model(f"feature_store_covid_{DA.cleaned_username}") # Deleting model if already created
except:
    None

# COMMAND ----------

import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

def train_model(table_name):
    X_train, X_test, y_train, y_test, training_set = load_data(table_name, "index")

    ## fit and log model
    with mlflow.start_run() as run:

        rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("r2", r2_score(y_test, y_pred))

        fs.log_model(
            model=rf,
            artifact_path="feature-store-model",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=f"feature_store_covid_{DA.cleaned_username}",
            input_example=X_train[:5],
            signature=infer_signature(X_train, y_train)
        )
    
train_model(table_name)

# COMMAND ----------

# MAGIC %md <i18n value="1836dc20-0e77-467e-af8f-33dcfdac7209"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now we have a trained model! Check the Feature Store UI to see that our model is now there. Can you tell which features this model uses from that table and which we excluded?
# MAGIC
# MAGIC Finally, let's apply the model.

# COMMAND ----------

## For sake of simplicity, we will just predict on the same inference_data_df
batch_input_df = target_df.drop("deceased") # Exclude true label
predictions_df = fs.score_batch(f"models:/feature_store_covid_{DA.cleaned_username}/1", 
                                  batch_input_df, result_type="double")
display(predictions_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
