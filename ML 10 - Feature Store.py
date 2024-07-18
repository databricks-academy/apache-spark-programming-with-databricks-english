# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="b69335d5-86c7-40c5-b430-509a7444dae7"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Feature Store
# MAGIC
# MAGIC The <a href="https://docs.databricks.com/applications/machine-learning/feature-store.html" target="_blank">Databricks Feature Store</a> is a centralized repository of features. It enables feature sharing and discovery across your organization and also ensures that the same feature computation code is used for model training and inference.
# MAGIC
# MAGIC Check out Feature Store Python API documentation <a href="https://docs.databricks.com/dev-tools/api/python/latest/index.html#feature-store-python-api-reference" target="_blank">here</a>.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC  - Build a feature store with the Databricks Feature Store
# MAGIC  - Update feature tables
# MAGIC  - Perform batch scoring

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id, lit, expr, rand
import uuid
from databricks import feature_store
from pyspark.sql.types import StringType, DoubleType
from databricks.feature_store import feature_table, FeatureLookup
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# COMMAND ----------

# MAGIC %md <i18n value="5dcd3e8e-2553-429f-bbe1-aef0bc1ef0ab"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's load in our data and generate a unique ID for each listing. The **`index`** column will serve as the "key" of the feature table and used to lookup features.

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path).coalesce(1).withColumn("index", monotonically_increasing_id())
display(airbnb_df)

# COMMAND ----------

# MAGIC %md <i18n value="a04b29f6-e7a6-4e6a-875f-945edf938e9e"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Create a new database and unique table name (in case you re-run the notebook multiple times)

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {DA.cleaned_username}")
table_name = f"{DA.cleaned_username}.airbnb_" + str(uuid.uuid4())[:6]
print(table_name)

# COMMAND ----------

# MAGIC %md <i18n value="a0712a39-b413-490f-a59e-dbd7f533e9a9"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's start creating a <a href="https://docs.databricks.com/applications/machine-learning/feature-store.html#create-a-feature-table-in-databricks-feature-store" target="_blank">Feature Store Client</a> so we can populate our feature store.

# COMMAND ----------

fs = feature_store.FeatureStoreClient()
# help(fs.create_table)

# COMMAND ----------

# MAGIC %md <i18n value="90998fdb-87ed-4cdd-8844-fbd59ac5631f"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC #### Create Feature Table
# MAGIC
# MAGIC Next, we can create the Feature Table using the **`create_table`** method.
# MAGIC
# MAGIC This method takes a few parameters as inputs:
# MAGIC * **`name`**- A feature table name of the form **`<database_name>.<table_name>`**
# MAGIC * **`primary_keys`**- The primary key(s). If multiple columns are required, specify a list of column names.
# MAGIC * **`df`**- Data to insert into this feature table.  The schema of **`features_df`** will be used as the feature table schema.
# MAGIC * **`schema`**- Feature table schema. Note that either **`schema`** or **`features_df`** must be provided.
# MAGIC * **`description`**- Description of the feature table
# MAGIC * **`partition_columns`**- Column(s) used to partition the feature table.

# COMMAND ----------

## select numeric features and exclude target column "price"
numeric_cols = [x.name for x in airbnb_df.schema.fields if (x.dataType == DoubleType()) and (x.name != "price")]
numeric_features_df = airbnb_df.select(["index"] + numeric_cols)
display(numeric_features_df)

# COMMAND ----------

fs.create_table(
    name=table_name,
    primary_keys=["index"],
    df=numeric_features_df,
    schema=numeric_features_df.schema,
    description="Numeric features of airbnb data"
)

# COMMAND ----------

# MAGIC %md <i18n value="4a7cbb2e-87a2-4ea8-85e6-207ec5e42147"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Alternatively, you can **`create_table`** with schema only (without **`df`**), and populate data to the feature table with **`fs.write_table`**. **`fs.write_table`** supports both **`overwrite`** and **`merge`** modes.
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC ```
# MAGIC fs.create_table(
# MAGIC     name=table_name,
# MAGIC     primary_keys=["index"],
# MAGIC     schema=numeric_features_df.schema,
# MAGIC     description="Original Airbnb data"
# MAGIC )
# MAGIC
# MAGIC fs.write_table(
# MAGIC     name=table_name,
# MAGIC     df=numeric_features_df,
# MAGIC     mode="overwrite"
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md <i18n value="44586907-302a-4916-93f6-e92210619c6f"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now let's explore the UI and see how it tracks the tables that we created. Navigate to the UI by first ensuring that you are in the Machine Learning workspace, and then clicking on the Feature Store icon on the bottom-left of the navigation bar.
# MAGIC
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/FS_Nav.png" alt="step12" width="150"/>

# COMMAND ----------

# MAGIC %md <i18n value="cf0ad0d0-8456-471b-935c-8a34a836fca7"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC In this screenshot, we can see the feature table that we created.
# MAGIC <br>
# MAGIC <br>
# MAGIC Note the section of **`Producers`**. This section indicates which notebook produces the feature table.
# MAGIC <br>
# MAGIC <br>
# MAGIC <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/mlflow/fs_details+(1).png" alt="step12" width="1000"/>

# COMMAND ----------

# MAGIC %md <i18n value="b07da702-485e-44b8-bd00-f0330c8b7657"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC We can also look at the metadata of the feature store via the FeatureStore client by using **`get_table()`**.

# COMMAND ----------

fs.get_table(table_name).path_data_sources

# COMMAND ----------

fs.get_table(table_name).description

# COMMAND ----------

# MAGIC %md <i18n value="1df7795c-1a07-47ae-92a8-1c5f7aec75ae"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Train a model with feature store

# COMMAND ----------

# MAGIC %md <i18n value="bcbf72b7-a013-40fd-bf55-a2b179a7728e"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC The prediction target **`price`** should NOT BE included as a feature in the registered feature table.
# MAGIC
# MAGIC Further, there may be other information that _can_ be supplied at inference time, but does not make sense to consider a feature to _look up_. 
# MAGIC
# MAGIC In this (fictional) example, we made up a feature **`score_diff_from_last_month`**. It is a feature generated at inference time and used in training as well.

# COMMAND ----------

## inference data -- index (key), price (target) and a online feature (make up a fictional column - diff of review score in a month) 
inference_data_df = airbnb_df.select("index", "price", (rand() * 0.5-0.25).alias("score_diff_from_last_month"))
display(inference_data_df)

# COMMAND ----------

# MAGIC %md <i18n value="b8301fa9-27bd-4d3b-bf13-9ab784205d81"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Build a training dataset that will use the indicated "key" to lookup features from the feature table and also the online feature **`score_diff_from_last_month`**. We will use <a href="https://docs.databricks.com/dev-tools/api/python/latest/index.html" target="_blank">FeatureLookup</a> and if you specify no features, it will return all of them except the primary key.

# COMMAND ----------

def load_data(table_name, lookup_key):
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

    # fs.create_training_set will look up features in model_feature_lookups with matched key from inference_data_df
    training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label="price", exclude_columns="index")
    training_pd = training_set.load_df().toPandas()

    # Create train and test datasets
    X = training_pd.drop("price", axis=1)
    y = training_pd["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, training_set

X_train, X_test, y_train, y_test, training_set = load_data(table_name, "index")
X_train.head()

# COMMAND ----------

# MAGIC %md <i18n value="eae1aa4a-f770-4173-9502-cb946e6949d2"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Train a **RandomForestRegressor** model and log the model with the Feature Store. An MLflow run is started to track the autologged components as well as the Feature Store logged model. However, we will disable the MLflow model autologging as the model will be explicitly logged via the Feature Store.
# MAGIC
# MAGIC NOTE: This is an overly simplistic example, used solely for demo purposes.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()

try:
    client.delete_registered_model(f"feature_store_airbnb_{DA.cleaned_username}") # Deleting model if already created
except:
    None

# COMMAND ----------

# Disable model autologging and instead log explicitly via the FeatureStore
mlflow.sklearn.autolog(log_models=False)

def train_model(X_train, X_test, y_train, y_test, training_set, fs):
    ## fit and log model
    with mlflow.start_run() as run:

        rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))

        fs.log_model(
            model=rf,
            artifact_path="feature-store-model",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=f"feature_store_airbnb_{DA.cleaned_username}",
            input_example=X_train[:5],
            signature=infer_signature(X_train, y_train)
        )

train_model(X_train, X_test, y_train, y_test, training_set, fs)

# COMMAND ----------

# MAGIC %md <i18n value="40b7718f-101c-4ac4-8639-545b8ef6d932"/>
# MAGIC
# MAGIC
# MAGIC  
# MAGIC Now, view the run from MLflow UI. You can find the model parameters logged with MLflow autolog.
# MAGIC <br>
# MAGIC <br>
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/fs_log_model_mlflow_params.png" alt="step12" width="1000"/>

# COMMAND ----------

# MAGIC %md <i18n value="f03314dc-1ade-4bd8-958f-ddf04ac1bb13"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Notice the saved model artifact **feature_store_model** : packaged feature store model that can be used directly for batch scoring - logged from **`fs.log_model`**
# MAGIC <br>
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/301/updated_feature_store_9_1.png" alt="step12" width="1000"/>

# COMMAND ----------

# MAGIC %md <i18n value="acd4d5a4-c4ed-4695-a911-5fd88dcfa513"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC The **`feature_store_model`** is registered in the MLflow model registry as well. You can find it in **`Models`** page. It is also logged at the feature store page, indicating which features in the feature table are used for the model. We will exam feature/model lineage through the UI together later.

# COMMAND ----------

# MAGIC %md <i18n value="921dc6c9-b9ed-43c7-86ff-608791a11367"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Feature Store Batch Scoring
# MAGIC
# MAGIC Apply a feature store registered MLflow model to features with **`score_batch`**. Input data only need the key column **`index`** and online feature **`score_diff_from_last_month`**. Everything else is looked up.

# COMMAND ----------

## For sake of simplicity, we will just predict on the same inference_data_df
batch_input_df = inference_data_df.drop("price") # Exclude true label
predictions_df = fs.score_batch(f"models:/feature_store_airbnb_{DA.cleaned_username}/1", 
                                  batch_input_df, result_type="double")
display(predictions_df)

# COMMAND ----------

# MAGIC %md <i18n value="fa42d4d3-a6a6-4205-b799-032154d1d8a3"/>
# MAGIC
# MAGIC
# MAGIC  
# MAGIC ### Overwrite feature table
# MAGIC Lastly, we'll condense some of the review columns and update the feature table: we'll do this by calculating the average review score for each listing.

# COMMAND ----------

## select numeric features and aggregate the review scores
review_columns = ["review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", 
                 "review_scores_communication", "review_scores_location", "review_scores_value"]

condensed_review_df = (airbnb_df
                       .select(["index"] + numeric_cols)
                       .withColumn("average_review_score", expr("+".join(review_columns)) / lit(len(review_columns)))
                       .drop(*review_columns)
                      )
             
display(condensed_review_df)

# COMMAND ----------

# MAGIC %md <i18n value="da3ee1df-391c-4f26-99d0-82937e91a40a"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's now drop those features using **`overwrite`**.

# COMMAND ----------

fs.write_table(
    name=table_name,
    df=condensed_review_df,
    mode="overwrite"
)

# COMMAND ----------

# MAGIC %md <i18n value="ae45b580-e79e-4f54-85a0-1274cb5f5c5f"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Explore the feature permission, lineage and freshness from Feature Store UI

# COMMAND ----------

# MAGIC %md <i18n value="5d4d8425-b9b7-4e47-8856-91e1142e9c47"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC On the UI, we can see that:
# MAGIC - A new column has been added to the feature list
# MAGIC - Columns that we deleted are also still present. However, the deleted features will have **`null`** as their values when we read in the table
# MAGIC - The "Models" column are populated, listing models use the features from the table
# MAGIC - The last column **`Notebooks`** are populated. This column indicates which notebooks consume the features in the feature table
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/feature_store_consumers.png" alt="step12" width="800"/>

# COMMAND ----------

# MAGIC %md <i18n value="884ff3ff-f965-4c37-8cff-f6a1600ee0b6"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now, let's read in the feature data from the Feature Store. By default, **`fs.read_table()`** reads in the latest version of the feature table. To read in the specific version of feature table, you can optionally specify the argument **`as_of_delta_timestamp`** by passing a date in a timestamp format or string.
# MAGIC
# MAGIC
# MAGIC Note that the values of the deleted columns have been replaced by **`null`**.

# COMMAND ----------

# Displays most recent table
display(fs.read_table(name=table_name))

# COMMAND ----------

# MAGIC %md <i18n value="4148328d-4046-4251-b4db-f9e427b2e0f9"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC If you need to use the features for real-time serving, you can publish your features to an <a href="https://docs.databricks.com/applications/machine-learning/feature-store.html#publish-features-to-an-online-feature-store" target="_blank">online store</a>.
# MAGIC
# MAGIC We can perform control who has permissions to the feature table on the UI.
# MAGIC
# MAGIC To delete the table, use the **`delete`** button on the UI. **You need to delete the delta table from database as well.**
# MAGIC <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/mlflow/fs_permissions+(1).png" alt="step12" width="700"/>

# COMMAND ----------

# MAGIC %md <i18n value="81e53dea-dc51-418c-b366-eed3a9c4ce2f"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Retrain a new model with the new average_review_score feature

# COMMAND ----------

def load_data(table_name, lookup_key):
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

    # fs.create_training_set will look up features in model_feature_lookups with matched key from inference_data_df
    training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label="price", exclude_columns="index")
    training_pd = training_set.load_df().drop(*review_columns).toPandas()  #remove all those null columns, should now have the new average_review_score in it

    # Create train and test datasets
    X = training_pd.drop("price", axis=1)
    y = training_pd["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, training_set

X_train, X_test, y_train, y_test, training_set = load_data(table_name, "index")
X_train.head()

# COMMAND ----------

# MAGIC %md <i18n value="94873d7f-3bb9-4d5f-a414-c24480a84f3b"/>
# MAGIC
# MAGIC
# MAGIC  
# MAGIC Build a training dataset that will use the indicated `key` to lookup features.

# COMMAND ----------

def train_model(X_train, X_test, y_train, y_test, training_set, fs):
    ## fit and log model
    with mlflow.start_run() as run:

        rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))

        fs.log_model(
            model=rf,
            artifact_path="feature-store-model",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=f"feature_store_airbnb_{DA.cleaned_username}",
            input_example=X_train[:5],
            signature=infer_signature(X_train, y_train)
        )

train_model(X_train, X_test, y_train, y_test, training_set, fs)

# COMMAND ----------

# MAGIC %md <i18n value="b0ffd91d-c73f-4f86-a02a-43ffdc73460c"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Feature Store Batch Scoring
# MAGIC
# MAGIC Apply the feature store registered MLflow model version2 to features with **`score_batch`**.

# COMMAND ----------

## For sake of simplicity, we will just predict on the same inference_data_df
batch_input_df = inference_data_df.drop("price") # Exclude true label
predictions_df = fs.score_batch(f"models:/feature_store_airbnb_{DA.cleaned_username}/2", #notice we are using version2
                                  batch_input_df, result_type="double")
display(predictions_df)

# COMMAND ----------

# MAGIC %md <i18n value="67471f1c-0dc0-445f-ae6a-beafb3508a16"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC On the UI, we can see that:
# MAGIC - The model version 2 is using the newly created feature of average_review_score
# MAGIC - Columns that we deleted are also still present. However, the deleted features will have **`null`** as their values when we read in the table
# MAGIC - The "Models" column are populated, listing versions of models use the versions of features from the table
# MAGIC - The last column **`Notebooks`** are populated. This column indicates which notebooks consume the features in the feature table
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/feature_store_consumers_2.png" alt="step12" width="1000"/>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
