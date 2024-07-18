# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="fd2d84ac-6a17-44c2-bb92-18b0c7fef797"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Delta Review
# MAGIC
# MAGIC There are a few key operations necessary to understand and make use of <a href="https://docs.delta.io/latest/quick-start.html#create-a-table" target="_blank">Delta Lake</a>.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC - Create a Delta Table
# MAGIC - Read data from your Delta Table
# MAGIC - Update data in your Delta Table
# MAGIC - Access previous versions of your Delta Table using <a href="https://databricks.com/blog/2019/02/04/introducing-delta-time-travel-for-large-scale-data-lakes.html" target="_blank">time travel</a>
# MAGIC - <a href="https://databricks.com/blog/2019/08/21/diving-into-delta-lake-unpacking-the-transaction-log.html" target="_blank">Understand the Transaction Log</a>
# MAGIC
# MAGIC In this notebook we will be using the SF Airbnb rental dataset from <a href="http://insideairbnb.com/get-the-data.html" target="_blank">Inside Airbnb</a>.

# COMMAND ----------

# MAGIC %md <i18n value="68fcecd4-2280-411c-94c1-3e111683c6a3"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ###Why Delta Lake?<br><br>
# MAGIC
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://user-images.githubusercontent.com/20408077/87175470-4d8e1580-c29e-11ea-8f33-0ee14348a2c1.png" width="500"/>
# MAGIC </div>
# MAGIC
# MAGIC At a glance, Delta Lake is an open source storage layer that brings both **reliability and performance** to data lakes. Delta Lake provides ACID transactions, scalable metadata handling, and unifies streaming and batch data processing. 
# MAGIC
# MAGIC Delta Lake runs on top of your existing data lake and is fully compatible with Apache Spark APIs. <a href="https://docs.databricks.com/delta/delta-intro.html" target="_blank">For more information </a>

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="8ce92b68-6e6c-4fd0-8d3c-a57f27e5bdd9"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ###Creating a Delta Table
# MAGIC First we need to read the Airbnb dataset as a Spark DataFrame

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/"
airbnb_df = spark.read.format("parquet").load(file_path)

display(airbnb_df)

# COMMAND ----------

# MAGIC %md <i18n value="c100b529-ac6b-4540-a3ff-4afa63577eee"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC The cell below converts the data to a Delta table using the schema provided by the Spark DataFrame.

# COMMAND ----------

# Converting Spark DataFrame to Delta Table
dbutils.fs.rm(DA.paths.working_dir, True)
airbnb_df.write.format("delta").mode("overwrite").save(DA.paths.working_dir)

# COMMAND ----------

# MAGIC %md <i18n value="090a31f6-1082-44cf-8e2a-6c659ea796ea"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC A Delta directory can also be registered as a table in the metastore.

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {DA.cleaned_username}")
spark.sql(f"USE {DA.cleaned_username}")

airbnb_df.write.format("delta").mode("overwrite").saveAsTable("delta_review")

# COMMAND ----------

DA.cleaned_username

# COMMAND ----------

# MAGIC %md <i18n value="732577c2-095d-4278-8466-74e494a9c1bd"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Delta supports partitioning. Partitioning puts data with the same value for the partitioned column into its own directory. Operations with a filter on the partitioned column will only read directories that match the filter. This optimization is called partition pruning. Choose partition columns based in the patterns in your data, this dataset for example might benefit if partitioned by neighborhood.

# COMMAND ----------

airbnb_df.write.format("delta").mode("overwrite").partitionBy("neighbourhood_cleansed").option("overwriteSchema", "true").save(DA.paths.working_dir)

# COMMAND ----------

# MAGIC %md <i18n value="e9ce863b-5761-4676-ae0b-95f3f5f027f6"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ###Understanding the <a href="https://databricks.com/blog/2019/08/21/diving-into-delta-lake-unpacking-the-transaction-log.html" target="_blank">Transaction Log </a>
# MAGIC Let's take a look at the Delta Transaction Log. We can see how Delta stores the different neighborhood partitions in separate files. Additionally, we can also see a directory called _delta_log.

# COMMAND ----------

display(dbutils.fs.ls(DA.paths.working_dir))

# COMMAND ----------

# MAGIC %md <i18n value="ac970bba-1cf6-4aa3-91bb-74a797496eef"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC <div style="img align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://user-images.githubusercontent.com/20408077/87174138-609fe600-c29c-11ea-90cc-84df0c1357f1.png" width="500"/>
# MAGIC </div>
# MAGIC
# MAGIC When a user creates a Delta Lake table, that table’s transaction log is automatically created in the _delta_log subdirectory. As he or she makes changes to that table, those changes are recorded as ordered, atomic commits in the transaction log. Each commit is written out as a JSON file, starting with 000000.json. Additional changes to the table generate more JSON files.

# COMMAND ----------

display(dbutils.fs.ls(f"{DA.paths.working_dir}/_delta_log/"))

# COMMAND ----------

# MAGIC %md <i18n value="2905b874-373b-493d-9084-8ff4f7583ccc"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Next, let's take a look at a Transaction Log File.
# MAGIC
# MAGIC The <a href="https://docs.databricks.com/delta/delta-utility.html" target="_blank">four columns</a> each represent a different part of the very first commit to the Delta Table where the table was created.<br><br>
# MAGIC
# MAGIC - The add column has statistics about the DataFrame as a whole and individual columns.
# MAGIC - The commitInfo column has useful information about what the operation was (WRITE or READ) and who executed the operation.
# MAGIC - The metaData column contains information about the column schema.
# MAGIC - The protocol version contains information about the minimum Delta version necessary to either write or read to this Delta Table.

# COMMAND ----------

display(spark.read.json(f"{DA.paths.working_dir}/_delta_log/00000000000000000000.json"))

# COMMAND ----------

# MAGIC %md <i18n value="8f79d1df-d777-4364-9783-b52bc0eed81a"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC The second transaction log has 39 rows. This includes metadata for each partition.

# COMMAND ----------

display(spark.read.json(f"{DA.paths.working_dir}/_delta_log/00000000000000000001.json"))

# COMMAND ----------

# MAGIC %md <i18n value="18500df8-b905-4f24-957c-58040920d554"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Finally, let's take a look at the files inside one of the Neighborhood partitions. The file inside corresponds to the partition commit (file 01) in the _delta_log directory.

# COMMAND ----------

display(dbutils.fs.ls(f"{DA.paths.working_dir}/neighbourhood_cleansed=Bayview/"))

# COMMAND ----------

# MAGIC %md <i18n value="9f817cd0-87ec-457b-8776-3fc275521868"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Reading data from your Delta table

# COMMAND ----------

df = spark.read.format("delta").load(DA.paths.working_dir)
display(df)

# COMMAND ----------

# MAGIC %md <i18n value="faba817b-7cbf-49d4-a32c-36a40f582021"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC #Updating your Delta Table
# MAGIC
# MAGIC Let's filter for rows where the host is a superhost.

# COMMAND ----------

df_update = airbnb_df.filter(airbnb_df["host_is_superhost"] == "t")
display(df_update)

# COMMAND ----------

df_update.write.format("delta").mode("overwrite").save(DA.paths.working_dir)

# COMMAND ----------

df = spark.read.format("delta").load(DA.paths.working_dir)
display(df)

# COMMAND ----------

# MAGIC %md <i18n value="e4cafdf4-a346-4729-81a6-fdea70f4929a"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's look at the files in the Bayview partition post-update. Remember, the different files in this directory are snapshots of your DataFrame corresponding to different commits.

# COMMAND ----------

display(dbutils.fs.ls(f"{DA.paths.working_dir}/neighbourhood_cleansed=Bayview/"))

# COMMAND ----------

# MAGIC %md <i18n value="25ca7489-8077-4b23-96af-8d801982367c"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC #Delta Time Travel

# COMMAND ----------

# MAGIC %md <i18n value="c6f2e771-502d-46ed-b8d4-b02e3e4f4134"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Oops, actually we need the entire dataset! You can access a previous version of your Delta Table using <a href="https://databricks.com/blog/2019/02/04/introducing-delta-time-travel-for-large-scale-data-lakes.html" target="_blank">Delta Time Travel</a>. Use the following two cells to access your version history. Delta Lake will keep a 30 day version history by default, though it can maintain that history for longer if needed.

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS train_delta;
# MAGIC CREATE TABLE train_delta USING DELTA LOCATION '${DA.paths.working_dir}'

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY train_delta

# COMMAND ----------

# MAGIC %md <i18n value="61faa23f-d940-479c-95fe-5aba72c29ddf"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Using the **`versionAsOf`** option allows you to easily access previous versions of our Delta Table.

# COMMAND ----------

df = spark.read.format("delta").option("versionAsOf", 0).load(DA.paths.working_dir)
display(df)

# COMMAND ----------

# MAGIC %md <i18n value="5664be65-8fd2-4746-8065-35ee8b563797"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC You can also access older versions using a timestamp.
# MAGIC
# MAGIC Replace the timestamp string with the information from your version history. Note that you can use a date without the time information if necessary.

# COMMAND ----------

# Use your own timestamp 
# time_stamp_string = "FILL_IN"

# OR programatically get the first verion's timestamp value
time_stamp_string = str(spark.sql("DESCRIBE HISTORY train_delta").collect()[-1]["timestamp"])

df = spark.read.format("delta").option("timestampAsOf", time_stamp_string).load(DA.paths.working_dir)
display(df)

# COMMAND ----------

# MAGIC %md <i18n value="6cbe5204-fe27-438a-af54-87492c2563b5"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now that we're happy with our Delta Table, we can clean up our directory using **`VACUUM`**. Vacuum accepts a retention period in hours as an input.

# COMMAND ----------

# MAGIC %md <i18n value="4da7827c-b312-4b66-8466-f0245f3787f4"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Uh-oh, our code doesn't run! By default, to prevent accidentally vacuuming recent commits, Delta Lake will not let users vacuum a period under 7 days or 168 hours. Once vacuumed, you cannot return to a prior commit through time travel, only your most recent Delta Table will be saved.
# MAGIC
# MAGIC Try changing the vacuum parameter to different values.

# COMMAND ----------

# from delta.tables import DeltaTable

# delta_table = DeltaTable.forPath(spark, DA.paths.working_dir)
# delta_table.vacuum(0)

# COMMAND ----------

# MAGIC %md <i18n value="1150e320-5ed2-4a38-b39f-b63157bca94f"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC We can workaround this by setting a spark configuration that will bypass the default retention period check.

# COMMAND ----------

from delta.tables import DeltaTable

spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
delta_table = DeltaTable.forPath(spark, DA.paths.working_dir)
delta_table.vacuum(0)

# COMMAND ----------

# MAGIC %md <i18n value="b845b2ea-2c11-4d6e-b083-d5908b65d313"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's take a look at our Delta Table files now. After vacuuming, the directory only holds the partition of our most recent Delta Table commit.

# COMMAND ----------

display(dbutils.fs.ls(f"{DA.paths.working_dir}/neighbourhood_cleansed=Bayview/"))

# COMMAND ----------

# MAGIC %md <i18n value="a7bcdad3-affb-4b00-b791-07c14f5e59d5"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Since vacuuming deletes files referenced by the Delta Table, we can no longer access past versions. The code below should throw an error.

# COMMAND ----------

# df = spark.read.format("delta").option("versionAsOf", 0).load(DA.paths.working_dir)
# display(df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
