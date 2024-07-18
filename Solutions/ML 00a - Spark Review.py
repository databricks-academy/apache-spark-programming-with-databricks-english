# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="1108b110-983d-4034-9156-6b95c04dc62c"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Spark Review
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Create a Spark DataFrame
# MAGIC  - Analyze the Spark UI
# MAGIC  - Cache data
# MAGIC  - Go between Pandas and Spark DataFrames

# COMMAND ----------

# MAGIC %md <i18n value="890d085b-9058-49a7-aa15-bff3649b9e05"/>
# MAGIC
# MAGIC ![](https://files.training.databricks.com/images/sparkcluster.png)

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="df081f79-6894-4174-a554-fa0943599408"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Spark DataFrame

# COMMAND ----------

from pyspark.sql.functions import col, rand

df = (spark.range(1, 1000000)
      .withColumn("id", (col("id") / 1000).cast("integer"))
      .withColumn("v", rand(seed=1)))

# COMMAND ----------

# MAGIC %md <i18n value="a0c6912d-a8d6-449b-a3ab-5ca91c7f9805"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Why were no Spark jobs kicked off above? Well, we didn't have to actually "touch" our data, so Spark didn't need to execute anything across the cluster.

# COMMAND ----------

display(df.sample(.001))

# COMMAND ----------

# MAGIC %md <i18n value="6eadef21-d75c-45ba-8d77-419d1ce0c06c"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Views
# MAGIC
# MAGIC How can I access this in SQL?

# COMMAND ----------

df.createOrReplaceTempView("df_temp")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM df_temp LIMIT 10

# COMMAND ----------

# MAGIC %md <i18n value="2593e6b0-d34b-4086-9fed-c4956575a623"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Count
# MAGIC
# MAGIC Let's see how many records we have.

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md <i18n value="5d00511e-15da-48e7-bd26-e89fbe56632c"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Spark UI
# MAGIC
# MAGIC Open up the Spark UI - what are the shuffle read and shuffle write fields? The command below should give you a clue.

# COMMAND ----------

df.rdd.getNumPartitions()

# COMMAND ----------

# MAGIC %md <i18n value="50330454-0168-4f50-8355-0204632b20ec"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Cache
# MAGIC
# MAGIC For repeated access, it will be much faster if we cache our data.

# COMMAND ----------

df.cache().count()

# COMMAND ----------

# MAGIC %md <i18n value="7dd81880-1575-410c-a168-8ac081a97e9d"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Re-run Count
# MAGIC
# MAGIC Wow! Look at how much faster it is now!

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md <i18n value="ce238b9e-fee4-4644-9469-b7d9910f6243"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Collect Data
# MAGIC
# MAGIC When you pull data back to the driver  (e.g. call **`.collect()`**, **`.toPandas()`**,  etc), you'll need to be careful of how much data you're bringing back. Otherwise, you might get OOM exceptions!
# MAGIC
# MAGIC A best practice is explicitly limit the number of records, unless you know your data set is small, before calling **`.collect()`** or **`.toPandas()`**.

# COMMAND ----------

df.limit(10).toPandas()

# COMMAND ----------

# MAGIC %md <i18n value="279e3325-b121-402b-a2d0-486e1cc26fc0"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## What's new in <a href="https://www.youtube.com/watch?v=l6SuXvhorDY&feature=emb_logo" target="_blank">Spark 3.0</a>
# MAGIC
# MAGIC * <a href="https://www.youtube.com/watch?v=jzrEc4r90N8&feature=emb_logo" target="_blank">Adaptive Query Execution</a>
# MAGIC   * Dynamic query optimization that happens in the middle of your query based on runtime statistics
# MAGIC     * Dynamically coalesce shuffle partitions
# MAGIC     * Dynamically switch join strategies
# MAGIC     * Dynamically optimize skew joins
# MAGIC   * Enable it with: **`spark.sql.adaptive.enabled=true`**
# MAGIC * Dynamic Partition Pruning (DPP)
# MAGIC   * Avoid partition scanning based on the query results of the other query fragments
# MAGIC * Join Hints
# MAGIC * <a href="https://www.youtube.com/watch?v=UZl0pHG-2HA&feature=emb_logo" target="_blank">Improved Pandas UDFs</a>
# MAGIC   * Type Hints
# MAGIC   * Iterators
# MAGIC   * Pandas Function API (mapInPandas, applyInPandas, etc)
# MAGIC * And many more! See the <a href="https://spark.apache.org/docs/latest/api/python/migration_guide/pyspark_2.4_to_3.0.html" target="_blank">migration guide</a> and resources linked above.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
