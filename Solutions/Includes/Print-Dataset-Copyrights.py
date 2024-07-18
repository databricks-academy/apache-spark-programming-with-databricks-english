# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

lesson_config.create_schema = False

DA = DBAcademyHelper(course_config, lesson_config)
DA.reset_lesson()
DA.init()
DA.conclude_setup()

# COMMAND ----------

DA.print_copyrights(mappings={
    "COVID/": "coronavirusdataset/coronavirusdataset-readme.md",
})

