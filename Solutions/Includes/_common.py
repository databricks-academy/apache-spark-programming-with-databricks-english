# Databricks notebook source
# MAGIC %run ./_libraries

# COMMAND ----------

# MAGIC %run ./_dataset_index

# COMMAND ----------

from dbacademy import dbgems
from dbacademy.dbhelper import DBAcademyHelper, Paths, CourseConfig, LessonConfig

course_config = CourseConfig(course_code = "sml",
                             course_name = "scalable-machine-learning-with-apache-spark",
                             data_source_name = "scalable-machine-learning-with-apache-spark",
                             data_source_version = "v02",
                             install_min_time = "2 min",
                             install_max_time = "5 min",
                             remote_files = remote_files,
                             supported_dbrs = ["11.3.x-cpu-ml-scala2.12"],
                             expected_dbrs = "11.3.x-cpu-ml-scala2.12")

lesson_config = LessonConfig(name = None,
                             create_schema = True,
                             create_catalog = False,
                             requires_uc = False,
                             installing_datasets = True,
                             enable_streaming_support = False)

