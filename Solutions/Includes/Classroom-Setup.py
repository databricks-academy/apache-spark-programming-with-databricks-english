# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

import re

DA = DBAcademyHelper(course_config, lesson_config)
DA.reset_lesson()
DA.init()

DA.cleaned_username = dbgems.clean_string(DA.username)
DA.init_mlflow_as_job()

DA.conclude_setup()

# COMMAND ----------

#############################################
# Legacy testing functions
#############################################

# Test results dict to store results
testResults = dict()

# Hash a string value
def toHash(value):
  from pyspark.sql.functions import hash
  from pyspark.sql.functions import abs
  values = [(value,)]
  return spark.createDataFrame(values, ["value"]).select(abs(hash("value")).cast("int")).first()[0]

# Clear the testResults map
def clearYourResults(passedOnly = True):
  whats = list(testResults.keys())
  for what in whats:
    passed = testResults[what][0]
    if passed or passedOnly == False : del testResults[what]

# Validate DataFrame schema
def validateYourSchema(what, df, expColumnName, expColumnType = None):
  label = "{}:{}".format(expColumnName, expColumnType)
  key = "{} contains {}".format(what, label)

  try:
    actualType = df.schema[expColumnName].dataType.typeName()
    
    if expColumnType == None: 
      testResults[key] = (True, "validated")
      print("""{}: validated""".format(key))
    elif actualType == expColumnType:
      testResults[key] = (True, "validated")
      print("""{}: validated""".format(key))
    else:
      answerStr = "{}:{}".format(expColumnName, actualType)
      testResults[key] = (False, answerStr)
      print("""{}: NOT matching ({})""".format(key, answerStr))
  except:
      testResults[what] = (False, "-not found-")
      print("{}: NOT found".format(key))

# Validate an answer
def validateYourAnswer(what, expectedHash, answer):
  # Convert the value to string, remove new lines and carriage returns and then escape quotes
  if (answer == None): answerStr = "null"
  elif (answer is True): answerStr = "true"
  elif (answer is False): answerStr = "false"
  else: answerStr = str(answer)

  hashValue = toHash(answerStr)
  
  if (hashValue == expectedHash):
    testResults[what] = (True, answerStr)
    print("""{} was correct, your answer: {}""".format(what, answerStr))
  else:
    testResults[what] = (False, answerStr)
    print("""{} was NOT correct, your answer: {}""".format(what, answerStr))

# Summarize results in the testResults dict
def summarizeYourResults():
  html = """<html><body><div style="font-weight:bold; font-size:larger; border-bottom: 1px solid #f0f0f0">Your Answers</div><table style='margin:0'>"""

  whats = list(testResults.keys())
  whats.sort()
  for what in whats:
    passed = testResults[what][0]
    answer = testResults[what][1]
    color = "green" if (passed) else "red" 
    passFail = "passed" if (passed) else "FAILED" 
    html += """<tr style='font-size:larger; white-space:pre'>
                  <td>{}:&nbsp;&nbsp;</td>
                  <td style="color:{}; text-align:center; font-weight:bold">{}</td>
                  <td style="white-space:pre; font-family: monospace">&nbsp;&nbsp;{}</td>
                </tr>""".format(what, color, passFail, answer)
  html += "</table></body></html>"
  displayHTML(html)

# Log test results to a file
def logYourTest(path, name, value):
  value = float(value)
  if "\"" in path: raise AssertionError("The name cannot contain quotes.")
  
  dbutils.fs.mkdirs(path)

  csv = """ "{}","{}" """.format(name, value).strip()
  file = "{}/{}.csv".format(path, name).replace(" ", "-").lower()
  dbutils.fs.put(file, csv, True)

# Load test results from log file
def loadYourTestResults(path):
  from pyspark.sql.functions import col
  return spark.read.schema("name string, value double").csv(path)

# Load test results from log file into a dict
def loadYourTestMap(path):
  rows = loadYourTestResults(path).collect()
  
  map = dict()
  for row in rows:
    map[row["name"]] = row["value"]
  
  return map

# ****************************************************************************
# Placeholder variables for coding challenge type specification
# ****************************************************************************
# class FILL_IN:
#   from pyspark.sql.types import Row, StructType
#   VALUE = None
#   LIST = []
#   SCHEMA = StructType([])
#   ROW = Row()
#   INT = 0
#   DATAFRAME = sqlContext.createDataFrame(sc.emptyRDD(), StructType([]))

# displayHTML("Defining courseware-specific utility methods...")

