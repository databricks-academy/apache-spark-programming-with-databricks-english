# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="94727771-3f7d-41a7-bcbd-774b1fc5837c"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Distributed K-Means
# MAGIC
# MAGIC In this notebook, we are going to use K-Means to cluster our data. We will be using the Iris dataset, which has labels (the type of iris), but we will only use the labels to evaluate the model, not to train it. 
# MAGIC
# MAGIC At the end, we will look at how it is implemented in the distributed setting.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Build a K-Means model
# MAGIC  - Analyze the computation and communication of K-Means in a distributed setting

# COMMAND ----------

from sklearn.datasets import load_iris
import pandas as pd

# Load in a Dataset from sklearn and convert to a Spark DataFrame
iris = load_iris()
iris_pd = pd.concat([pd.DataFrame(iris.data, columns=iris.feature_names), pd.DataFrame(iris.target, columns=["label"])], axis=1)
iris_df = spark.createDataFrame(iris_pd)
display(iris_df)

# COMMAND ----------

# MAGIC %md <i18n value="efd06e75-816c-4ab5-84b5-dd1da377fa01"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Notice that we have four values as "features".  We'll reduce those down to two values (for visualization purposes) and convert them to a **`DenseVector`**.  To do that we'll use the **`VectorAssembler`**.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(inputCols=["sepal length (cm)", "sepal width (cm)"], outputCol="features")
iris_two_features_df = vec_assembler.transform(iris_df)
display(iris_two_features_df)

# COMMAND ----------

from pyspark.ml.clustering import KMeans

kmeans = KMeans(k=3, seed=221, maxIter=20)

#  Call fit on the estimator and pass in iris_two_features_df
model = kmeans.fit(iris_two_features_df)

# Obtain the clusterCenters from the KMeansModel
centers = model.clusterCenters()

# Use the model to transform the DataFrame by adding cluster predictions
transformed_df = model.transform(iris_two_features_df)

print(centers)

# COMMAND ----------

model_centers = []
iterations = [0, 2, 4, 7, 10, 20]
for i in iterations:
    kmeans = KMeans(k=3, seed=221, maxIter=i)
    model = kmeans.fit(iris_two_features_df)
    model_centers.append(model.clusterCenters())   

# COMMAND ----------

print("model_centers:")
for centroids in model_centers:
    print(centroids)

# COMMAND ----------

# MAGIC %md <i18n value="840acc4b-58f7-439d-afe7-5a70d5718dc1"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's visualize how our clustering performed against the true labels of our data.
# MAGIC
# MAGIC Remember: K-means doesn't use the true labels when training, but we can use them to evaluate. 
# MAGIC
# MAGIC Here, the star marks the cluster center.

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def prepare_subplot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor="#999999", gridWidth=1.0, subplots=(1, 1)):
    """Template for generating the plot layout."""
    fig, ax_list = plt.subplots(subplots[0], subplots[1], figsize=figsize, facecolor="white", 
                               edgecolor="white")
    if not isinstance(ax_list, np.ndarray):
        ax_list = np.array([ax_list])
    
    for ax in ax_list.flatten():
        ax.axes.tick_params(labelcolor="#999999", labelsize="10")
        for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
            axis.set_ticks_position("none")
            axis.set_ticks(ticks)
            axis.label.set_color("#999999")
            if hideLabels: axis.set_ticklabels([])
        ax.grid(color=gridColor, linewidth=gridWidth, linestyle="-")
        map(lambda position: ax.spines[position].set_visible(False), ["bottom", "top", "left", "right"])
        
    if ax_list.size == 1:
        ax_list = ax_list[0]  # Just return a single axes object for a regular plot
    return fig, ax_list

# COMMAND ----------

data = iris_two_features_df.select("features", "label").collect()
features, labels = zip(*data)

x, y = zip(*features)
centers = model_centers[5]
centroid_x, centroid_y = zip(*centers)
color_map = "Set1"

fig, ax = prepare_subplot(np.arange(-1, 1.1, .4), np.arange(-1, 1.1, .4), figsize=(8,6))
plt.scatter(x, y, s=14**2, c=labels, edgecolors="#8cbfd0", alpha=0.80, cmap=color_map)
plt.scatter(centroid_x, centroid_y, s=22**2, marker="*", c="yellow")
cmap = cm.get_cmap(color_map)

color_index = [.5, .99, .0]
for i, (x,y) in enumerate(centers):
    print(cmap(color_index[i]))
    for size in [.10, .20, .30, .40, .50]:
        circle1=plt.Circle((x,y), size, color=cmap(color_index[i]), alpha=.10, linewidth=2)
        ax.add_artist(circle1)

ax.set_xlabel("Sepal Length"), ax.set_ylabel("Sepal Width")
fig

# COMMAND ----------

# MAGIC %md <i18n value="b5b5d89a-1595-4e0c-99a1-54209435cf81"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC In addition to seeing the overlay of the clusters at each iteration, we can see how the cluster centers moved with each iteration (and what our results would have looked like if we used fewer iterations).

# COMMAND ----------

x, y = zip(*features)

old_centroid_x, old_centroid_y = None, None

fig, ax_list = prepare_subplot(np.arange(-1, 1.1, .4), np.arange(-1, 1.1, .4), figsize=(11, 15),
                             subplots=(3, 2))
ax_list = ax_list.flatten()

for i,ax in enumerate(ax_list[:]):
    ax.set_title("K-means for {0} iterations".format(iterations[i]), color="#999999")
    centroids = model_centers[i]
    centroid_x, centroid_y = zip(*centroids)
    
    ax.scatter(x, y, s=10**2, c=labels, edgecolors="#8cbfd0", alpha=0.80, cmap=color_map, zorder=0)
    ax.scatter(centroid_x, centroid_y, s=16**2, marker="*", c="yellow", zorder=2)
    if old_centroid_x and old_centroid_y:
      ax.scatter(old_centroid_x, old_centroid_y, s=16**2, marker="*", c="grey", zorder=1)
    cmap = cm.get_cmap(color_map)
    
    color_index = [.5, .99, 0.]
    for i, (x1,y1) in enumerate(centroids):
      print(cmap(color_index[i]))
      circle1=plt.Circle((x1,y1),.35,color=cmap(color_index[i]), alpha=.40)
      ax.add_artist(circle1)
    
    ax.set_xlabel("Sepal Length"), ax.set_ylabel("Sepal Width")
    old_centroid_x, old_centroid_y = centroid_x, centroid_y

plt.tight_layout()

fig

# COMMAND ----------

# MAGIC %md <i18n value="06e7d08a-e824-435d-9835-adc29bd5c12e"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC So let's take a look at what's happening here in the distributed setting.

# COMMAND ----------

# MAGIC %md <i18n value="edc1d38d-5cc3-4bf5-bfc5-bb85a145bb16"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/Mapstage.png" height=200px>

# COMMAND ----------

# MAGIC %md <i18n value="aa078ae4-fbfd-4dc2-b0cb-92bc10714981"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/Mapstage2.png" height=500px>

# COMMAND ----------

# MAGIC %md <i18n value="9cf17004-1750-49fe-bb92-ce38c54c1ced"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/ReduceStage.png" height=500px>

# COMMAND ----------

# MAGIC %md <i18n value="80c66031-e786-404e-8c77-c90a91fa3f4a"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/Communication.png" height=500px>

# COMMAND ----------

# MAGIC %md <i18n value="e0f585c1-1d13-4f8c-b9ae-dfd184547653"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Take Aways
# MAGIC
# MAGIC When designing/choosing distributed ML algorithms
# MAGIC * Communication is key!
# MAGIC * Consider your data/model dimensions & how much data you need.
# MAGIC * Data partitioning/organization is important.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
