# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="54c3040f-38b6-4562-8dd3-61a8bb6aeba1"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Time Series Forecasting
# MAGIC
# MAGIC Working with time series data is an often under-represented skill in data science.  In this notebook, you explore three main approaches to time series: Prophet, ARIMA, and exponential smoothing.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC - Introduce the main concepts in Time Series
# MAGIC - Forecast COVID data using Prophet
# MAGIC - Forecast using ARIMA
# MAGIC - Forecast using Exponential Smoothing
# MAGIC
# MAGIC In this notebook we will be using the <a href="https://www.kaggle.com/kimjihoo/coronavirusdataset" target="_blank">Coronavirus dataset</a> containing data about Coronavirus patients in South Korea.

# COMMAND ----------

# MAGIC %pip install --upgrade pystan==2.19.1.1 fbprophet

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md <i18n value="5537f13d-b402-464f-8814-bb981709ffb2"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### <a href="https://en.wikipedia.org/wiki/Time_series", target="_blank">Time Series</a>
# MAGIC
# MAGIC A time series is a series of data points indexed (or listed or graphed) in time order. Most commonly, a time series is a sequence taken at successive equally spaced points in time. Thus it is a sequence of discrete-time data. Examples of time series include:<br><br>
# MAGIC
# MAGIC - Heights of ocean tides
# MAGIC - Counts of sunspots
# MAGIC - Daily closing value of the Dow Jones Industrial Average
# MAGIC
# MAGIC In this notebook, we will be focusing on time series forecasting, or, the use of a model to predict future values based on previously observed values.

# COMMAND ----------

file_path = f"{DA.paths.datasets}/COVID/coronavirusdataset/Time.csv"

spark_df = (spark
            .read
            .option("inferSchema", True)
            .option("header", True)
            .csv(file_path)
           )
  
display(spark_df)

# COMMAND ----------

# MAGIC %md <i18n value="91681688-70e2-4eee-b18a-4afa353bce3f"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Convert the Spark DataFrame to a Pandas DataFrame.

# COMMAND ----------

df = spark_df.toPandas()

# COMMAND ----------

# MAGIC %md <i18n value="920f1e35-2a54-4588-b4bf-72c6bed85e07"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Looking at the data, the time column (what time of day the data was reported) is not especially relevant to our forecast, so we can go ahead and drop it.

# COMMAND ----------

df = df.drop(columns="time")
df.head()

# COMMAND ----------

# MAGIC %md <i18n value="f5c365d6-4d8b-49a2-a8be-36aa3232d6c1"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Prophet
# MAGIC <a href="https://facebook.github.io/prophet/" target="_blank">Facebook's Prophet</a> is widely considered the easiest way to forecast because it generally does all the heavy lifting for the user. Let's take a look at how Prophet works with our dataset.

# COMMAND ----------

import pandas as pd
from fbprophet import Prophet
import logging

# Suppresses `java_gateway` messages from Prophet as it runs.
logging.getLogger("py4j").setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md <i18n value="a0a43507-9db1-41da-b7bf-8d5c2f4b2a67"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Prophet expects certain column names for its input DataFrame. The date column must be renamed ds, and the column to be forecast should be renamed y. Let's go ahead and forecast the number of confirmed patients in South Korea.

# COMMAND ----------

prophet_df = pd.DataFrame()
prophet_df["ds"] = pd.to_datetime(df["date"])
prophet_df["y"] = df["confirmed"]
prophet_df.head()

# COMMAND ----------

# MAGIC %md <i18n value="daf04369-b1c3-4c84-80c1-0f7da47fc3e6"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Next, let's specify how many days we want to forecast for. We can do this using the **`Prophet.make_future_dataframe`** method. With the size of our data, let's take a look at the numbers a month from now. 
# MAGIC
# MAGIC We can see dates up to one month in the future.

# COMMAND ----------

prophet_obj = Prophet()
prophet_obj.fit(prophet_df)
prophet_future = prophet_obj.make_future_dataframe(periods=30)
prophet_future.tail()

# COMMAND ----------

# MAGIC %md <i18n value="b79ef0fd-8017-4d04-a1f4-f4e8f04dfc87"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Finally, we can run the **`predict`** method to forecast our data points. The **`yhat`** column contains the forecasted values. You can also look at the entire DataFrame to see what other values Prophet generates.

# COMMAND ----------

prophet_forecast = prophet_obj.predict(prophet_future)
prophet_forecast[['ds', 'yhat']].tail()

# COMMAND ----------

# MAGIC %md <i18n value="02352d36-96cb-4c11-aca6-d47a194f9942"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's take a look at a graph representation of our forecast using **`plot`**

# COMMAND ----------

prophet_plot = prophet_obj.plot(prophet_forecast)

# COMMAND ----------

# MAGIC %md <i18n value="d260f48c-7aaa-4cf2-8ab1-50c3a9c7318d"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC We can also use **`plot_components`** to get a more detailed look at our forecast.

# COMMAND ----------

prophet_plot2 = prophet_obj.plot_components(prophet_forecast)

# COMMAND ----------

# MAGIC %md <i18n value="44baaf89-7f68-48a7-9cfd-8ed431613bfa"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC We can also use Prophet to identify <a href="https://facebook.github.io/prophet/docs/trend_changepoints.html" target="_blank">changepoints</a>, points where the dataset had an abrupt change. This is especially useful for our dataset because it could identify time periods where Coronavirus cases spiked.

# COMMAND ----------

from fbprophet.plot import add_changepoints_to_plot

prophet_plot = prophet_obj.plot(prophet_forecast)
changepts = add_changepoints_to_plot(prophet_plot.gca(), prophet_obj, prophet_forecast)

# COMMAND ----------

print(prophet_obj.changepoints)

# COMMAND ----------

# MAGIC %md <i18n value="93d9af60-11de-472f-8a26-f9a679ff29f2"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Next, let's find out if there's any correlation between holidays in South Korea and increases in confirmed cases. We can use the built-in **`add_country_holidays`** <a href="https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#built-in-country-holidays" target="_blank">method</a> to find out about any trends.
# MAGIC
# MAGIC You can find a complete list of country codes <a href="https://github.com/dr-prodigy/python-holidays/blob/master/holidays/countries/" target="_blank">here</a>.

# COMMAND ----------

holidays = pd.DataFrame({"ds": [], "holiday": []})
prophet_holiday = Prophet(holidays=holidays)

prophet_holiday.add_country_holidays(country_name='KR')
prophet_holiday.fit(prophet_df)

# COMMAND ----------

# MAGIC %md <i18n value="0a5f7168-3877-4457-81d4-4ceebce8ec02"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC You can check what holidays are included by running the following cell.

# COMMAND ----------

prophet_holiday.train_holiday_names

# COMMAND ----------

prophet_future = prophet_holiday.make_future_dataframe(periods=30)
prophet_forecast = prophet_holiday.predict(prophet_future)
prophet_plot_holiday = prophet_holiday.plot_components(prophet_forecast)

# COMMAND ----------

# MAGIC %md <i18n value="81681565-2eb4-467c-8c5e-c6546c7230aa"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### ARIMA
# MAGIC
# MAGIC ARIMA stands for Auto-Regressive (AR) Integrated (I) Moving Average (MA). An ARIMA model is a form of regression analysis that gauges the strength of one dependent variable relative to other changing variables.
# MAGIC
# MAGIC Much like Prophet, ARIMA  predicts future values based on the past values of your dataset. Unlike Prophet, ARIMA has a lot more set-up work but can be applied to a wide variety of time series.
# MAGIC
# MAGIC To create our ARIMA model, we need to find the following parameters:<br><br>
# MAGIC
# MAGIC - **`p`**: The number of lag observations included in the model, also called the lag order.
# MAGIC - **`d`**: The number of times that the raw observations are differenced, also called the degree of differencing.
# MAGIC - **`q`**: The size of the moving average window, also called the order of moving average.

# COMMAND ----------

# MAGIC %md <i18n value="99b5826a-6cf1-4bb0-a859-8cce45c50f74"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Start with making our new ARIMA DataFrame. Since we already forecast the confirmed cases using Prophet, let's take a look at predictions for released patients.

# COMMAND ----------

arima_df = pd.DataFrame()
arima_df["date"] = pd.to_datetime(df["date"])
arima_df["released"] = df["released"]
arima_df.head()

# COMMAND ----------

# MAGIC %md <i18n value="286ec724-d52c-46ff-bd35-84201bede0a5"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC The first step of creating an ARIMA model is to find the d-parameter by making sure your dataset is stationary. This is easy to check using an <a href="https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test" target="_blank">Augmented Dickey Fuller Test</a> from the **`statsmodels`** library. 
# MAGIC
# MAGIC Since the P-value is larger than the ADF statistic, we will have to difference the dataset. Differencing helps stabilize the mean of the dataset, therefore removing the influence of past trends and seasonality on your data.

# COMMAND ----------

from statsmodels.tsa.stattools import adfuller
from numpy import log

result = adfuller(df.released.dropna())
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# COMMAND ----------

# MAGIC %md <i18n value="8766e03d-4623-40c2-a5fa-25684db75670"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC To difference the dataset, call **`diff`** on the value column. We are looking for a near-stationary series which roams around a defined mean and an ACF plot that reaches zero fairly quickly. After looking at our graphs, we can determine that our d-parameter should either be 1 or 2.

# COMMAND ----------

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.rcParams.update({"figure.figsize":(9,7), "figure.dpi":120})

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(arima_df.released); axes[0, 0].set_title('Original Series')
plot_acf(arima_df.released, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(arima_df.released.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(arima_df.released.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(arima_df.released.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(arima_df.released.diff().diff().dropna(), ax=axes[2, 1])

plt.show()

# COMMAND ----------

# MAGIC %md <i18n value="7e2a2b58-4516-4036-9f93-e6514c529de5"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC In the next section we'll find the required number of AR terms using the Partial Autocorrection Plot. This is the p-parameter.
# MAGIC
# MAGIC Partial Autocorrection is the correlation between a series and its lag. From the graphs, our p-parameter should be 1.

# COMMAND ----------

plt.rcParams.update({"figure.figsize":(9,3), "figure.dpi":120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(arima_df.released.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(arima_df.released.diff().dropna(), ax=axes[1])

plt.show()

# COMMAND ----------

# MAGIC %md <i18n value="42204bfb-229b-4695-82f1-56b34ad04ba2"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Finally, we'll find the q-parameter by looking at the ACF plot to find the number of Moving Average terms. A Moving Average incorporates the dependency between an observation and a residual error applied to lagged observations. From the graphs, our q-parameter should be 1.

# COMMAND ----------

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(arima_df.released.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(arima_df.released.diff().dropna(), ax=axes[1])

plt.show()

# COMMAND ----------

# MAGIC %md <i18n value="7855a2df-0091-4a6d-b05f-538c20784b66"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Once we have found our p, d, and q parameter values, we can fit our ARIMA model by passing the parameters in. The following cell shows a summary of the model including dataset information and model coefficients.

# COMMAND ----------

from statsmodels.tsa.arima_model import ARIMA

# p, d, q
# 1, 2, 1 ARIMA Model
model = ARIMA(arima_df.released, order=(1,2,1))
arima_fit = model.fit(disp=0)
print(arima_fit.summary())

# COMMAND ----------

# MAGIC %md <i18n value="b787949f-d55a-4178-baf2-66b023567904"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Finally, let's split our data into train and test data to test the accuracy of our model. Note that since we have to split the data sequentially for time series, functions like sklearn's **`train_test_split`** cannot be used here.

# COMMAND ----------

split_ind = int(len(arima_df)*.7)
train_df = arima_df[ :split_ind]
test_df = arima_df[split_ind: ]
#train_df.tail()
#test_df.head()

# COMMAND ----------

# MAGIC %md <i18n value="5e50d9a6-321f-4577-afa3-e146ad7a38ab"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC To forecast we use Out of Sample Cross Validation. We can see from the graph that our forecast is slightly more linear than the actual values but overall the values are pretty close to expected.

# COMMAND ----------

train_model = ARIMA(train_df.released, order=(1,2,1))  
train_fit = train_model.fit()  

fc, se, conf = train_fit.forecast(int(len(arima_df)-split_ind))

fc_series = pd.Series(fc, index=test_df.index)

plt.plot(train_df.released, label='train', color="dodgerblue")
plt.plot(test_df.released, label='actual', color="orange")
plt.plot(fc_series, label='forecast', color="green")
plt.title('Forecast vs Actuals')
plt.ylabel("Number of Released Patients")
plt.xlabel("Day Number")
plt.legend(loc='upper left', fontsize=8)
plt.show()

# COMMAND ----------

# MAGIC %md <i18n value="ce127e7b-c717-4d46-b354-77bf6a0f8dc0"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Exponential Smoothing
# MAGIC
# MAGIC <a href="(https://en.wikipedia.org/wiki/Exponential_smoothing" target="_blank">Exponential smoothing</a> is a rule of thumb technique for smoothing time series data using the exponential window function. Whereas in the simple moving average the past observations are weighted equally, exponential functions are used to assign exponentially decreasing weights over time. It is an easily learned and easily applied procedure for making some determination based on prior assumptions by the user, such as seasonality. Exponential smoothing is often used for analysis of time-series data.
# MAGIC
# MAGIC There are three types of Exponential Smoothing:<br><br>
# MAGIC - Single Exponential Smoothing (SES)
# MAGIC   - Used for datasets without trends or seasonality.
# MAGIC - Double Exponential Smoothing (also known as Holt's Linear Smoothing)
# MAGIC   - Used for datasets with trends but without seasonality.
# MAGIC - Triple Exponential Smoothing (also known as Holt-Winters Exponential Smoothing)
# MAGIC   - Used for datasets with both trends and seasonality.
# MAGIC
# MAGIC In our case, the Coronavirus dataset has a clear trend, but seasonality is not especially important, therefore we will be using double exponential smoothing.

# COMMAND ----------

# MAGIC %md <i18n value="294eaf9b-8ba0-4137-bb52-d27b39f3d34f"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Since we have already forecast the other two columns, let's take a look at a forecast for the number of coronavirus related deaths.

# COMMAND ----------

exp_df = pd.DataFrame()
exp_df["date"] = pd.to_datetime(df["date"])
exp_df["deceased"] = df["deceased"]
exp_df.head()

# COMMAND ----------

# MAGIC %md <i18n value="7ddb8a5e-453b-491d-ac0d-3087a2c7f955"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Holt's Linear Smoothing only works on data points that are greater than 0, therefore we have to drop the corresponding rows. Additionally, we need to set the index of our DataFrame to the date column.

# COMMAND ----------

exp_df = exp_df[exp_df["deceased"] != 0]
exp_df = exp_df.set_index("date")
exp_df.head()

# COMMAND ----------

# MAGIC %md <i18n value="b4fb0e59-5554-44b0-81b3-efbfeca88e33"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Luckily, statsmodel does most of the work for us. However, we still have to tweak the parameters to get an accurate forecast. The available parameters here are α or **`smoothing_level`** and β or **`smoothing_slope`**. α defines the smoothing factor of the level and β defines the smoothing factor of the trend.
# MAGIC
# MAGIC In the cell below, we are trying three different kinds of predictions. The first, Holt's Linear Trend, forecasts with a linear trend. The second, Exponential Trend, forecasts with an exponential trend. The third, Additive Damped Trend, damps the forecast trend linearly.

# COMMAND ----------

from statsmodels.tsa.holtwinters import Holt

exp_fit1 = Holt(exp_df.deceased).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
exp_forecast1 = exp_fit1.forecast(30).rename("Holt's linear trend")

exp_fit2 = Holt(exp_df.deceased, exponential=True).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
exp_forecast2 = exp_fit2.forecast(30).rename("Exponential trend")

exp_fit3 = Holt(exp_df.deceased, damped=True).fit(smoothing_level=0.8, smoothing_slope=0.2)
exp_forecast3 = exp_fit3.forecast(30).rename("Additive damped trend")

# COMMAND ----------

# MAGIC %md <i18n value="4e3c58f7-afdc-42c6-805f-6f34770ea4d8"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC After plotting the three models, we can see that the standard Holt's Linear, and the Exponential trend lines give very similar forecasts while the Additive Damped trend gives a slightly lower number of deceased patients.

# COMMAND ----------

exp_fit1.fittedvalues.plot(color="orange", label="Holt's linear trend")
exp_fit2.fittedvalues.plot(color="red", label="Exponential trend")
exp_fit3.fittedvalues.plot(color="green", label="Additive damped trend")

plt.legend()
plt.ylabel("Number of Deceased Patients")
plt.xlabel("Day Number")
plt.show()

# COMMAND ----------

# MAGIC %md <i18n value="f3bbd647-3586-482f-880a-369258cfc7d0"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC We can zoom in on the forecast part of our graph to see the graph in more detail.
# MAGIC
# MAGIC We can see that the exponential trendline starts in-line with the linear trendline but slowly starts resembling an exponential trend towards the end of the graph. The damped trendline starts and ends below the other trendlines.

# COMMAND ----------

exp_forecast1.plot(legend=True, color="orange")
exp_forecast2.plot(legend=True, color="red")
exp_forecast3.plot(legend=True, color="green")

plt.ylabel("Number of Deceased Patients")
plt.xlabel("Day Number")
plt.show()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
