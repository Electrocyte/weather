#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 07:57:18 2022

@author: mangi
"""

from pathlib import Path
import pandas as pd
from dateutil import tz
import seaborn as sns; sns.set()
from datetime import date, datetime, timezone
import matplotlib.pyplot as plt
import datetime as dt
from os import path
import math
import glob
import numpy as np
import warnings
warnings.filterwarnings("ignore") # specify to ignore warning messages


def formatTime(timestamp, t_format, city_timezone):
    utc = datetime.fromtimestamp(timestamp, timezone.utc)
    city_timezone = tz.gettz(city_timezone)
    return utc.astimezone(city_timezone).strftime(t_format)


def time_columns(df, time__zone):

    df["Year"] = df["UNIX_UTC"].apply(formatTime, t_format = "%Y", city_timezone=time__zone)
    df["Year"] = pd.to_numeric(df["Year"], errors='coerce')
    df["Month"] = df["UNIX_UTC"].apply(formatTime, t_format = "%m", city_timezone=time__zone)
    df["Month"] = pd.to_numeric(df["Month"], errors='coerce')
    df["Day"] = df["UNIX_UTC"].apply(formatTime, t_format = "%d", city_timezone=time__zone)
    df["Day"] = pd.to_numeric(df["Day"], errors='coerce')
    df["Hour"] = df["UNIX_UTC"].apply(formatTime, t_format = "%H", city_timezone=time__zone)
    df["Hour"] = pd.to_numeric(df["Hour"], errors='coerce')
    df["Minute"] = df["UNIX_UTC"].apply(formatTime, t_format = "%M", city_timezone=time__zone)
    df["Minute"] = pd.to_numeric(df["Minute"], errors='coerce')
    df["FloatHour"] = df["Hour"]+(df["Minute"]/60)

    df['New_time'] = pd.to_datetime(df['UNIX_UTC'], unit='s')
    df = df.set_index('New_time')
    df = df.sort_index()        
    return df


def monthly_mean(i, tz):

    i = time_columns(i, tz)

    year_month = i.groupby(['Year','Month'])
    maxT = year_month['Temp'].max()
    maxT = maxT.rename("maxT")
    minT = year_month['Temp'].min()
    minT = minT.rename("minT")
    meanT = year_month['Temp'].mean()
    meanT = meanT.rename("meanT")

    return meanT, i


def check_days_with_data(i: pd.DataFrame) -> dict:
    days_counted = {}
    init = 0
    ii1 = min(i["Year"].unique())
    for ii, group in i.groupby(['Year','Month','Day']):
        i1,i2,i3 = ii
        init += 1
        if i1 > ii1:
            days_counted[i1] = init
            init = 0
            ii1 += 1
    return days_counted


def min_max(df):

    year_month_day = df.groupby(['Year','Month','Day'])

    maxT = year_month_day['Temp'].max()
    maxT = maxT.rename("maxT")
    minT = year_month_day['Temp'].min()
    minT = minT.rename("minT")
    meanT = year_month_day['Temp'].mean()
    meanT = meanT.rename("meanT")
    
    day_df = pd.concat([minT, meanT, maxT], axis=1)
    
    tropical_nights = day_df[day_df["minT"] > 20]
    tropical_nights.reset_index(inplace = True)
    no_trop_nights = tropical_nights.groupby(["Year"])["minT"].count().rename("no. tropical nights")
    hottest_night = tropical_nights.groupby(["Year"])["minT"].max().rename("hottest-night")
    heat = pd.concat([hottest_night, no_trop_nights], axis = 1)
    
    for n, value in enumerate(list(heat.index)):
        print(f"Year: {value}, No. tropical nights (>20C): {int(heat.iloc[n]['no. tropical nights'])}, hottest night: {heat.iloc[n]['hottest-night']}")
    
    return day_df


def make_datetime(row):
    return date(year=int(row['Year']), month=int(row['Month']), day=int(row['Day']))


#################################

directory = "/mnt/e/weather/new/"

city = ["Bayonne"]
# city = ["Penvenan","Penvénan"]
# city = ["Boston"]
# city = ["Perpignan"]
# city = ["Saint-Geoire-en-Valdaine"]
# city = ["Singapore"]
# city = ["San Diego"]
timezone___ = f"Europe/{city}"

globs = []
for citi in city:
    globs1 = glob.glob(f"{directory}/**/{citi}*tab", recursive = True)
    globs2 = glob.glob(f"{directory}/**/{citi}/*tab", recursive = True)
    globs.append(globs1)
    globs.append(globs2)
globs = [x for xs in globs for x in xs]

dfs = []
for found_glob in globs:
    if path.exists(found_glob):
        df = pd.read_csv(found_glob, delimiter = "\t", index_col=None)
        dfs.append(df)
cat_df = pd.concat(dfs)
cat_df = cat_df.drop(['Unnamed: 0'], axis = 1)
cat_df = cat_df.drop_duplicates(["City", "UNIX_UTC"])
cat_df.reset_index(inplace = True, drop = True)

cat_df["simpleTime"] = cat_df["Time"].str.split(" ", expand = True)[0]

mean_monthly_df, cat_df = monthly_mean(cat_df, timezone___)
cat_df.reset_index(inplace = True)

time_diffs = pd.DataFrame(cat_df["UNIX_UTC"].diff())
gaps = time_diffs.loc[time_diffs["UNIX_UTC"] > 86400]

############ SIMPLE INTERPOLATION ############

all_times = cat_df["New_time"].reindex(pd.date_range(start = cat_df["New_time"].min(), end = cat_df["New_time"].max(), freq='30min'))
all_times = pd.DataFrame(all_times)
all_times.reset_index(inplace = True)
all_times = all_times.drop(["New_time"], axis = 1)

gap_pairs = [(i-1, i) for i in gaps.index]
all_missing_times = []
for pair in gap_pairs:
    before = cat_df.iloc[list(pair)[0]]["New_time"]
    after = cat_df.iloc[list(pair)[1]]["New_time"]
    delta = after - before
    missing_timepoints = all_times.loc[(all_times["index"] > before) & (all_times["index"] < after)]
    missing_timepoints["UNIX_UTC"] = missing_timepoints['index'].astype('int64')//1e9
    all_missing_times .append ( missing_timepoints)
cat_miss_times = pd.concat(all_missing_times)
cat_miss_times = cat_miss_times.rename(columns={"index": "New_time"})
cat_miss_times = cat_miss_times.set_index(["UNIX_UTC"])
cat_df2 = cat_df.set_index(["UNIX_UTC"])

interpolated_df = pd.concat([cat_df2, cat_miss_times])
interpolated_df = interpolated_df.sort_index(ascending=True)
interpolated_df.reset_index(inplace = True)

for col in ['Min_temp', 'Max_temp', 'Temp']:
    interpolated_df[col] = interpolated_df[col].interpolate()
    # interpolated_df[col] = interpolated_df[col].interpolate(method = "time")

interpolated_df = time_columns(interpolated_df, timezone___)
interpolated_df["City"] = citi

############ SIMPLE INTERPOLATION ############
############ similar time INTERPOLATION ############

cat_miss_times["no-year"] = cat_miss_times["New_time"].astype(str).str.replace(r"\d\d\d\d-", "")
cat_miss_times.reset_index(inplace=True)
cat_miss_times = time_columns(cat_miss_times, timezone___)
missing_dates = cat_miss_times[['UNIX_UTC', 'Month', 'Day', 'Hour']]

def find_other_readings(row):
    
    within_one_hour = cat_df2.loc[(cat_df2['Month'] == row["Month"]) & (cat_df2['Day'] == row["Day"]) & (cat_df2['Hour'] == row["Hour"])]
    
    row["Temp"] = within_one_hour["Temp"].mean()
    row["Max_temp"] = within_one_hour["Temp"].max()
    row["Min_temp"] = within_one_hour["Temp"].min()
    row["std"] = within_one_hour["Temp"].std()
    
    return row

gg = missing_dates.apply(find_other_readings, axis = 1)
gg.reset_index(inplace = True)
gg = gg.set_index(["UNIX_UTC"])

mean_prev_year_df = pd.concat([cat_df2, gg[['New_time','Min_temp', 'Max_temp', 'Temp']]])
mean_prev_year_df = mean_prev_year_df.sort_index(ascending=True)
mean_prev_year_df.reset_index(inplace = True)

for col in ['Min_temp', 'Max_temp', 'Temp']:
    mean_prev_year_df[col] = mean_prev_year_df[col].interpolate()

mean_prev_year_df = time_columns(mean_prev_year_df, timezone___)
mean_prev_year_df["City"] = citi

############ similar time INTERPOLATION ############

days_w_data = check_days_with_data(cat_df)
days_w_data = check_days_with_data(interpolated_df)
days_w_data = check_days_with_data(mean_prev_year_df)


def plot_temperature(df):
    city = df['City'][0]
    year = df['Year'].unique()
    day_df = min_max(df)
    day_df = day_df.reset_index()
    day_df['datetime'] = day_df.apply(make_datetime, axis=1)
    
    x = day_df['datetime'].values
    y1 = day_df['minT'].values
    y2 = day_df['maxT'].values
    
    f, ax = plt.subplots(figsize=(25, 15))
    # ax.set_ylim([0, 40])
    plt.plot(x, y1, color = 'skyblue', linewidth=1)
    plt.plot(x, y2, color = 'darkred', linewidth=1)
    ax.fill_between(x, y1, y2, color = 'black')

    ax.set_ylabel('Temperature (celsius)', fontsize=30)
    ax.set_xlabel('Time', fontsize=30)
    plt.xticks(rotation=90)
    plt.title(f'Daily temperature extremes for {city} {min(year)}-{max(year)}', fontsize=50)
    ax.tick_params(axis='both', which='major', labelsize=25)


plot_temperature(cat_df)
plot_temperature(interpolated_df)
plot_temperature(mean_prev_year_df)


only_2019 = mean_prev_year_df.loc[mean_prev_year_df["Year"] == 2019]
only_june = only_2019.loc[only_2019["Month"] == 6]
two_weeks = only_june.loc[only_june["Day"] .isin (range(1,14))]
one_day = only_june.loc[only_june["Day"] .isin ([1])]

from statsmodels.tsa.seasonal import MSTL
pd.plotting.register_matplotlib_converters()

import statsmodels.api as sm
from scipy.stats import norm
import pylab

data_MSTL = pd.DataFrame(data=only_2019["Temp"], index=only_2019.index)

# qq plot
# normal distribution for qq plot looks like a linear plot 
# https://towardsdatascience.com/6-ways-to-test-for-a-normal-distribution-which-one-to-use-9dcf47d8fa93 
sm.qqplot(data_MSTL, line='45')
pylab.show()

# Kolmogorov Smirnov test
    # If the P-Value of the KS Test is larger than 0.05, we assume a normal distribution
    # If the P-Value of the KS Test is smaller than 0.05, we do not assume a normal distribution
from scipy.stats import kstest, norm
ks_statistic, p_value = kstest(data_MSTL, 'norm')
print(ks_statistic, p_value)

# Shapiro Wilk test # best test
    # If the P-Value of the Shapiro Wilk Test is larger than 0.05, we assume a normal distribution
    # If the P-Value of the Shapiro Wilk Test is smaller than 0.05, we do not assume a normal distribution
from scipy import stats
shapiro_test = stats.shapiro(data_MSTL)
print(shapiro_test.statistic, shapiro_test.pvalue)

# if the data is present in non-normal shape (which it is), it can be transformed into a normal distribution using the box cox
# https://www.statisticshowto.com/box-cox-transformation/
# Normality is an important assumption for many statistical techniques; 
# if your data isn’t normal, applying a Box-Cox means that you are able to run a broader number of tests.
from sklearn.preprocessing import power_transform
xt, lmbda = stats.yeojohnson(data_MSTL)
print(power_transform(data_MSTL["Temp"].values.reshape(-1, 1), method='yeo-johnson', standardize = False))
xts = power_transform(data_MSTL["Temp"].values.reshape(-1, 1), method='yeo-johnson')
shapiro_test = stats.shapiro(xt)
print(shapiro_test.statistic, shapiro_test.pvalue)

comparison = pd.concat([data_MSTL, pd.DataFrame(xt, index = data_MSTL.index).rename(columns={0: "stats-non-standardised"}), pd.DataFrame(xts, index = data_MSTL.index).rename(columns={0: "standardised"})], axis = 1)

fig = plt.figure()
ax1 = fig.add_subplot(221)
prob = stats.probplot(comparison["Temp"], dist=stats.norm, plot=ax1)
ax1.set_xlabel('')
ax1.set_title('Probplot against normal distribution')

ax2 = fig.add_subplot(222)
prob = stats.probplot(comparison["stats-non-standardised"], dist=stats.norm, plot=ax2)
ax2.set_title('Probplot after Yeo-Johnson transformation')

ax3 = fig.add_subplot(223)
prob = stats.probplot(comparison["standardised"], dist=stats.norm, plot=ax3)
ax3.set_title('Probplot after Yeo-Johnson transformation, standardised')
plt.show()

# seasonal_deg, the polynomial degree used by Loess to extract the seasonal component in STL (typically set to 0 or 1).
stl_kwargs = {"seasonal_deg": 0} 

# model = MSTL(data_MSTL, periods=(24, 24 * 365), stl_kwargs=stl_kwargs)
# https://arxiv.org/pdf/2107.13462.pdf
model = MSTL(data_MSTL, periods=(24, 24 * 28), stl_kwargs=stl_kwargs)

res = model.fit()

# Start with the plot from the results object `res`
plt.rc("figure", figsize=(16, 20))
plt.rc("font", size=13)
fig = res.plot()

plt.tight_layout()
plt.savefig(f"MSTL-plot.png", dpi = 300)
plt.show()

seasonal_components = res.seasonal
seasonal_trend = res.trend
seasonal_resid = res.resid
print(seasonal_components)
print(seasonal_trend)
print(seasonal_resid)

# check for stationarity
# Since the p-value is not less than .05, we fail to reject the null hypothesis.
# This means the time series is non-stationary. 
# In other words, it has some time-dependent structure and does not have constant variance over time.
# H0: The time series is non-stationary. 
# HA: The time series is stationary.
# 0.006 < 0.05; reject H0
# However, this is misleading and may be removed if we look at a daily basis
from statsmodels.tsa.stattools import adfuller
adfuller(xts) # Test-stat = -3.557; p-value = 0.0066
adfuller(xt)  # Test-stat = -3.557; p-value = 0.0066
adfuller(data_MSTL["Temp"]) #  Test-stat = -3.514; p-value = 0.0076 --> REJECT i.e. stationary
adfuller(only_2019["Temp"]) #  Test-stat = -3.514; p-value = 0.0076 --> REJECT i.e. stationary
adfuller(only_june["Temp"]) #  Test-stat = -3.049; p-value = 0.0306 --> REJECT i.e. stationary
adfuller(two_weeks["Temp"]) #  Test-stat = -2.789; p-value = 0.0598 --> ACCEPT i.e. non-stationary
adfuller(one_day["Temp"])   #  Test-stat = -2.350; p-value = 0.1563 --> ACCEPT i.e. non-stationary
# Observations from a non-stationary time series show seasonal effects, trends, and other structures that depend on the time index.
# Summary statistics like the mean and variance do change over time, providing a drift in the concepts a model may try to capture.
# Classical time series analysis and forecasting methods are concerned with making non-stationary time series data stationary by identifying and removing trends and removing stationary effects.


#####################
# If you have clear trend and seasonality in your time series, then model these components, remove them from observations, then train models on the residuals.

## DIFFERENCING TO REMOVE (LINEAR) TRENDS.
# How to apply the difference transform to remove a linear trend from a series.
# A trend makes a time series non-stationary by increasing the level. This has the effect of varying the mean time series value over time.
# create a differenced series
# 48 = readings summing to one day (if every 30 minutes)
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff


# invert differenced forecast
def inverse_difference(last_ob, value):
	return value + last_ob
 

diff_trend_day = difference(two_weeks['Temp'], 48)
inverted = [inverse_difference(two_weeks['Temp'][i], diff_trend_day[i]) for i in range(len(diff_trend_day))]

## Differencing to Remove Seasonality
# How to apply the difference transform to remove a seasonal signal from a series.
# Seasonal variation, or seasonality, are cycles that repeat regularly over time.

from math import sin
from math import radians
from matplotlib import pyplot


# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff


# invert differenced forecast
def inverse_difference(last_ob, value):
	return value + last_ob


# difference the dataset
diff_seasonality_day = difference(two_weeks['Temp'], 48)
pyplot.plot(diff_seasonality_day)
pyplot.show()

# invert the difference
inverted = [inverse_difference(two_weeks['Temp'][i], diff_seasonality_day[i]) for i in range(len(diff_seasonality_day))]
pyplot.plot(inverted)
pyplot.show()

# ARIMA: AutoRegressive Integrated Moving Average.

# 🎓 Stationarity. From a statistical context, stationarity refers to data whose distribution does not change when shifted in time. 
# Non-stationary data, then, shows fluctuations due to trends that must be transformed to be analyzed. 
# Seasonality, for example, can introduce fluctuations in data and can be eliminated by a process of 'seasonal-differencing'.

# 🎓 Differencing. Differencing data, again from a statistical context, refers to the process of transforming non-stationary data 
# to make it stationary by removing its non-constant trend. "Differencing removes the changes in the level of a time series, 
# eliminating trend and seasonality and consequently stabilizing the mean of the time series." Paper by Shixiong et al

# AR - for AutoRegressive. Autoregressive models, as the name implies, look 'back' in time to analyze previous 
# values in your data and make assumptions about them. These previous values are called 'lags'. 
# An example would be data that shows monthly sales of pencils. 
# Each month's sales total would be considered an 'evolving variable' in the dataset. 
# This model is built as the "evolving variable of interest is 
# regressed on its own lagged (i.e., prior) values." wikipedia

# I - for Integrated. As opposed to the similar 'ARMA' models, the 'I' in ARIMA refers to its integrated aspect. 
# The data is 'integrated' when differencing steps are applied so as to eliminate non-stationarity.

# MA - for Moving Average. The moving-average aspect of this model refers to the output variable that 
# is determined by observing the current and past values of lags.

from sklearn.preprocessing import MinMaxScaler

train_start_dt = only_2019.index[0]
test_start_dt = only_2019.index[int(len(only_2019) * 0.98)]

only_2019[(only_2019.index < test_start_dt) & (only_2019.index >= train_start_dt)][['Temp']].rename(columns={'Temp':'train'}) \
    .join(only_2019[test_start_dt:][['Temp']].rename(columns={'Temp':'test'}), how='outer') \
    .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
plt.title("Train-test-split")
plt.xlabel('timestamp', fontsize=12)
plt.ylabel('Temp', fontsize=12)
plt.show()

train = only_2019.copy()[(only_2019.index >= train_start_dt) & (only_2019.index < test_start_dt)][['Temp']]
test = only_2019.copy()[only_2019.index >= test_start_dt][['Temp']]

print('Training data shape: ', train.shape)
print('Test data shape: ', test.shape)

scaler = MinMaxScaler()
train['Temp'] = scaler.fit_transform(train)
train.head(10)

only_2019[(only_2019.index >= train_start_dt) & (only_2019.index < test_start_dt)][['Temp']].rename(columns={'Temp':'original Temp'}).plot.hist(bins=100, fontsize=12)
train.rename(columns={'Temp':'scaled Temp'}).plot.hist(bins=100, fontsize=12)
plt.show()

test['Temp'] = scaler.transform(test)
test.head()

from statsmodels.tsa.statespace.sarimax import SARIMAX
# from common.utils import load_data, mape
from IPython.display import Image
from pandas.plotting import autocorrelation_plot
pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)


# Define the model by calling SARIMAX() and passing in the model parameters: 
# p, d, and q parameters, and P, D, and Q parameters.
# Prepare the model for the training data by calling the fit() function.
# Make predictions calling the forecast() function and specifying the number of steps (the horizon) to forecast.

# 🎓 What are all these parameters for? 
# In an ARIMA model there are 3 parameters that are used to help model the major aspects of a time series: 
# seasonality, trend, and noise. These parameters are:
#     p: the parameter associated with the auto-regressive aspect of the model, which incorporates past values.
#     d: the parameter associated with the integrated part of the model, 
#         which affects the amount of differencing (🎓 remember differencing 👆?) to apply to a time series. 
#     q: the parameter associated with the moving-average part of the model.

# Note: If your data has a seasonal aspect - which this one does - we use a seasonal ARIMA model (SARIMA). 
# In that case you need to use another set of parameters: P, D, and Q 
# which describe the same associations as p, d, and q, but correspond to the seasonal components of the model.

# Specify the number of steps to forecast ahead
HORIZON = 48
print('Forecasting horizon:', HORIZON/2, 'hours')

# Selecting the best values for an ARIMA model's parameters can be challenging as it's somewhat subjective and time intensive. 
# You might consider using an auto_arima() function from the pyramid library

order = (4, 1, 0) # p, d, q
seasonal_order = (1, 1, 0, 48) # P, D, Q, s

model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
results = model.fit()

print(results.summary())

#                                      SARIMAX Results                                      
# ==========================================================================================
# Dep. Variable:                               Temp   No. Observations:                15632
# Model:             SARIMAX(4, 1, 0)x(1, 1, 0, 48)   Log Likelihood               46193.325
# Date:                            Tue, 14 Jun 2022   AIC                         -92374.650
# Time:                                    16:50:59   BIC                         -92328.726
# Sample:                                         0   HQIC                        -92359.442
#                                           - 15632                                         
# Covariance Type:                              opg                                         
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# ar.L1          0.0220      0.006      3.929      0.000       0.011       0.033
# ar.L2          0.1336      0.005     24.385      0.000       0.123       0.144
# ar.L3          0.0130      0.006      2.352      0.019       0.002       0.024
# ar.L4          0.0609      0.006     10.230      0.000       0.049       0.073
# ar.S.L48      -0.4634      0.004   -111.952      0.000      -0.472      -0.455
# sigma2         0.0002   9.17e-07    169.642      0.000       0.000       0.000
# ===================================================================================
# Ljung-Box (L1) (Q):                   6.35   Jarque-Bera (JB):             20711.42
# Prob(Q):                              0.01   Prob(JB):                         0.00
# Heteroskedasticity (H):               1.03   Skew:                            -0.04
# Prob(H) (two-sided):                  0.23   Kurtosis:                         8.65
# ===================================================================================

# Warnings:
# [1] Covariance matrix calculated using the outer product of gradients (complex-step).

#############################
#############################
# Walk-forward validation is the gold standard of time series model evaluation and is recommended for your own projects.


test_shifted = test.copy()

for t in range(1, HORIZON+1):
    test_shifted['Temp+'+str(t)] = test_shifted['Temp'].shift(-t)

test_shifted = test_shifted.dropna(how='any')
test_shifted.head(5)

# Make predictions on your test data using this sliding window approach in a loop the size of the test data length:

%time
training_window = 720 # dedicate 30 days (720 hours) for training
# probably needs editing for my data where 48 x no. days
# this model needs optimising for number of iterations... taking too long to run or whatever.

train_ts = train['Temp']
test_ts = test_shifted

history = [x for x in train_ts]
history = history[(-training_window):]

predictions = list()

order = (4, 1, 0) # p, d, q
seasonal_order = (1, 1, 0, 48) # P, D, Q, s

# why is this the same size as the data set.
# might be required due to 
# ValueError: Length of values (1690) does not match length of index (446)
# very slow
# ValueError: Length of values (14700) does not match length of index (14400)
for t in range(test_ts.shape[0]):
    model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    yhat = model_fit.forecast(steps = HORIZON)
    predictions.append(yhat)
    obs = list(test_ts.iloc[t])
    # move the training window
    history.append(obs[0])
    history.pop(0)
    print(test_ts.index[t])
    print(t+1, ': predicted =', yhat, 'expected =', obs)


# Compare the predictions to the actual load:

eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
# eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON]
eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
# eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
eval_df['actual'] = np.array(np.transpose(test_ts.iloc[:, 1:])).ravel()
eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
eval_df.head()

eval_df.to_csv("evaluation_df.csv")

load_eval_df = pd.read_csv("evaluation_df.csv")