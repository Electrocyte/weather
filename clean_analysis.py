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
from os import path
import math
import glob
import numpy as np


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
    year = df['Year'].iloc[-1]
    hottest_night = tropical_nights['minT'].max()
    print (f'Number of tropical nights (>20C) in {year}: {len(tropical_nights)}\nHottest night: {hottest_night}')
    
    return day_df


def make_datetime(row):
    return date(year=int(row['Year']), month=int(row['Month']), day=int(row['Day']))


#################################

directory = "E:/weather/new/"

city = ["Bayonne"]
city = ["Penvenan","Penvénan"]
city = ["Boston"]
city = ["Perpignan"]
city = ["Saint-Geoire-en-Valdaine"]
city = ["Singapore"]

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

mean_monthly_df, sub_df = monthly_mean(cat_df, f"Europe/{city}")
days_w_data = check_days_with_data(cat_df)


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

    plt.plot(x, y1, color = 'skyblue', linewidth=1)
    plt.plot(x, y2, color = 'darkred', linewidth=1)
    ax.fill_between(x, y1, y2, color = 'black')

    ax.set_ylabel('Temperature (celsius)', fontsize=30)
    ax.set_xlabel('Time', fontsize=30)
    plt.xticks(rotation=90)
    plt.title(f'Daily temperature extremes for {city} {min(year)}-{max(year)}', fontsize=50)
    ax.tick_params(axis='both', which='major', labelsize=25)


plot_temperature(sub_df)










