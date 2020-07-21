# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:49:38 2020

@author: Mangifera
"""

import altair as alt
import streamlit as st

from pathlib import Path
import pandas as pd
from dateutil import tz
import seaborn as sns; sns.set()
from datetime import date, datetime, timezone
import matplotlib.pyplot as plt
from os import path





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

#     Set index as a datetime
    df['New_time'] = pd.to_datetime(df['UNIX_UTC'], unit='s')
    df = df.set_index('New_time')
    df = df.sort_index()        
    return df


def formatTime(timestamp, t_format, city_timezone):
    utc = datetime.fromtimestamp(timestamp, timezone.utc)
    city_timezone = tz.gettz(city_timezone)
    return utc.astimezone(city_timezone).strftime(t_format)
   

def monthly_mean_ori(i):
#
    year_month = i.groupby(['Year','Month'])
    maxT = year_month['Temp'].max()
    maxT = maxT.rename("maxT")
    minT = year_month['Temp'].min()
    minT = minT.rename("minT")
    meanT = year_month['Temp'].mean()
    meanT = meanT.rename("meanT")

    return meanT


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
    st.markdown (f'Number of tropical nights (>20C) in {year}: {len(tropical_nights)}\nHottest night: {hottest_night}')
    
    return day_df


def split_df(i):
    city = i['City'][0]
    time__zone = CITY_TZS_FILE.loc[city]['Time_Zone']    
    i = time_columns(i, time__zone) 
    mask = i['Year'] < 2019
    
    _2018 = i[mask]
    _2019 = i[~mask]
    
    _2018_ = monthly_mean_ori(_2018)
    _2019_ = monthly_mean_ori(_2019)
    
    return _2018_, _2019_, _2018, _2019



def monthly_mean(i):
    city = i['City'][0]
    time__zone = CITY_TZS_FILE.loc[city]['Time_Zone']   
    i = time_columns(i, time__zone)
    
    # Find the "Year" with the most number of rows
    year_to_keep = i.groupby(['Year'])["Year"].count().idxmax()
    i = i[i["Year"] == year_to_keep]
    
    year_month = i.groupby(['Year','Month'])
    maxT = year_month['Temp'].max()
    maxT = maxT.rename("maxT")
    minT = year_month['Temp'].min()
    minT = minT.rename("minT")
    meanT = year_month['Temp'].mean()
    meanT = meanT.rename("meanT")

    return meanT, i


def plot_precipitation(df):
    year_month_day = df.groupby(['Year','Month','Day'])
    daily_max = year_month_day['Rain[1h][mm]'].sum()
    st.markdown(f"# Rainfall for {df.City.unique()[0]}")
    daily_max = daily_max.reset_index()
    daily_max['datetime'] = daily_max.apply(make_datetime, axis=1)    
    cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
    ax = sns.scatterplot(x="datetime", y="Rain[1h][mm]", hue="Rain[1h][mm]", palette=cmap, data=daily_max)
    ax.set_xticklabels(daily_max['Day'], rotation='vertical', fontsize=10)
    plt.xlim(daily_max["datetime"].min(), daily_max["datetime"].max())
    plt.ylim((0, daily_max["Rain[1h][mm]"].max()*1.1))
    st.pyplot()
    plt.show()


def make_datetime(row):
    return date(year=int(row['Year']), month=int(row['Month']), day=int(row['Day']))


def plot_temperature(df):
    city = df['City'][0]
    year = df['Year'][0]
    day_df = min_max(df)
    day_df = day_df.reset_index()
    day_df['datetime'] = day_df.apply(make_datetime, axis=1)
    
    x = day_df['datetime'].values
    y1 = day_df['minT'].values
    y2 = day_df['maxT'].values
    
    fig, ax1 = plt.subplots(1, 1, sharex=True, dpi = 300)
    plt.subplots_adjust(top=0.85,bottom=0.25)

#     plot minimum and maximum line colours
    plt.plot(x, y1, color = 'skyblue', linewidth=1)
    plt.plot(x, y2, color = 'darkred', linewidth=1)
    ax1.fill_between(x, y1, y2, color = 'black')

    ax1.set_ylabel('Temperature (celsius)', fontsize=14)
    ax1.set_xlabel('Days', fontsize=14)
    plt.xticks(rotation=45)
    plt.title(f'Daily temperature extremes \nfor {city} {year}', fontsize=25)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    st.pyplot()
#    plt.savefig(f'Daily temperature extremes for {city} {year}', bbox_inches='tight', dpi=300)


#===============================================================================
INPUT_CITY = "Rochecorbon"

data_folders = ["new_system", "output"]
directory = f"/mnt/d/james/Documents/SpiderOak Hive/Data/weather/new_system/{INPUT_CITY}/"
directory_ = Path(directory)

old = "/mnt/d/james/Documents/SpiderOak Hive/Data/weather/a_older_system/"
old_bayonne = f"{old}{INPUT_CITY}.tab"

DIRECTORY = "/mnt/d/james/Documents/Python Scripts/"
CITY_TZS = DIRECTORY+"CityTzs.csv"
with open(CITY_TZS, 'r') as myfile:
    CITY_TZS_FILE = pd.read_csv(CITY_TZS, index_col = "City_name")
#===============================================================================   

st.write(f"# Weather charts for {INPUT_CITY}")


    
if path.exists(old_bayonne):
    with open(old_bayonne) as f:
        i = pd.read_csv(f, delimiter = "\t", index_col=None)
        i = i.drop(['Unnamed: 0'], axis=1)
        
        hack = "2019.tab"
        with open(directory+hack) as f:
            j = pd.read_csv(f, delimiter = "\t", index_col=None)
            j = j.drop(['Unnamed: 0'], axis=1)
            _, j_2019 = monthly_mean(j)
        meanT_2018, meanT_2019, _2018, _2019 = split_df(i)
        
        frames = [_2019, j_2019]
        _2019_merged = pd.concat(frames)
        meanT_2019_new, j_2019_new = monthly_mean(_2019_merged)
        
        plot_precipitation(j_2019_new)
        
#        st.markdown (f'{[x for x in meanT_2018.iteritems()]}')
        st.markdown (f'Mean temperature for period of {_2018.Month.unique().min()}-{_2018.Month.unique().max()} in {_2018.Year.unique()} was {"%.1f" % _2018.Temp.mean()}')
        plot_temperature(_2018)
#        st.markdown (f'{[x for x in meanT_2019_new.iteritems()]}')
        st.markdown (f'Mean temperature for period of {j_2019_new.Month.unique().min()}-{j_2019_new.Month.unique().max()} in {j_2019_new.Year.unique()} was {"%.1f" % j_2019_new.Temp.mean()}')
        plot_temperature(j_2019_new)
        
for path_ in directory_.rglob('*.tab'):
    if path_.name == hack:
        continue
    with open(path_) as f:
        i = pd.read_csv(f, delimiter = "\t", index_col=None)
        i = i.drop(['Unnamed: 0'], axis=1)
        meanT, i = monthly_mean(i)

        # Find the "Year" with the most number of rows
        year_to_keep = i.groupby(['Year'])["Year"].count().idxmax()
        i = i[i["Year"] == year_to_keep]
        
#        print (meanT)
        st.markdown (f'Mean temperature for period of {i.Month.unique().min()}-{i.Month.unique().max()} in {i.Year.unique()} was {"%.1f" % i.Temp.mean()}')
        plot_temperature(i)