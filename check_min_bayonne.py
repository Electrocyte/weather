# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 15:08:00 2018

@author: Mangifera
"""
from datetime import datetime, timezone
from dateutil import tz
import pandas as pd

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

#     Set index as a datetime
    df['New_time'] = pd.to_datetime(df['UNIX_UTC'], unit='s')
    df = df.set_index('New_time')
    df = df.sort_index()

    return df

# ================================================================================================
DIRECTORY = "D:\\James\\Documents\\Python Scripts\\"
file_names = ["Bayonne.tab", "B_2019.tab", "B_2020.tab"]
#file_names = ["Penvenan.tab", "P_2019.tab", "P_2020.tab"]
#file_names = ["Rochecorbon.tab", "R_2019.tab", "R_2020.tab"]

for file_name in file_names:
    df = DIRECTORY+file_name
    tzz = DIRECTORY+"CityTzs.csv"
    
    city_ = pd.read_csv(df, sep = "\t")
    CITY_TZS_FILE = pd.read_csv(tzz, sep = ",")
    CITY_TZS_FILE = CITY_TZS_FILE.set_index('City_name')
    
    city = city_["City"][0]
    time__zone = CITY_TZS_FILE.loc[city]['Time_Zone']
    updated_df = time_columns(city_, time__zone)
    
    #drop duplicates method
    less_days = updated_df[updated_df["Temp"] < 1]
    edited = less_days.sort_values('Temp').drop_duplicates(subset='Day', keep='first')
    edited = edited.sort_index()
    
    #groupby method
    attempt = updated_df.groupby(["Day","Month","Year"])
    min_att = attempt.min()
    Min_temp = 1
    cold = min_att[min_att['Temp'] < Min_temp]
    
    cold_index = cold.set_index("Time")
    cold_clean = cold_index.sort_index().loc[:,"Temp"]
    print (cold_clean)
    print (f"Days ({file_name}) below {Min_temp}C: {len(cold_clean)}")
    
    #groupby method
    attempt_hot = updated_df.groupby(["Day","Month","Year"])
    max_att = attempt_hot.max()
    Max_temp = 30
    hot = max_att[max_att['Temp'] > Max_temp ]
    hot_index = hot.set_index("Time")
    hot_clean = hot_index.sort_index().loc[:,"Temp"]
    
    print (f"Days above {Max_temp}C: {len(hot_clean)}")