# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:20:26 2020

@author: Mangifera
"""
from pathlib import Path
import pandas as pd
from dateutil import tz
import seaborn as sns; sns.set()
from datetime import datetime, timezone
#import os.path
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
    city = i['City'][0]
    time__zone = CITY_TZS_FILE.loc[city]['Time_Zone']    
    i = time_columns(i, time__zone)    
    year_month = i.groupby(['Year','Month'])
    maxT = year_month['Temp'].max()
    maxT = maxT.rename("maxT")
    minT = year_month['Temp'].min()
    minT = minT.rename("minT")
    meanT = year_month['Temp'].mean()
    meanT = meanT.rename("meanT")

    return meanT


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

    return meanT


#===============================================================================
directory = "D:/James/Documents/SpiderOak Hive/Data/weather/new_system/Bayonne/"
directory = Path(directory)

old = "D:/James/Documents/SpiderOak Hive/Data/weather/a_older_system/"
old_bayonne = old+"Bayonne.tab"

DIRECTORY = "D:\\James\\Documents\\Python Scripts\\"
CITY_TZS = DIRECTORY+"CityTzs.csv"
with open(CITY_TZS, 'r') as myfile:
    CITY_TZS_FILE = pd.read_csv(CITY_TZS, index_col = "City_name")
#===============================================================================    


if path.exists(old_bayonne):
    with open(old_bayonne) as f:
        i = pd.read_csv(f, delimiter = "\t", index_col=None)
        i = i.drop(['Unnamed: 0'], axis=1)
        meanT = monthly_mean_ori(i)
        print (meanT)
        print (f'Mean temperature for period of {i.Month.unique()} in {i.Year.unique()} was {i.Temp.mean()}')

for path_ in directory.rglob('*.tab'):
    with open(path_) as f:
        i = pd.read_csv(f, delimiter = "\t", index_col=None)
        i = i.drop(['Unnamed: 0'], axis=1)
        meanT = monthly_mean(i)

        # Find the "Year" with the most number of rows
        year_to_keep = i.groupby(['Year'])["Year"].count().idxmax()
        i = i[i["Year"] == year_to_keep]
        
        print (meanT)
        print (f'Mean temperature for period of {i.Month.unique()} in {i.Year.unique()} was {i.Temp.mean()}')
#








#for path in directory.rglob('*.tab'):
#    with open(path) as f:
#        i = pd.read_csv(f, delimiter = "\t", index_col=None)
#        i = i.drop(['Unnamed: 0'], axis=1)
#        city = i['City'][0]
#        time__zone = CITY_TZS_FILE.loc[city]['Time_Zone']
#        
#        time_columns(i, time__zone)
#        
#        year_month = i.groupby(['Year','Month'])
#        maxT = year_month['Temp'].max()
#        maxT = maxT.rename("maxT")
#        minT = year_month['Temp'].min()
#        minT = minT.rename("minT")
#        meanT = year_month['Temp'].mean()
#        meanT = meanT.rename("meanT")
#        
#        print (meanT)





