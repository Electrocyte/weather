import os
import os.path
import pandas as pd
import glob
import random
from dateutil import tz
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import re

# ================================================================================================
#     Only edit these two variables
#DIRECTORY = "D:\\Enter\\sample\\load\\data\\directory\\"
#OUTPUT = "D:\\Enter\\sample\\save\\data\\directory\\"
DIRECTORY = "D:\\James\\Documents\\Python Scripts\\"
OUTPUT = "D:\\James\\Documents\\Python Scripts\\"
# ================================================================================================

CITY_TZS = DIRECTORY+"CityTzs.csv"
with open(CITY_TZS, 'r') as myfile:
    CITY_TZS_FILE = pd.read_csv(CITY_TZS, index_col = "City_name")
#     CITY_TZS_FILE = pd.read_csv(CITY_TZS)

TAB_EXTENSION = ".tab"

cities_of_interest = [
#     "Adelaide",
#     "Auckland",
#     "Bayonne",
#     "Boston",
#     "Cherrapunji",
#     "City of Edinburgh",
#     "Edinburgh",
#     "Grenoble",
#     "Haikou",
#     "Helsinki",
#     "Kunming",
#     "Lausanne",
#     "Lorient",
#     "Manchester",
#     "Nelson",
#     "Okinawa"
#     "Paignton",
#     "Panama City",
#     "Penvenan",
#     "Perpignan",
#     "Port Moresby",
#     "Queenstown",
#     "Quito",
#     "Republic of Singapore",
#    "Rochecorbon",
#     "Saint-Geoire-en-Valdaine",
#     "Sanya",
    "Singapore",
#     "Tours",
#     "Vancouver",
#     "Whangerei",
#     "Zurich",
]

def colourMixer(mix):
    red = random.uniform(0, 1)
    green = random.uniform(0, 1)
    blue = random.uniform(0, 1)

    mixRed, mixGreen, mixBlue = mix

    finalRed = (red + mixRed)/2
    finalGreen = (green + mixGreen)/2
    finalBlue = (blue + mixBlue)/2

    return (finalRed, finalGreen, finalBlue)

def formatTime(timestamp, t_format, city_timezone):
    utc = datetime.fromtimestamp(timestamp, timezone.utc)
    city_timezone = tz.gettz(city_timezone)
    return utc.astimezone(city_timezone).strftime(t_format)

def load_dfs(directory, cities_of_interest):

    DfDic = {}

    for absolute_path in glob.glob(directory + "\\*.tab"):

        filename = os.path.basename(absolute_path)
        city = filename.split('.')[0]
        if city not in cities_of_interest:
            continue

        DfDic[filename] = pd.read_csv(absolute_path, sep='\t')

#         print (os.path.exists(absolute_path))

    return DfDic

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

#     Calculate 2 days moving average
#    moving_2d_average = df[['Temp']]
#    moving_2d_average = moving_2d_average.rolling("2d").mean()
#    moving_2d_average = moving_2d_average.rename(index=str,columns={'Temp':"2d_MA_Temperature"})
#    df = pd.concat([df, moving_2d_average], axis=1)

    return df

def min_max(df):

    year_month_day = df.groupby(['Year','Month','Day'])

    maxT = year_month_day['Temp'].max()
    maxT = maxT.rename("maxT")
    minT = year_month_day['Temp'].min()
    minT = minT.rename("minT")
    meanT = year_month_day['Temp'].mean()
    meanT = meanT.rename("meanT")

    maxHum = year_month_day['Humidity[%]'].max()
    maxHum = maxHum.rename("maxHum")
    minHum = year_month_day['Humidity[%]'].min()
    minHum = minHum.rename("minHum")
    meanHum = year_month_day['Humidity[%]'].mean()
    meanHum = meanHum.rename("meanHum")

    merge_df = pd.concat([meanT, minT, maxT, meanHum, minHum, maxHum], axis=1)
    merge_df = merge_df.reset_index()
    merge_df["maxT"] = pd.to_numeric(merge_df["maxT"], errors='coerce')
    merge_df["minT"] = pd.to_numeric(merge_df["minT"], errors='coerce')
    merge_df["meanT"] = pd.to_numeric(merge_df["meanT"], errors='coerce')
    merge_df["maxHum"] = pd.to_numeric(merge_df["maxHum"], errors='coerce')
    merge_df["minHum"] = pd.to_numeric(merge_df["minHum"], errors='coerce')
    merge_df["meanHum"] = pd.to_numeric(merge_df["meanHum"], errors='coerce')

    return merge_df

def plot_humidity(df):

    year_month = df.groupby(['Year','Month'])
    CityStr = df["City"][0]

    for (year, month), readings in year_month:
        days = list(readings['Day'].values)
        ax = sns.lineplot(x='Day', y='Humidity[%]', data=readings)
        ax.set_title('{} {}-{} \n Daily Average Humidity'.format(CityStr, year, month))
        ax.title.set_position([.5, 1.15])
        plt.xlabel('Day')
        plt.ylabel('Humidity (%)')
        plt.ylim((0,100))
        plt.xlim((0, 31))
        plt.show()

def plot_humidity_mean(df):

    year_month = df.groupby(['Year','Month'])
    CityStr = df["City"][0]

    merged_df = min_max(df)

    for (year, month), monthly_readings in year_month:

        min_max_mean = merged_df.loc[(merged_df['Month'] == month) & (merged_df['Year'] == year)]

        fig, ax = plt.subplots(1, 1)
        min_max_mean.plot(x="Day", y="maxHum", ax=ax, label="Max")
        min_max_mean.plot(x="Day", y="minHum", ax=ax, label="Min")
        min_max_mean.plot(x="Day", y="meanHum", ax=ax, label="Mean")

        ax.set_title('{} {}-{} \n Daily Average Humidity'.format(CityStr, year, month))
        ax.title.set_position([.5, 1.15])
        plt.xlabel('Day')
        plt.ylabel('Humidity (%)')
        plt.ylim((0,101))
        plt.xlim((0, 31))
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size" : 7})
        plt.show()

def plot_humidity_daily(df):

    year_month = df.groupby(['Year','Month'])
    CityStr = df["City"][0]


    for (year, month), monthly_readings in year_month:

        fig, ax = plt.subplots(1, 1)
        ax.set_title('{} \n {}-{} Daily Average Humidity'.format(CityStr, year, month))
        ax.title.set_position([.5, 1.15])
        plt.xlabel('Hour')
        plt.ylabel('Humidity (%)')

        for dayy, readings in monthly_readings.groupby(['Day']):

            readings.plot(x="FloatHour", y="Humidity[%]", ax=ax, label=dayy)
            plt.xlabel('Hour')

        plt.ylim((0,101))
        plt.xlim((0, 24))
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size" : 7})
        plt.show()

def plot_temperature_mean(df):

    year_month = df.groupby(['Year','Month'])
    CityStr = df["City"][0]

    merged_df = min_max(df)

    for (year, month), monthly_readings in year_month:

        min_max_mean = merged_df.loc[(merged_df['Month'] == month) & (merged_df['Year'] == year)]

        fig, ax = plt.subplots(1, 1)
        min_max_mean.plot(x="Day", y="maxT", ax=ax, label="Max")
        min_max_mean.plot(x="Day", y="minT", ax=ax, label="Min")
        min_max_mean.plot(x="Day", y="meanT", ax=ax, label="Mean")

        ax.set_title('{} \n {}-{} Daily Average Temperature'.format(CityStr, year, month))
        ax.title.set_position([.5, 1.15])
        plt.xlabel('Day')
        plt.ylabel('Temperature (°C)')
        plt.ylim((0, 40))
        plt.xlim((0, 31))
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size" : 7})
        plt.show()

def plot_temperature_daily(df):

    year_month = df.groupby(['Year','Month'])
#     year_month_day = df.groupby(['Year','Month','Day'])
    CityStr = df["City"][0]

    for (year, month), monthly_readings in year_month:

        fig, ax = plt.subplots(1, 1)
        ax.set_title('{} \n {}-{} Daily Temperature'.format(CityStr, year, month))
        ax.title.set_position([.5, 1.15])
        plt.xlabel('Hour')
        plt.ylabel('Temperature (°C)')

        for dayy, readings in monthly_readings.groupby(['Day']):

            readings.plot(x="FloatHour", y="Temp", ax=ax, label=dayy)
            plt.xlabel('Hour')
#         plt.ylim((0, 40))
        plt.xlim((0, 24))
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size" : 7})
        plt.show()

# Mostly unused because spread given does not include outliers i.e. min-max

#def plot_temperature_moving_average(df):
#
#    year_month = df.groupby(['Year','Month'])
#    CityStr = df["City"][0]
#
#    for (year, month), readings in year_month:
#        days = list(readings['Day'].values)
#
#        fig, ax = plt.subplots(1, 1)
##         ax = sns.lineplot(x='Day', y='2d_MA_Temperature', data=readings)
#        df.plot(x="Time", y="Temp", ax=ax, label="Temperature")
#        df.plot(x="Time", y="2d_MA_Temperature", ax=ax, label="2D_Moving Average")
#
#        ax.set_title('{} {}-{} \n Two Day Moving Average Temperature'.format(CityStr, year, month))
#        ax.title.set_position([.5, 1.15])
#        plt.xlabel('Day')
#        plt.ylabel('Temperature (°C)')
##         plt.ylim((0,40))
#        plt.xlim((min(days), max(days)))
#        plt.xlim((0, 31))
#        plt.xticks(rotation=90)
#        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size" : 7})
#        plt.show()

def moving_average_5_day(df):
    year_month_day = df.groupby(['Year','Month','Day'])
    day_list = []
    for (year, month, day), readings in year_month_day:
        meanT = year_month_day['Temp'].mean()
        day_list.append(day)
    average_daily = meanT
    new_df = pd.DataFrame()
    new_df['Time'] = meanT.index.tolist()
    new_df['Day'] = day_list
    new_df['D_average_Temp'] = average_daily.tolist()
    n = len(meanT) # total number of days in series
    k = 5 # moving average number
    for i in range(n-k+1):
        meanT[i:i+k] = meanT[i:i+k].mean()
    new_df[f'MA_{k}_Temp'] = meanT.tolist()
#    new_df['MA_{k}_Temp'.format(k=k)] = meanT.tolist() # must use this jupyter for now
    new_df.iloc[:, 3] = new_df.iloc[:, 3].shift(k)
#    new_df = new_df.set_index('Time')
    return new_df

def plot_temperature_moving_average(df, meanT):
    CityStr = df["City"][0]
    x = list((list(meanT))[i] for i in [3])
    k = ''.join(re.findall(r'\d+', str(x)))
    fig, ax = plt.subplots(1, 1)
    meanT.plot(x="Time", y="D_average_Temp", ax=ax, label="Daily_Average")
    meanT.plot(x="Time", y=f"MA_{k}_Temp", ax=ax, label=f"{k}Day_Moving Average")
    ax.set_title('{} \n {} Day Moving Average Temperature'.format(CityStr, k))
    ax.title.set_position([.5, 1.15])
    plt.xlabel('Day')
    plt.ylabel('Temperature (°C)')
    plt.xticks(rotation=90)
#    ax.set_xticklabels(meanT['Day'], rotation=90)
    plt.ylim((-10, 35))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size" : 7})
    plt.show()

#     graph setup
barColour = colourMixer((1, 1, 1))
font = {'family' : 'DejaVu Sans',
            'weight' : 'bold',
            'size'   : 22}

plt.rc('font', **font)
sns.set_context("notebook", font_scale=2.0, rc={"lines.linewidth": 2.5})

files = load_dfs(DIRECTORY, cities_of_interest)



#     Filename is not used BUT still need it because a dictionary item has a KEY and VALUE
for _FileName, df in sorted(files.items()):
    city = df['City'][0]
    time__zone = CITY_TZS_FILE.loc[city]['Time_Zone']
    df = time_columns(df, time__zone)
    min_max(df)
    move_av = moving_average_5_day(df)
#    plot_humidity(df)
#    plot_humidity_mean(df)
#    plot_humidity_daily(df)
#    plot_temperature_mean(df)
    plot_temperature_daily(df)
    plot_temperature_moving_average(df, move_av)
#    plot_temperature_moving_average(df)
