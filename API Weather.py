# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 21:07:33 2018

@author: Mangifera
"""

# http://api.openweathermap.org/data/2.5/find?lat=55.5&lon=37.5&cnt=10
# Gets 10 cities in a circle using lat and long as centrepoint

import requests
import pandas as pd
from datetime import datetime
from dateutil import tz
import schedule
import time
import toml
import os


def current_weather(api_key, Place_id, saveme, tab_extension):
    urlStart = "http://api.openweathermap.org/data/2.5/weather?id="
    urlFin = "&units=metric&APPID="+api_key
    print (urlFin)

    for i in Place_id:

        Full_url = urlStart+i+urlFin
        response1 = requests.get(Full_url)
        WeatherAll = response1.json()

        WeatherAll['dtN'] = (datetime.utcfromtimestamp(
            WeatherAll['dt']).strftime('%Y-%m-%d %H:%M:%S'))
        current_year = (datetime.utcfromtimestamp(
            WeatherAll['dt']).strftime('%Y'))
        
        CityStr = WeatherAll["name"]
        DescStr = WeatherAll['weather'][0]['description']
        IconStr = WeatherAll['weather'][0]['icon']
        MainStr = WeatherAll['weather'][0]['main']

        WeatherAll['Rain[3h]'] = '0'
        WeatherAll['Rain[1h]'] = '0'
        WeatherAll['Snow[3h]'] = '0'
        WeatherAll['Snow[1h]'] = '0'
        WeatherAll['Visibility'] = '0'
        WeatherAll["Wind_direction"] = '0'

        # Check if any rain data - if yes collect, if not ... moving on !

        if "rain" in WeatherAll:
            print(f"{CityStr} Rain")

            if "3h" in WeatherAll["rain"]:
                Rain3hList = WeatherAll["rain"]["3h"]
                WeatherAll['Rain[3h]'] = Rain3hList

                print("The heavens have opened in", CityStr,
                      ":", Rain3hList, "mm in the last 3h.")

            if "1h" in WeatherAll["rain"]:
                Rain1hList = WeatherAll["rain"]["1h"]
                WeatherAll['Rain[1h]'] = Rain1hList

                print("The heavens have opened in", CityStr,
                      ":", Rain1hList, "mm in the last 1h.")

        # Check if any snow data - if yes collect, if not ... moving on !
#
        if "snow" in WeatherAll:
            print(f"{CityStr} Snow")

            if "3h" in WeatherAll["snow"]:
                Snow3hList = WeatherAll["snow"]["3h"]
                WeatherAll['Snow[3h]'] = Snow3hList

                print("The heavens have opened in", CityStr,
                      ":", Snow3hList, "mm in the last 3h.")

            if "1h" in WeatherAll["snow"]:
                Snow1hList = WeatherAll["snow"]["1h"]
                WeatherAll['Snow[1h]'] = Snow1hList

                print("The heavens have opened in", CityStr,
                      ":", Snow1hList, "mm in the last 1h.")

        # Check if any visibility data - if yes collect, if not ... moving on !
        try:
            if WeatherAll["visibility"]:
                VisibList = WeatherAll["visibility"]
                WeatherAll['Visibility'] = VisibList
        except:
            pass

        # Check wind direction
        if "wind" in WeatherAll["wind"] and "deg" in WeatherAll["wind"]:
            WindDList = WeatherAll["wind"]["deg"]
            WeatherAll['Wind_direction'] = WindDList

        # Generate pandas df
        weather = pd.DataFrame({

            "City":          CityStr,
            "Description":   DescStr,
            "Icon":          IconStr,
            "Main_Weather":  MainStr,
            "ID":            WeatherAll['id'],
            "Visibility[m]": WeatherAll["Visibility"],
            "Humidity[%]":   WeatherAll["main"]["humidity"],
            "Pressure[hPa]": [WeatherAll['main']["pressure"]],
            "Time":          [WeatherAll['dtN']],
            "UNIX_UTC":      [WeatherAll['dt']],
            "Rain[3h][mm]":  [WeatherAll["Rain[3h]"]],
            "Rain[1h][mm]":  [WeatherAll["Rain[1h]"]],
            "Snow[3h][mm]":  [WeatherAll["Snow[3h]"]],
            "Snow[1h][mm]":  [WeatherAll["Snow[1h]"]],
            "Min_temp":      [WeatherAll['main']["temp_min"]],
            "Max_temp":      [WeatherAll['main']["temp_max"]],
            "Temp":          [WeatherAll['main']["temp"]],
            "Country":       [WeatherAll['sys']['country']],
            "Sunrise":       [WeatherAll['sys']['sunrise']],
            "Sunset":        [WeatherAll['sys']['sunset']],
            "Clouds[%]":     [WeatherAll['clouds']['all']],
            "Wind_direction": [WeatherAll["Wind_direction"]],
            "Wind_speed[m/s]": [WeatherAll["wind"]["speed"]],
            "Latitude":      [WeatherAll["coord"]["lat"]],
            "Longitude":     [WeatherAll["coord"]["lon"]]

        })

        # Files will be saved as <saveme>/<city>/<year>.<tab_extension>
        folder_name = os.path.join(saveme, CityStr)
        os.makedirs(folder_name, exist_ok=True) 
        
        file_name =  os.path.join(folder_name, f"{current_year}.{tab_extension}")

        if os.path.isfile(file_name):
            weather.to_csv(file_name, sep='\t', mode='a', header=False)
        else:
            weather.to_csv(file_name, sep='\t')

        #print (weather)
        from_zone = tz.gettz('UTC')
        to_zone = tz.gettz('Asia/Singapore')
        utc = datetime.utcnow()
        utc = utc.replace(tzinfo=from_zone)
        singapore = utc.astimezone(to_zone)

        #print ("Data for",CityStr,"at",datetime.utcnow(),"(UTC).")
        print("Data for", CityStr, "at", singapore, ".")

        print("Temperature at", CityStr, "is currently",
              WeatherAll['main']["temp"], "celsius.")


def load_config(file):
    return toml.load(file)


def main():
    config_file = os.getenv('CONFIG_FILE', "config.toml")
    config = load_config(config_file)
#    schedule.every(30).seconds.do(current_weather, **config)
#    schedule.every(2).hours.do(current_weather, **config)
#    schedule.every(1).hour.do(current_weather, **config)
    schedule.every(30).minutes.do(current_weather, **config)

    while True:
        schedule.run_pending()
        time.sleep(5)


if __name__ == "__main__":
    main()

#Full_url = []
# for i in Place_id:
#    Full_url.append(urlStart+i+urlFin)

#weatherNowAll = []
# for i in Full_url:
#    response1 = requests.get(i)
#    weatherNowAll.append(response1.json())

# list comprehension
#Full_url = [urlStart+i+urlFin for i in Place_id]
#weatherNowAll = [requests.get(i).json() for i in Full_url]

# Reorder columns in pandas df
#weather = weather[["City", "Humidity","Pressure","temp", "Min temp", "Max temp","Time"]]
