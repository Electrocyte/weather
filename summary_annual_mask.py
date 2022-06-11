#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:36:03 2022

@author: mangi
"""

import pandas as pd
import glob


def convert(F):
    Celcius=(F-32)*5/9
    return Celcius


directory = "E:/weather/new/allsites/"

file = "FRBRDAUX.txt"

find_file = glob.glob(f"{directory}{file}")

df = pd.read_csv(find_file[0], delimiter = "\t", header = None)
df = df.replace(r'  *', '-', regex=True)
df[['empty', 'Month', 'Day', 'Year', 'Temp-F']] = df[0].str.split('-', 4, expand=True)
df = df.drop([0, 'empty'], axis = 1)
df = df.loc[~df["Temp-F"].str.contains('99')]
df.reset_index(inplace = True, drop = True)

df['Temp-F'] = df['Temp-F'].astype(float)

df['Temp-C'] = df['Temp-F'].apply(convert)