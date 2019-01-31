'''
     File name: hw2_eda_gdotan.py
     Author: Guy Dotan
     Date: 01/27/2019
     Course: UCLA Stats 404
     Description: HW #2. Exploratory data analysis on the NBA stats and social media dataset.
 '''

import pandas as pd
import glob

path = 'Repository/HWs/HW2/social-power-nba'
filelist = glob.glob(path + "/*.csv")
filename = [s.replace('.csv', '').replace(path,'') for s in filelist]

# creates dictionary of dataframes
dfs = {}
for x in range(len(filename)):
    dfs[filename[x]] = pd.read_csv(filelist[x])


