#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:00:15 2020

@author: matthewayala
"""

from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# NBA Season to analyze
year = 2021

# URL Page to be extracted
url = "https://www.basketball-reference.com/leagues/NBA_{}_games.html".format(year)

# HTML from the given URL
html = urlopen(url)

soup = BeautifulSoup(html, features = 'lxml')

# Using FindAll to get column headers
soup.findAll("tr",limit = 2)

# Using getText to extract the text needed
headers = [th.getText() for th in soup.findAll("tr",limit = 2)[0].findAll('th')]

headers = headers[1:]


# Extracting the data for the rows including dates of games.
rows = soup.findAll("tr")[1:]

game_stats =[[td.getText() for td in rows[i].findAll('td')]
             for i in range(len(rows))]

game_dates = [[th.getText() for th in rows[i].findAll('th')]
              for i in range(len(rows))]

# Putting data and column headers into a dataframe and dropping Boxscore,Notes,OT Columns
stats = pd.DataFrame(game_stats,columns = headers)

# Reorganizng columns, putting date as the first column.
stats.insert(0,'Date',game_dates)
stats.drop(stats.columns[[6,9]], axis = 1, inplace = True)

# Renaming columns for accessibility and readability
stats.columns = ['Date','Start(ET)','Away','Away PTS','Home','Home PTS','Attend.']

# Dropping commas from dataframe so columns can be read as float64
stats = stats.replace(',','', regex=True)

# Converting columns into the correct data types.
stats['Date'] = pd.to_datetime(stats.Date.apply(lambda x: x[0]))
stats.iloc[:,3] = pd.to_numeric(stats.iloc[:,3])
stats.iloc[:,5] = pd.to_numeric(stats.iloc[:,5])
stats.iloc[:,6] = pd.to_numeric(stats.iloc[:,6])

# For some reason applying pd.to_numeric to all the columns at once was changing values
#stats.iloc[:,[3,5,6]] = stats.iloc[:,[3,5,6]].apply(pd.to_numeric)

# Adding a column for the winner of the game
stats['Winner'] = np.where(stats['Home PTS'] > stats['Away PTS'],stats['Home'],stats['Away'])

with pd.ExcelWriter('nba_scraping.xlsx') as writer:
    stats.to_excel(writer, sheet_name='Main NBA Data')
   # nba_scoring_averages.to_excel(writer, sheet_name='HOME-ROAD Splits')




