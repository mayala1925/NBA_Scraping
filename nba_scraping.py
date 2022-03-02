#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:00:15 2020

@author: matthew ayala
"""

from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# NBA Season to analyze
# year = 2022  # Season
months = ['october', 'november', 'december', 'january', 'february','march']  # Months of the season


def get_stats(year, month):
    # Creating an empty dataframe to append stats too.
    cols = ['date', 'start', 'away', 'away_pts', 'home', 'home_pts', 'attend']
    final_df = pd.DataFrame(columns=cols)

    for i in month:
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{i}.html"

        # HTML from the given URL
        html = urlopen(url)
        soup = BeautifulSoup(html, features='lxml')
        soup.findAll("tr", limit=2)

        # Using getText to extract the text needed
        headers = [th.getText() for th in soup.findAll("tr", limit=2)[0].findAll('th')]

        headers = headers[1:]

        # Extracting the data for the rows including dates of games.
        rows = soup.findAll("tr")[1:]

        game_stats = [[td.getText() for td in rows[i].findAll('td')]
                      for i in range(len(rows))]

        game_dates = [[th.getText() for th in rows[i].findAll('th')]
                      for i in range(len(rows))]

        # Putting data and column headers into a dataframe and dropping Boxscore,Notes,OT Columns
        stats_df = pd.DataFrame(game_stats, columns=headers)

        # Reorganizing columns, putting date as the first column.
        stats_df.insert(0, 'Date', game_dates)
        stats_df.drop(stats_df.columns[[6, 9]], axis=1, inplace=True)

        # Renaming columns for accessibility and readability
        stats_df.columns = cols

        # Dropping commas from dataframe so columns can be read as float64
        stats_df = stats_df.replace(',', '', regex=True)

        # Converting columns into the correct data types.
        stats_df['date'] = pd.to_datetime(stats_df.date.apply(lambda x: x[0]))
        stats_df.iloc[:, 3] = pd.to_numeric(stats_df.iloc[:, 3])
        stats_df.iloc[:, 5] = pd.to_numeric(stats_df.iloc[:, 5])
        stats_df.iloc[:, 6] = pd.to_numeric(stats_df.iloc[:, 6])

        # Adding a winner column
        stats_df['winner'] = np.where((stats_df['away_pts'] > stats_df['home_pts']), 'away', 'home')

        # Append months stats to the previous stats.
        final_df = final_df.append(stats_df)

    return final_df


nba_stats = get_stats(2022, months)

# Writing the data into a csv for classifier use.
nba_stats.to_csv('nba_scrapping_data/nba_schedule.csv', index=False)

# Option to write to an excel file (NEED TO EDIT).
# with pd.ExcelWriter('nba_scraping.xlsx') as writer:
#     nba_stats.to_excel(writer, sheet_name='Main NBA Data')
