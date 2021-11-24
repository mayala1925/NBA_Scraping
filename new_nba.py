# Importing libraries
import pandas as pd
import numpy as np
from datetime import date, timedelta


# Reading in the updated schedule from nba_scrapping
schedule = pd.read_csv('nba_scrapping_data/nba_schedule.csv')

# Creating empty dictionary
ratings_dict = {}

# Creating time delta to read in data from a stretch of days.
today = date.today()

sdate = date(2021, 11, 15)   # start date
edate = today   # end date (today)

delta = edate - sdate       # as timedelta

# Reading in data through time delta and creating dictionary
for i in range(delta.days + 1):
    day = sdate + timedelta(days=i)
    ratings_df = pd.read_csv(f'ratings/{day}_ratings.csv')

    # Putting data in a dictionary with the date as the key and dataframe as the value
    ratings_dict[day] = ratings_df

ratings_dict= {np.datetime64(k):(v) for k,v in ratings_dict.items()}

schedule['date'] = pd.to_datetime(schedule['date'])

# Filtering data starting on 11/15 because that is when I started collecting ratings data.
schedule = schedule[schedule['date'] >= '11-15-21']

# Creating a function that will make dictionaries for each stat to be mapped to the new dataframe
def make_stat_dictionary(df,stat):
    stat_dict = df[['Team',stat]].set_index('Team').T.to_dict('list')
    return stat_dict

# Matching schedule data with ratings data.

# Looping through ratings dictionary and mapping to the schedule dataframe for each statistic
for key,value in ratings_dict.items():
    # Margin of Victory (MOV)
    schedule.loc[schedule['date'] == key, 'a_mov'] = schedule['away'].map(make_stat_dictionary(value,'MOV'))
    schedule.loc[schedule['date'] == key, 'h_mov'] = schedule['home'].map(make_stat_dictionary(value,'MOV'))

    # Offensive Rating (ORtg)
    schedule.loc[schedule['date'] == key, 'a_ORtg'] = schedule['away'].map(make_stat_dictionary(value,'ORtg'))
    schedule.loc[schedule['date'] == key, 'h_ORtg'] = schedule['home'].map(make_stat_dictionary(value,'ORtg'))

    # Defensive Rating (DRtg)
    schedule.loc[schedule['date'] == key, 'a_DRtg'] = schedule['away'].map(make_stat_dictionary(value,'DRtg'))
    schedule.loc[schedule['date'] == key, 'h_DRtg'] = schedule['home'].map(make_stat_dictionary(value,'DRtg'))

    # Net Rating (NRtg)
    schedule.loc[schedule['date'] == key, 'a_NRtg'] = schedule['away'].map(make_stat_dictionary(value,'NRtg'))
    schedule.loc[schedule['date'] == key, 'h_NRtg'] = schedule['home'].map(make_stat_dictionary(value,'NRtg'))

    # Margin of Victory/Adjusted for strength of opponent (MOV/A)
    schedule.loc[schedule['date'] == key, 'a_mov/a'] = schedule['away'].map(make_stat_dictionary(value,'MOV/A'))
    schedule.loc[schedule['date'] == key, 'h_mov/a'] = schedule['home'].map(make_stat_dictionary(value,'MOV/A'))

    # Offensive Rating/ Adjusted for opponent defense (ORtg/A)
    schedule.loc[schedule['date'] == key, 'a_ORtg/a'] = schedule['away'].map(make_stat_dictionary(value,'ORtg/A'))
    schedule.loc[schedule['date'] == key, 'h_ORtg/a'] = schedule['home'].map(make_stat_dictionary(value,'ORtg/A'))

    # Defensive Rating/ Adjusted for opponent offense (DRtg/A)
    schedule.loc[schedule['date'] == key, 'a_DRtg/a'] = schedule['away'].map(make_stat_dictionary(value,'DRtg/A'))
    schedule.loc[schedule['date'] == key, 'h_DRtg/a'] = schedule['home'].map(make_stat_dictionary(value,'DRtg/A'))

    # Net Rating/ Adjusted for opponent strength (NRtg/A)
    schedule.loc[schedule['date'] == key, 'a_NRtg/a'] = schedule['away'].map(make_stat_dictionary(value,'NRtg/A'))
    schedule.loc[schedule['date'] == key, 'h_NRtg/a'] = schedule['home'].map(make_stat_dictionary(value,'NRtg/A'))



# Cleaning up the columns and rearranging to how I want it.
schedule = schedule.drop(['start','attend'],axis = 1)
schedule_cols = ['date', 'away', 'away_pts', 'home', 'home_pts', 'a_mov', 'h_mov',
                 'a_ORtg', 'h_ORtg', 'a_DRtg', 'h_DRtg', 'a_NRtg', 'h_NRtg',
                 'a_mov/a','h_mov/a','a_ORtg/a','h_ORtg/a','a_DRtg/a','h_DRtg/a','a_NRtg/a','h_NRtg/a','winner',]
schedule = schedule[schedule_cols]

# Dropping rows for data that we don't have yet. (today + 1 day)
schedule = schedule.drop(schedule[schedule['date'] > np.datetime64(edate)].index)

# Resetting the index
schedule = schedule.reset_index(drop = True)

# Function that to convert stat values to floats because they were in lists of 1
def take_out_list(column):
    replace_col = []
    for i in column:
        val = i[0]
        replace_col.append(val)

    return replace_col

# looping through each column that needs to be converted from a list
for i in schedule.columns[5:21]:
   schedule[i] = take_out_list(schedule[i])

schedule.to_csv(f'nba_scrapping_data/schedule_ratings.csv',index = False)




# print(ratings_df.info())
# print(schedule.info())











