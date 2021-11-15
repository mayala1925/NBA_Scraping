# Importing Libraries
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import date

# def get_team_ratings(date):
year = 2022
today= date.today()

# URL to open
url = f"https://www.basketball-reference.com/leagues/NBA_{year}_ratings.html"

html = urlopen(url)
soup = BeautifulSoup(html, features = 'lxml')
soup.findAll("tr")

# Extracting data that we need
rows = soup.findAll("tr")[1:]
team_stats = [[td.getText() for td in rows[i].findAll('td')]
                for i in range(len(rows))]

team_headers = [[th.getText() for th in rows[i].findAll('th')]
                for i in range(len(rows))]

# Changing headers into an actual list instead of a list of 1.
cols = team_headers[:1]
cols = ','.join(cols[0])
cols = cols.split(',')
cols = cols[1:]

team_stats = team_stats[1:31]

# Creating ratings dataframe
ratings_df = pd.DataFrame(columns = cols, data = team_stats)

# Dropping unneeded columns.
ratings_df = ratings_df.drop(labels  = ['Conf','Div'], axis = 1)

# Creating dictionary for each days team ratings: Starting 11/15/21
# ratings_dict = {}
#
# ratings_dict[today] = ratings_df
#
# dict_df = pd.DataFrame.from_dict(ratings_dict)

ratings_df.to_csv(f'ratings/{today}_ratings.csv',index = False)



