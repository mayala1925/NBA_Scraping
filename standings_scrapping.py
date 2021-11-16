from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import janitor

year = 2022

# url for standings webpage
url = f"https://www.basketball-reference.com/leagues/NBA_{year}_standings.html"

# HTML from the given url
html = urlopen(url)
soup = BeautifulSoup(html, features = 'lxml')
soup.findAll("tr", limit=2)

# Using getText to extract the text needed
headers = [th.getText() for th in soup.findAll("tr", limit=2)[0].findAll('th')]

# Eastern and Western conference headers
headers_east = headers[1:8]
headers_west = headers[73:80]

# Getting data for each conference
rows = soup.findAll("tr")[1:]

game_stats = [[td.getText() for td in rows[i].findAll('td')]
                      for i in range(len(rows))]

east_stats = game_stats[1:16]
west_stats = game_stats[17:32]

# Creating dataframes
east_df = pd.DataFrame(east_stats,columns=headers_east)
west_df = pd.DataFrame(west_stats,columns=headers_west)


# Changing column names for accessibility and readability
east_df = east_df.clean_names()
west_df = west_df.clean_names()








