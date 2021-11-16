import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('nba_scrapping_data/nba_stats.csv')

# Drop rows with null values, these are the rows that were pulled that the games haven't been played yet...
classifier_data = data.dropna()

# Scaling the data

# Need to reduce hte scope to one team, stregnth, days rest, point diff