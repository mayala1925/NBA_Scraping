import pandas as pd
import numpy as np
import sklearn

data = pd.read_csv('nba_scrapping_data/nba_stats.csv')


# Drop rows with null values, these are the rows that were pulled that the games haven't been played yet...
classifier_data = nba_stats.dropna()

print(classifier_data.isnull().sum())
print(data.head())
