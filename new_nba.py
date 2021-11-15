# Importing libraries
import pandas as pd
import numpy as np
from datetime import date

today = date.today()

# Reading in data and preprocessing
ratings_data = pd.read_csv(f'ratings/{today}_ratings.csv')

# Creating empty dictionary

ratings_dict = {}

ratings_dict[today] = ratings_data






