# Importing libraries
import pandas as pd
import numpy as np
from datetime import date, timedelta
import matplotlib.pyplot as plt

today = date.today()

# Reading in data and preprocessing
ratings_data = pd.read_csv(f'ratings/{today}_ratings.csv')

# Creating empty dictionary
ratings_dict = {}

# Creating time delta to read in data from a stretch of days.
sdate = date(2021, 11, 15)   # start date
edate = date(2021, 11, 16)   # end date

delta = edate - sdate       # as timedelta

# Reading in data through time delta and creating dictionary
for i in range(delta.days + 1):
    day = sdate + timedelta(days=i)
    ratings_df = pd.read_csv(f'ratings/{day}_ratings.csv')

    # Putting data in a dictionary with the date as the key and dataframe as the value
    ratings_dict[day] = ratings_df










