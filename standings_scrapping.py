from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

year = 2022

# url for standings webpage
url = f"https://www.basketball-reference.com/leagues/NBA_{year}_standings.html"

# HTML from the given url
html = urlopen(url)
        soup = BeautifulSoup(html, features = 'lxml')
        soup.findAll("tr", limit=2)