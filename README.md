# NBA_Scraping
Project to try and scrap data from basketball reference and create a dataframe that with features and labels to eventually model win/loss.

## First

Run nba_scrapping.py to grab the updated schedule of games up to the current date.
This will return a csv file to the "nba_scrapping_data" directory that is a dateframe of the games that have been played, the teams that played, the dates of the games, scores, and who won or lost the game.

This dataframe (schedule) is the will be the main dataframe being modifed and added to for features and labels.

## Second

Run team_ratings.py to grab the team ratings on the specific day that it is being run. (e.g. Run it on 2021/11/18 and it will grab the team ratings and statistics for that 2021/11/18)

This will create a csv file for that date and put it into a the "ratings" directory.

Throughout the season this ratings directory should hold team ratings for every day that it is run (hopefully for each day of the season) starting on 2021/11/15. 

### Third
Run new_nba.py (will eventually change the name of this script).

This will extract the ratings from each rating.csv in the "ratings" directory and create a dictionary with the keys being the date of the ratings and the value being the entire dateframe of ratings on that date (ratings_dict). This dictionary is then used to map the ratings on each specific date to home and away teams in the schedule dateframe. 

This is final dateframe that will be used to model.
