# Predicting CONCACAF's World Cup Qualifiers
Project by: Aaron O'Neal
Date: 01/14/2022

---

### Overview and Objectives


**Problem Statement**

Predicting the score of soccer matches, let alone the outcome (win, loss, draw), has proven soccer to be one of the most difficult sports to predict.  However, sports-betters still look for opportunities to enhance their predictions on specific games or series of games in hopes of making some money on different bets.


**Excecutive Summary**

This project will look at taking data of previously played international matches and use neural networks with a linear regression output in an attempt to predict the scores of the home and away teams for CONCACAF's 2022 World Cup Qualifying matches.  Data on international team's FIFA ranking as well as other engineered features will be used to predict the scores.  Those predictions will be fed into a new table that will predict the three teams that qualify for the World Cup and the fourth team that will participate in a play-in game against an opponent from either the Asia, Oceania, or South American qualifying regions.


The notebooks in this repository are numbered in order.  The data folder contains all .csv files that were used and created throughout this project and the images folder contains the visuals that were saved and used during presentation.

---

### The data


**Data folder files:**

[`results.csv`](./data/results.csv):  International fixtures and scores dating back to 1872.  Obtained from https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017?select=results.csv.
[`concacaf_wcq_fixtures.csv`](./data/concacaf_wcq_fixtures.csv):  Upcoming and previously played CONCACAF World Cup Qualifying fixtures.  Obtained from https://www.fifa.com/tournaments/mens/worldcup/qatar2022/qualifiers/concacaf.
[`fifa_ranking_up_to_2021-05-27.csv`](./data/fifa_ranking_up_to_2021-05-27.csv):  World FIFA rankings prior to 05/27/2021.  Obtained from https://www.kaggle.com/cashncarry/fifaworldranking.
[`fifa_ranking_2021-08-12_to_2021-11-19.csv`](./data/fifa_ranking_2021-08-12_to_2021-11-19.csv):  World FIFA rankings from 08/12/2021 to 11/19/2021 (to be combined with rankings from 05/27/2021 and prior data).  Obtained from https://www.fifa.com/fifa-world-ranking/.
[`current_table_12.14.2021.csv`](./data/current_table_12.14.2021.csv):  Current CONCACAF World Cup Qualifying table for comparison.  Obtained from https://www.fifa.com/tournaments/mens/worldcup/qatar2022/qualifiers/concacaf.
[`cleaned_results_and_rankings.csv`](./data/cleaned_results_and_rankings.csv):  International fixtures with rankings pulled in and engineered features of 10 game home/away goals for/against averages and dummied match types (used for modeling for predictions)
[`cleaned_qualifying_fixtures.csv`](./data/cleaned_qualifying_fixtures.csv):  CONCACAF qualifying fixtures with scores removed and rankings and features added to use model to predict on
[`qualifier_predictions.csv`](./data/qualifier_predictions.csv):  CONCACAF qualifying fixtures with predicted scores from the model


**Data dictionary for [`cleaned_results_and_rankings.csv`](./data/cleaned_results_and_rankings.csv) file:**

|Feature|Type|Description|
|---|---|---|
|date|string|date of match|
|home_team|string|team usually playing in their home country unless on neutral ground|
|away_team|string|team playing outside of their home country|
|home_score|integer|number of goals scored by the home team|
|away_score|integer|number of goals scored by the away team|
|neutral|integer|1 if played at a neutral site and 0 if played in the country of the home team|
|home_rank|integer|FIFA ranking for the home team at the time of the game|
|away_rank|integer|FIFA ranking for the away team at the time of the game|
|h_goals_for_avg|float|average of goals scored by the home team for their 10 previous home games|
|h_goals_against_avg|float|average of goals scored against the home team for their 10 previous home games|
|a_goals_for_avg|float|average of goals scored by the away team for their 10 previous away games|
|a_goals_against_avg|float|average of goals scored against the away team for their 10 previous away games|
|datetime|datetime|date column converted to datetime datatype|
|match_type_FIFA_WCQ|integer|1 if match is a FIFA World Cup Qualification Match and 0 if it is not|
|match_type_Friendly|integer|1 if match is a Friendly Match and 0 if it is not|
|match_type_Qualifier|integer|1 if match is a tournament or cup qualification match except for FIFA World Cup Qualifying and 0 if it is not|
|match_type_Tournament/Cup_Match|integer|1 if match is a tournament or cup match except for FIFA World Cup and 0 if it is not|

---

### Cleaning, Preprocessing, and EDA


My first step in organizing and cleaning the data was to remove all fixtures with teams that were not members of FIFA and therefore, would not have a ranking associated with that country.  In doing so, I found that there were several teams that were not actually countries as well.  There were also countries that did not officially join FIFA until after the first rankings came out in 1992 and did not have rankings for some games, so those matches were dropped as well.  I also dropped all games prior to the first FIFA rankings in 1992 since teams were not internationally ranked and after 09/01/2021 because those contained matches that I was looking to predict.  Lastly, I needed to ensure that the team names in the rankings and results tables match in order to pull the rankings for a team over to the results table.

Once those cleaning steps were completed, I used for loops to derive averages for goals scored and goals conceded by the home and away teams in their previous 10 respective home and away games.  I also wrote functions to apply the teams most current ranking to the results dataset from the rankings dataset.  I also converted the neutral columns True/False to 1/0 to prepare the data for modeling and condensed and one-hot encoded the match type.

Once the results dataset was cleaned with engineered features, I ensured that the qualifying fixtures dataset that I was going to predict on had the same features.  For rankings and 10 game goal averages, I provided the dataset with the most recent data from the results table in order to predict the entire set of matches.

Exploratory Data Analysis came next and with that came insights into what the model may predict.  Looking at the distribution of goals by home and away teams, I found that it is most common that the home team scores 1 goal and the away team scores 0 goals.  This occurred in about 30% and 40% of the data, respectively.  I also looked at goal averages for CONCACAF teams over the entire dataset and found that Mexico had the best goal differential in home and away games.

Finally, I looked at win percentages based on rankings and whether or not a team was the home team.  Higher ranked teams won 55% of the time compared to 21% for lower ranked teams and 24% draws.  The home team won 48% of the games played compared to 28% wins by the away team and 24% draws.  After a deeper understanding of the data, it was time to move to modeling.

---

### Modeling, Analysis, and Recommendations

Upon researching this topic, I found that not many people use a linear regression model or output to predict scores or outcomes of soccer matches.  In fact, most models took the classification root to predict either win, lose, or draw by a team.  Predicting scores let alone outcomes in such a low scoring sport where so many different factors can affect the score/outcome has proven to be difficult.

I decided to go with the Keras library and create a neural network models to help predict the home and away scores.  The models each were made up of 4 Dense layers with a linear regression output layer for deep learning of the data.  I used the cleaned results and rankings dataset that contained my engineered features to train both models.  

The models did not perform very well, but that was to be expected with the limited amount of data available for each match and the complexity of predicting the exact score.  The model predicting the home score had an r2 score of 0.18 and a mean absolute error of 1.0.  The model predicting the away score had an r2 score of 0.11 and a mean absolute error of 0.81.  While the r2 scores and errors of almost 1 goal for both home and away scores may not seem great, I moved forward with using the models to predict on the CONCACAF qualifying matches.

---

### Presentation

[CONCACAF World Cup Qualifying Match Predictions](https://docs.google.com/presentation/d/136yhEa5arWjzEvIrlBj98Zj64CgmEy_qG1A76P6mUpU/edit?usp=sharing)






Things to discuss in analysis/presentation:

soccer difficult to predict, only predicting 14 games

not a lot of people use regression to predict goals scored (most predict outcome/use classification)

recommend trying another model(random forest regressor), looking at weighting the rankings so they don't as heavily impact predicted scores, look to add more stats, possibly using a RNN model to predict home/away goals for/against averages in the upcoming fixtures