# NBA game predictions

## Summary

**Situation**: The outcome of Basketball games can be influenced by multiple factors and it's accurate prediction is used by Odd-makers to create profit from betting customers

**Task**: Create a way of predicting the winner of a basketball game in the National Basketball Association (NBA)

**Action**:
  - Collected data relevant for the prediction from multiple sources for 26000 games played in the NBA (seasons: 2000-2021)
  - Explored the importance of different features by checking distributions, correlation and shapely values
  - Compared the accuracy of predictions for different models (Logistic Regression, XGBoost, K-Nearest Neighbors, Multilayer Perceptron, Random Forest)

**Result**:
  - Homecourt advantage plays a large role for the outcome of NBA games. **59.5% +- 0.3%** of games are won by the home town team
  - The team strength calculated by an Elo-like algorithm yields great predictive power. Teams with a higher Elo/Trueskill rating win 64.3% +- 0.3% of the games. The FTE-Elo calculation which combines homecourt advantage and an Elo algorithm has an accuracy of **66.3% +- 0.3%**
  - The different models have similar accuracies
  - Highest accuracy of predicting the correct winner is ~67%

## Introduction

Betting on the outcome of sports games has been a popular activity for a long time.
Professional Odd-makers can use data collected in the past to predict the probability of future events.
The more precise these predicted probabilities are the more profit can be generated from betting people.


## How are other people solving this problem?

## Methods

### Data sources

The data was collected from three different sources:
1. nba_api: python bindings for the NBA stats API
    - Team stats for each game played in the NBA
2. [FiveThirtyEight (FTE)](https://projects.fivethirtyeight.com/nba-model/nba_elo_latest.csv):
    - FTEs own Elo calculation described in [this article](https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/)
3. payroll data for the NBA teams scraped from [Hoopshype.com](https://hoopshype.com/salaries/)

### Features

List of Features investigated:
  - is_back_to_back: True if the team has played a game on the day before
  - fraction_total_payroll: payroll(team)/Sum of all payrolls
  - ELO: standard Elo value
  - ELO_winprob: win probability calculated from the Elo of both teams
  - trueskill_mu: Elo equivalent for Trueskill
  - trueskill_winprob: Approximate win probability from the Trueskill of both teams
  - FTE_ELO: Elo calculated by FiveThirtyEight (FTE)
  - FTE_ELO_winprob: win probability provided by FTE and based on FTE_ELO
  - payroll_oppo_ratio: payroll(team)/payroll(opponent)
  - won_last_game: True if the team has won its last game

The following features are averaged over the last 20 games (excluding the game trying to predict) (postfix: `_20G`):
  - WL: Win/Loss record
  - ELO_mean: mean of ELO
  - ELO_mean_change: mean ELO change
  - trueskill_mu_mean: mean of trueskill_mu
  - trueskill_mu_mean_change: mean trueskill_mu change
  - FTE_ELO_mean: mean FTE_ELO
  - FTE_ELO_mean_change: mean FTE_ELO change
  - PTS1_frac: fraction of total points scored coming from free throws
  - FT_PCT: Free throw %
  - PTS2_frac: fraction of total points scored coming from two-pointers
  - FG2_PCT: two-pointers %
  - PTS3_frac: fraction of total points scored coming from three-pointers
  - FG3_PCT: three-pointers %
  - PTS_oppo_ratio: pts(team) / pts(opponent)
  - FGM_AST_frac: AST/FGM - fraction of field goals assited on

All features are avialable for the home (prefix: `HOME_`) and away (prefix: `AWAY_`) team

Other features:
  - is_Playoffs: True if the game is a Playoff game
  - random_winprob: random number uniformly distributed between 0 and 1. Used for validation purposes


<img src="output/feature_correlation_pearson.png" alt="correlation matrix" width="600"/>
<img src="output/target_feature_correlation_kendall.png" alt="correlation matrix" width="400"/>

#### Elo algorithms

- [Elo](https://en.wikipedia.org/wiki/Elo_rating_system): an algorithm most famously used in Chess. All teams start with a rating of 1400 and the maximum possible adjustment per game, called the K-factor, is set to 20
- [FiveThirtyEight (FTE) Elo](https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/): An adapted version of the Elo algorithm. This algorith also takes into account the home court advantage by adding 100 ranking points to the home team for the calculation of the win probability.
- [Trueskill](https://trueskill.org/): Rating system developed by Microsoft for the goal of high quality match making in online games.

### Models

- Logistic Regression
- XGBoost
- KNN
- MLP
- Random Forrest


### Preprocessing

- From the 26000 games that were collected the first 1000 games are removed from the training in order to give the Elo and Trueskill algorithm times to settle in.
Of the remaining 25000 games, 20000 are randomly collected for the training and 5000 for the test data set.
- During training the trainings data set is split using a [10-fold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold) method for cross-validation purposes.
- The input features are transformed using the sklearn.preprocessing.StandardScalar

## Results


<!-- <img src="output/team_FTE_ELO_plot.png" alt="team FTE Elo" width="600"/> -->

<img src="output/probability_comparison_HOME_FTE_ELO_winprob.png" alt="FTE winprob closure" width="600"/>

<img src="output/probability_comparison_LinearRegression.png" alt="FTE winprob closure" width="600"/>

<img src="output/acc_per_season_HOME_FTE_ELO_winprob.png" alt="acc per season FTE" width="600"/>

<img src="output/acc_per_season_LinearRegression.png" alt="acc per season FTE" width="600"/>

<img src="output/all_features/xgboost_shap_bar_summary.png" alt="acc per season FTE" width="600"/>



Method | Accuracy
---------|----------
Home team wins | 59.5 +- 0.3
Trueskill | 64 +- 0.3
Elo | 64.2 +- 0.3
FTE Elo | 66.3 +- 0.3


Model Accuracies:
Algorithm | Test | Validation | Train
---------|---------|----------|---------
Logistic Regression | 67 | 66.5 +- 1 | 66.6
XGBoost | 66.6 | 66.2 +- 0.9 | 66.8
Random Forrest | 64.6 | 63.7 +- 0.8 | 64.8
MLPClassifier | 66.9 | 66.4 +- 0.9 | 66.7
KNN | 64.7 | 63.6 +- 1.1 | 66.1


## Conclusion

**Key Insights**:


## Next steps

- Include injury data for predictions
- Compare to accuracy of predictions from Odd-makers
- Check differences in predicted probabilities compared to Odd-makers

---------


<details>
  <summary>Getting Started</summary>

### Dependencies

- python 3.10
- python poetry

### Setup

```bash
poetry install
pre-commit install
```

### Docker

A Docker image is also available:

```bash
docker pull janikvapp/nba-game-prediction:latest
```

### Scripts

Scripts should be run in the following order:

```bash
poetry shell
python nba_game_prediction/scripts/collect_game_data.py
python nba_game_prediction/scripts/create_train_data.py
python nba_game_prediction/scripts/plot_train_data.py
python nba_game_prediction/scripts/train_model.py
```

Running plot_train_data.py is optional
Most Configuration can be done using the config file in `data/config.yaml`

### Testing

Only running unit tests:

```bash
make tests
```

Running integration tests (this is slow):

```bash
make integration
```

</details>
