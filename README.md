# NBA game predictions

## Summary

**Situation**: The outcome of Basketball games can be influenced by multiple factors and it's accurate prediction is used by Odd-makers to create profit from betting customers

**Task**: Create a way of predicting the winner of a basketball game in the National Basketball Association (NBA)

**Action**:
  - Collected data relevant for the prediction from multiple sources for 26000 games played in the NBA (seasons: 2000-2021)
  - Explored the importance of different features by checking distributions, correlation and shapely values
  - Compared the accuracy of predictions for different models using scikit-learn and XGBoost

**Result**:
  - The most important feature for the predictions is the win probability given by Elo based calculations
  - The type of model had no big difference on the accuracy of the predictions
  - Highest accuracy of predicting the correct winner is 67%

## Introduction

## How are other people solving this problem?

## Methods

### Data sources

nba_api
FiveThirtyEight
    - https://datahub.io/five-thirty-eight/nba-elo#readme
    - https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/
    - https://projects.fivethirtyeight.com/nba-model/nba_elo_latest.csv
payroll salary

### Features

List of Features investigated:
  - is_back_to_back
  - fraction_total_payroll
  - ELO
  - ELO_winprob
  - trueskill_mu
  - trueskill_winprob
  - FTE_ELO
  - FTE_ELO_winprob
  - payroll_oppo_ratio
  - won_last_game

The following features are averaged over the last 20 games (excluding the game trying to predict) (postfix: `_20G`):
  - WL
  - ELO_mean
  - ELO_mean_change
  - trueskill_mu_mean
  - trueskill_mu_mean_change
  - FTE_ELO_mean
  - FTE_ELO_mean_change
  - PTS1_frac
  - FT_PCT
  - PTS2_frac
  - FG2_PCT
  - PTS3_frac
  - FG3_PCT
  - PTS_oppo_ratio
  - FGM_AST_frac

All features are avialable for the home (prefix: `HOME_`) and away (prefix: `AWAY_`) team

Other features:
  - is_Playoffs
  - random_winprob


<img src="output/feature_correlation_pearson.png" alt="correlation matrix" width="600"/>
<img src="output/target_feature_correlation_kendall.png" alt="correlation matrix" width="400"/>

#### Elo algorithms

- Chess Elo
- FiveThirtyEight (FTE) Elo
- Microsoft Trueskill

### Models

## Results

20000 for training 5000 for testing, droped first 1000



<!-- <img src="output/team_FTE_ELO_plot.png" alt="team FTE Elo" width="600"/> -->

<img src="output/probability_comparison_HOME_FTE_ELO_winprob.png" alt="FTE winprob closure" width="600"/>

<img src="output/probability_comparison_LinearRegression.png" alt="FTE winprob closure" width="600"/>

<img src="output/acc_per_season_HOME_FTE_ELO_winprob.png" alt="acc per season FTE" width="600"/>

<img src="output/acc_per_season_LinearRegression.png" alt="acc per season FTE" width="600"/>

<!-- <img src="output/xgboost_shap_bar_summary.png" alt="acc per season FTE" width="600"/> -->



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
