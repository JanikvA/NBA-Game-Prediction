# NBA game predictions

## Summary

**Situation**: The outcome of Basketball games can be influenced by multiple factors and it's accurate prediction is used by Odd-makers to create profit from betting customers
**Task**: Create a way of predicting the winner of a basketball game in the National Basketball Association (NBA)
**Action**:
  - Collected data relevant to the prediction of NBA games from multiple sources for 26000 games played (seasons: 2000-2021)
  - Explored the importance of different features using distributions, correlation and shapely values
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

All features are avialable for the home (prefix: `_HOME`) and away (prefix: `AWAY_`) team

Other features:
  - is_Playoffs
  - random_winprob

### Models

## Results

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

### Getting Started

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
