# NBA game predictions

## Introduction

## How are other people solving this problem?

## Methods

### Data sources
FiveThirtyEight
    - https://datahub.io/five-thirty-eight/nba-elo#readme
    - https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/
    - https://projects.fivethirtyeight.com/nba-model/nba_elo_latest.csv
nba_api
payroll salary

### Features

### Models

## Results

## Conclusion

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
