## Introduction

## Data sources
 FiveThirtyEight
    - https://datahub.io/five-thirty-eight/nba-elo#readme
    - https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/
    - https://projects.fivethirtyeight.com/nba-model/nba_elo_latest.csv

## Dependencies

- python 3.10
- python poetry

## Setup

```bash
poetry install
poetry shell
pre-commit install
```

## Publishing to DockerHub

Needs github-cli (e.g. from AUR)

```bash
gh workflow run publish-docker
```

## Getting Started
