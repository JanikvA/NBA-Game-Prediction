import pandas as pd
from loguru import logger

from nba_game_prediction.scripts.collect_game_data import (
    combine_team_games,
    new_orleans_name,
)


def test_new_orleans_name():
    io_dict = {
        2005: "New Orleans/Oklahoma City Hornets",
        2008: "New Orleans Hornets",
        2014: "New Orleans Pelicans",
    }
    for input, exp_output in io_dict.items():
        output = new_orleans_name(input)
        assert output == exp_output


def test_combine_team_games():
    input_cols = [
        "SEASON_ID",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "TEAM_NAME",
        "GAME_ID",
        "GAME_DATE",
        "MATCHUP",
        "WL",
        "MIN",
        "PTS",
        "FGM",
        "FGA",
        "FG_PCT",
        "FG3M",
        "FG3A",
        "FG3_PCT",
        "FTM",
        "FTA",
        "FT_PCT",
        "OREB",
        "DREB",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "PLUS_MINUS",
        "SEASON_TYPE",
        "SEASON",
    ]
    df = pd.DataFrame(columns=input_cols)
    df = combine_team_games(df)
    exp_output_cols = set(
        [
            "SEASON_ID",
            "TEAM_ID_HOME",
            "TEAM_ABBREVIATION_HOME",
            "TEAM_NAME_HOME",
            "GAME_ID",
            "GAME_DATE",
            "MATCHUP_HOME",
            "WL_HOME",
            "MIN_HOME",
            "PTS_HOME",
            "FGM_HOME",
            "FGA_HOME",
            "FG_PCT_HOME",
            "FG3M_HOME",
            "FG3A_HOME",
            "FG3_PCT_HOME",
            "FTM_HOME",
            "FTA_HOME",
            "FT_PCT_HOME",
            "OREB_HOME",
            "DREB_HOME",
            "REB_HOME",
            "AST_HOME",
            "STL_HOME",
            "BLK_HOME",
            "TOV_HOME",
            "PF_HOME",
            "PLUS_MINUS_HOME",
            "SEASON_TYPE",
            "SEASON",
            "TEAM_ID_AWAY",
            "TEAM_ABBREVIATION_AWAY",
            "TEAM_NAME_AWAY",
            "MATCHUP_AWAY",
            "WL_AWAY",
            "MIN_AWAY",
            "PTS_AWAY",
            "FGM_AWAY",
            "FGA_AWAY",
            "FG_PCT_AWAY",
            "FG3M_AWAY",
            "FG3A_AWAY",
            "FG3_PCT_AWAY",
            "FTM_AWAY",
            "FTA_AWAY",
            "FT_PCT_AWAY",
            "OREB_AWAY",
            "DREB_AWAY",
            "REB_AWAY",
            "AST_AWAY",
            "STL_AWAY",
            "BLK_AWAY",
            "TOV_AWAY",
            "PF_AWAY",
            "PLUS_MINUS_AWAY",
        ]
    )
    differences = set(df.columns.values) ^ exp_output_cols
    logger.warning(f"{differences=}")
    assert not differences
