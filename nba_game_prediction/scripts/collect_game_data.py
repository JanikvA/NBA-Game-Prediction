import sqlite3
import time

import pandas as pd
from loguru import logger
from nba_api.stats.endpoints import leaguegamefinder

from nba_game_prediction import config_modul


def combine_team_games(df: pd.DataFrame) -> pd.DataFrame:
    """Combine a TEAM_ID-GAME_ID unique table into rows by game. Slow.
    From: https://github.com/swar/nba_api/blob/master/docs/examples/Finding%20Games.ipynb

    Parameters
    ----------
    df : Input DataFrame.

    Returns
    -------
    result : DataFrame
    """
    # Join every row to all others with the same game ID.
    joined = pd.merge(
        df,
        df,
        suffixes=["_HOME", "_AWAY"],
        on=["SEASON_ID", "GAME_ID", "GAME_DATE", "SEASON_TYPE", "SEASON"],
    )
    # Filter out any row that is joined to itself.
    result = joined[joined.TEAM_ID_HOME != joined.TEAM_ID_AWAY]
    # Take action based on the keep_method flag.
    result = result[result.MATCHUP_HOME.str.contains(" vs. ")]
    return result


def main(config: dict) -> None:
    """Collect data for NBA games using the nba_api package

    Args:
        config (dict): config
    """
    all_games = pd.DataFrame()
    for season in config["collect_game_data"]["seasons"]:
        logger.info(f"Collecting game data for {season}")
        for season_type in config["collect_game_data"]["season_types"]:
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable="00",
                season_type_nullable=season_type,
                timeout=300,
            )
            games = gamefinder.get_data_frames()[0]
            games["SEASON_TYPE"] = season_type
            games["SEASON"] = season[: season.find("-")]
            all_games = pd.concat([all_games, games], ignore_index=True)
            time.sleep(1)

    combined_games = combine_team_games(all_games)

    logger.info(f"Saving {len(all_games)} collected games to {config['sql_db_path']}")
    sql_connection = sqlite3.connect(config["sql_db_path"])
    all_games.to_sql(
        "NBA_games_per_team", sql_connection, if_exists="replace", index=False
    )
    combined_games.to_sql("NBA_games", sql_connection, if_exists="replace", index=False)
    sql_connection.close()


if __name__ == "__main__":
    main(config_modul.load_config(config_modul.get_comandline_arguments()["config"]))
