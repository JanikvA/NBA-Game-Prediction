import os
import time

import pandas as pd
from loguru import logger
from nba_api.stats.endpoints import leaguegamefinder

from nba_game_prediction.config import config


def combine_team_games(df):
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
        df, df, suffixes=["_HOME", "_AWAY"], on=["SEASON_ID", "GAME_ID", "GAME_DATE"]
    )
    # Filter out any row that is joined to itself.
    result = joined[joined.TEAM_ID_HOME != joined.TEAM_ID_AWAY]
    # Take action based on the keep_method flag.
    result = result[result.MATCHUP_HOME.str.contains(" vs. ")]
    return result


def main():
    all_games = pd.DataFrame()
    for season in config["collect_game_data"]["seasons"]:
        logger.info(f"Collecting game data for {season}")
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season, league_id_nullable="00"
        )
        games = gamefinder.get_data_frames()[0]
        all_games = pd.concat([all_games, games], ignore_index=True)
        time.sleep(1)
    all_games.to_csv(
        os.path.join(config["data_dir"], config["collect_game_data"]["raw_output_name"])
    )

    combined_games = combine_team_games(all_games)
    combined_games.to_csv(
        os.path.join(
            config["data_dir"], config["collect_game_data"]["combined_output_name"]
        )
    )


if __name__ == "__main__":
    main()
