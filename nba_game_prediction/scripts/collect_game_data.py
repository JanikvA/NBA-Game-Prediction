import sqlite3
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger
from nba_api.stats.endpoints import leaguegamefinder

from nba_game_prediction import config_modul


def new_orleans_name(season: int):
    if season > 2012:
        return "New Orleans Pelicans"
    elif season in [2005, 2006]:
        return "New Orleans/Oklahoma City Hornets"
    else:
        return "New Orleans Hornets"


def scrape_team_payroll(url: str, season: int) -> pd.DataFrame:
    response = requests.get(url)
    r_html = response.text
    soup = BeautifulSoup(r_html, "html.parser")
    payroll_table = soup.find("table")
    length = len(payroll_table.find_all("td"))
    entries_per_row = 4
    team_names = [
        payroll_table.find_all("td")[i].text.strip()
        for i in range(5, length, entries_per_row)
    ]
    if len(team_names) == 0:
        logger.error(
            f"No team names were extracted for the {season} season payroll data!"
        )
        return
    payroll = [
        float(payroll_table.find_all("td")[i].text.strip().strip("$").replace(",", ""))
        for i in range(6, length, entries_per_row)
    ]
    inflation_adjusted_payroll = [
        float(payroll_table.find_all("td")[i].text.strip().strip("$").replace(",", ""))
        for i in range(7, length, entries_per_row)
    ]

    fraction_total_payroll = []
    total_payroll = sum(payroll)
    for pr in payroll:
        fraction_total_payroll.append(len(team_names) * pr / total_payroll)

    team_name_dic = {
        "Orlando": "Orlando Magic",
        "San Antonio": "San Antonio Spurs",
        "Toronto": "Toronto Raptors",
        "Brooklyn": "New Jersey Nets" if season < 2012 else "Brooklyn Nets",
        "Milwaukee": "Milwaukee Bucks",
        "New York": "New York Knicks",
        "LA Clippers": "Los Angeles Clippers" if season < 2015 else "LA Clippers",
        "Atlanta": "Atlanta Hawks",
        "Chicago": "Chicago Bulls",
        "New Orleans": new_orleans_name(season),
        "Oklahoma City": "Seattle SuperSonics"
        if season < 2008
        else "Oklahoma City Thunder",
        "Washington": "Washington Wizards",
        "Detroit": "Detroit Pistons",
        "Utah": "Utah Jazz",
        "Houston": "Houston Rockets",
        "Golden State": "Golden State Warriors",
        "Memphis": "Vancouver Grizzlies" if season < 2001 else "Memphis Grizzlies",
        "Cleveland": "Cleveland Cavaliers",
        "Charlotte": "Charlotte Bobcats"
        if season in range(2004, 2014)
        else "Charlotte Hornets",
        "Sacramento": "Sacramento Kings",
        "Denver": "Denver Nuggets",
        "Indiana": "Indiana Pacers",
        "Portland": "Portland Trail Blazers",
        "Phoenix": "Phoenix Suns",
        "LA Lakers": "Los Angeles Lakers",
        "Dallas": "Dallas Mavericks",
        "Minnesota": "Minnesota Timberwolves",
        "Miami": "Miami Heat",
        "Philadelphia": "Philadelphia 76ers",
        "Boston": "Boston Celtics",
        "Seattle": "Seattle SuperSonics",
    }

    for n, name in enumerate(team_names):
        if name not in team_name_dic:
            logger.warning(
                f"{name} is not in team_name_dic. The payroll data will not be used"
            )
        team_names[n] = team_name_dic[name]

    df_dict = {
        "team_name": team_names,
        "payroll": payroll,
        "inflation_adjusted_payroll": inflation_adjusted_payroll,
        "fraction_total_payroll": fraction_total_payroll,
        "season": season,
    }

    payroll_df = pd.DataFrame(df_dict)
    return payroll_df


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
    all_team_payrolls = pd.DataFrame()
    for season in config["collect_game_data"]["seasons"]:
        logger.info(f"Collecting game data and team payroll for {season}")
        team_payrolls = scrape_team_payroll(
            f"https://hoopshype.com/salaries/{season.replace('-','-20')}/",
            int(season[: season.find("-")]),
        )
        all_team_payrolls = pd.concat(
            (all_team_payrolls, team_payrolls), ignore_index=True
        )

        for season_type in config["collect_game_data"]["season_types"]:
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable="00",
                season_type_nullable=season_type,
            )
            games = gamefinder.get_data_frames()[0]
            games["SEASON_TYPE"] = season_type
            games["SEASON"] = season[: season.find("-")]
            if not all_games.empty:
                teams_in_all_games = all_games["TEAM_NAME"].unique()
                teams_in_games = games["TEAM_NAME"].unique()
                teams_not_in_all_games = set(teams_in_games) - set(teams_in_all_games)
                if teams_not_in_all_games:
                    logger.warning(f"These teams are new! {teams_not_in_all_games}")
            all_games = pd.concat([all_games, games], ignore_index=True)

            time.sleep(1)

    logger.info(
        f"The scraped games have the following columns: {all_games.columns.values}"
    )
    combined_games = combine_team_games(all_games)
    time.sleep(1)

    logger.info(f"Saving {len(all_games)} collected games to {config['sql_db_path']}")
    sql_connection = sqlite3.connect(config["sql_db_path"])
    logger.info(
        f"The table NBA_games_per_team has the following columns:\n{all_games.columns.values}"
    )
    all_games.to_sql(
        "NBA_games_per_team", sql_connection, if_exists="replace", index=False
    )
    logger.info(
        f"The table NBA_games has the following columns:\n{combined_games.columns.values}"
    )
    combined_games.to_sql("NBA_games", sql_connection, if_exists="replace", index=False)
    logger.info(
        f"The table team_payroll has the following columns:\n{all_team_payrolls.columns.values}"
    )
    all_team_payrolls.to_sql(
        "team_payroll", sql_connection, if_exists="replace", index=False
    )
    sql_connection.close()


if __name__ == "__main__":
    main(config_modul.load_config(config_modul.get_comandline_arguments()["config"]))
