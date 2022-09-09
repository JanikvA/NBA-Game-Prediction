import datetime
import sqlite3
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import trueskill
from loguru import logger
from rich.progress import Progress

from nba_game_prediction import config_modul, elo_modul


class NBATeam:
    """Class to handle NBA teams and the data associated to them"""

    nba_teams: Dict[str, Any] = {}
    home_away: List[str] = ["HOME", "AWAY"]
    team_stats: List[str] = [
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
    ]

    def __init__(self, name: str) -> None:
        """Creates a NBATeam object and initalizes important attributes.
        The object will also be added to the NBATeam.nba_teams dict

        Args:
            name (str): Name of the NBA team

        Raises:
            Exception: If a team with the same name is already in the NBATeam.nba_teams dict.
        """
        self.name = name
        self.games = pd.DataFrame()
        self.elo: float = 1400.0
        self.trueskill = trueskill.Rating()
        if self.name in NBATeam.nba_teams:
            raise Exception(f"{self.name} is already in NBATeam.nba_teams!")
        else:
            NBATeam.nba_teams[self.name] = self

    def get_stats_between_dates(
        self, from_date: datetime.datetime, to_date: datetime.datetime
    ) -> Dict[str, Any]:
        """Calculates the mean for numeric values and all games played between two dates

        Args:
            from_date (datetime.datetime): start date
            to_date (datetime.datetime): end date

        Returns:
            Dict[str, Any]: key: stat name - value: mean for numeric values
        """
        selected_games = self.games[
            (self.games.index < to_date) & (self.games.index > from_date)
        ]
        data = selected_games.mean(numeric_only=True).to_dict()
        for k in list(data.keys()):
            data[k + "_mean"] = data.pop(k)
        return data

    def get_stats_for_date(
        self, date: datetime.datetime, games_back: int = 10
    ) -> Dict[str, Any]:
        """Get the team data for a specific date

        Args:
            date (datetime.datetime): which date to look at
            games_back (int, optional): For some stats the mean over
            {games_back} is calculated. Defaults to 10.

        Returns:
            Dict[str, Any]: key: stat name - value: value
        """
        last_ten_games = self.get_last_N_games(date, games_back)
        data = last_ten_games.mean(numeric_only=True).to_dict()
        for k in list(data.keys()):
            data[k + f"_{games_back}G"] = data.pop(k)
        if date not in self.games.index:
            logger.warning(f"No Game was played on the given date ({date})!")
        else:
            day_before = date - datetime.timedelta(days=1)
            data["is_back_to_back"] = int(day_before in self.games)
            for key in [
                "ELO",
                "ELO_winprob",
                "trueskill_mu",
                "trueskill_sigma",
                "trueskill_winprob",
            ]:
                data[key] = self.games.loc[date][key]
        return data

    def get_last_N_games(
        self, date: datetime.datetime, n_games: int = 10
    ) -> pd.DataFrame:
        """Get data of last games played

        Args:
            date (datetime.datetime): only games before this date are considered
            n_games (int, optional): Get the last {n_games} played before {date}. Defaults to 10.

        Returns:
            pd.DataFrame: games fulfilling the requirements
        """
        games_before = self.games[self.games.index < date]
        return games_before.head(n_games)

    @classmethod
    def get_opposite_home_away(cls, home_away: str) -> str:
        """Turns 'HOME' into 'AWAY' and 'AWAY' into 'HOME'

        Args:
            home_away (str): either 'HOME' or 'AWAY'

        Raises:
            Exception: if {home_away} is not in ['HOME', 'AWAY']

        Returns:
            str: Returns the opposite of {home_away}
        """
        if home_away not in ["HOME", "AWAY"]:
            raise Exception(f"{home_away} has to be either 'HOME' or 'AWAY'")
        return "HOME" if home_away == "AWAY" else "AWAY"


# TODO optimize
def extract_game_data(game_data: pd.DataFrame) -> None:
    """Extracts relevant data from the game and adds the
    information to the data of both the home and away team

    Args:
        game_data (pd.DataFrame): Data of the game to be analyzed
    """
    team_data_dic: Dict[str, Dict] = {ha: {} for ha in NBATeam.home_away}
    team_obj_dic: Dict[str, NBATeam] = {
        ha: NBATeam.nba_teams[game_data[f"TEAM_NAME_{ha}"]] for ha in NBATeam.home_away
    }

    team_data_dic["HOME"]["HOME_GAME"] = 1
    team_data_dic["AWAY"]["HOME_GAME"] = 0
    team_data_dic["HOME"]["TEAM_NAME_opponent"] = game_data["TEAM_NAME_AWAY"]
    team_data_dic["AWAY"]["TEAM_NAME_opponent"] = game_data["TEAM_NAME_HOME"]

    for ha in NBATeam.home_away:
        opposite_ha = NBATeam.get_opposite_home_away(ha)
        team_data_dic[ha]["GAME_DATE"] = game_data.name
        for uniq in ["SEASON_ID", "GAME_ID", "SEASON_TYPE", "SEASON"]:
            team_data_dic[ha][uniq] = game_data[uniq]
        for team_stat in NBATeam.team_stats:
            team_data_dic[ha][team_stat] = game_data[team_stat + "_" + ha]
        team_data_dic[ha]["WL"] = int(game_data[f"WL_{ha}"])
        team_data_dic[ha]["TEAM_NAME_opponent"] = game_data[f"TEAM_NAME_{opposite_ha}"]
        team_data_dic[ha]["ELO"] = team_obj_dic[ha].elo
        team_data_dic[ha]["ELO_opponent"] = team_obj_dic[opposite_ha].elo
        team_data_dic[ha]["ELO_winprob"] = elo_modul.ELO.expected_Ea(
            team_obj_dic[ha].elo, team_obj_dic[opposite_ha].elo
        )
        team_data_dic[ha]["trueskill_mu"] = team_obj_dic[ha].trueskill.mu
        team_data_dic[ha]["trueskill_sigma"] = team_obj_dic[ha].trueskill.sigma
        team_data_dic[ha]["trueskill_mu_opponent"] = team_obj_dic[
            opposite_ha
        ].trueskill.mu
        team_data_dic[ha]["trueskill_sigma_opponent"] = team_obj_dic[
            opposite_ha
        ].trueskill.sigma
        team_data_dic[ha]["trueskill_winprob"] = elo_modul.trueskill_win_probability(
            team_obj_dic[ha].trueskill, team_obj_dic[opposite_ha].trueskill
        )
        formatted_data = pd.DataFrame.from_records(team_data_dic[ha], index=[0])
        team_obj_dic[ha].games = pd.concat(
            [team_obj_dic[ha].games, formatted_data], ignore_index=True
        )

    # update ratings
    if int(game_data["WL_HOME"]) == 1:
        (
            team_obj_dic["HOME"].trueskill,
            team_obj_dic["AWAY"].trueskill,
        ) = trueskill.rate_1vs1(
            team_obj_dic["HOME"].trueskill, team_obj_dic["AWAY"].trueskill
        )
        team_obj_dic["HOME"].elo, team_obj_dic["AWAY"].elo = elo_modul.ELO.rate_1vs1(
            team_obj_dic["HOME"].elo, team_obj_dic["AWAY"].elo
        )
    else:
        (
            team_obj_dic["AWAY"].trueskill,
            team_obj_dic["HOME"].trueskill,
        ) = trueskill.rate_1vs1(
            team_obj_dic["AWAY"].trueskill, team_obj_dic["HOME"].trueskill
        )
        team_obj_dic["AWAY"].elo, team_obj_dic["HOME"].elo = elo_modul.ELO.rate_1vs1(
            team_obj_dic["AWAY"].elo, team_obj_dic["HOME"].elo
        )


def fill_team_game_data(games: pd.DataFrame) -> None:
    """Adds the statistics from all games
    provided to their respective teams

    Args:
        games (pd.DataFrame): Games to be analyzed
    """
    logger.info(f"Loading {len(games)} games for {len(NBATeam.nba_teams)} teams")
    with Progress() as progress:
        task = progress.add_task("[green]Loading game data...", total=len(games))
        for index, row in games.iterrows():
            extract_game_data(row)
            progress.update(task, advance=1)
    for team_name, team in NBATeam.nba_teams.items():
        team.games = team.games.set_index("GAME_DATE")


def get_train_data_from_game(
    game: pd.DataFrame, feature_list: List[str]
) -> Dict[str, Any]:
    """Get the features relevant for model training from a game

    Args:
        game (pd.DataFrame): data of the game
        feature_list (List[str]): list of important features specified in the config file

    Returns:
        Dict[str, Any]: key: feature - value: value
    """
    tmp_dic = {}
    train_data_dict = {}
    for ha in NBATeam.home_away:
        team = NBATeam.nba_teams[game[f"TEAM_NAME_{ha}"]]
        train_data_dict[f"{ha}_TEAM_NAME"] = game[f"TEAM_NAME_{ha}"]
        train_data_dict[f"{ha}_WL"] = int(game[f"WL_{ha}"])
        tmp_dic[ha] = team.get_stats_for_date(game.name)
        for feature in feature_list:
            train_data_dict[ha + "_" + feature] = tmp_dic[ha][feature]
    for simple_int_feature in ["SEASON_ID", "SEASON", "GAME_ID"]:
        train_data_dict[simple_int_feature] = int(game[simple_int_feature])
    train_data_dict["is_Playoffs"] = int(game["SEASON_TYPE"] == "Playoffs")
    train_data_dict["GAME_DATE"] = game.name
    return train_data_dict


def create_train_data(
    games: pd.DataFrame, sql_db_connection: sqlite3.Connection, feature_list: List[str]
) -> None:
    """Creates and saves the data used for training the models

    Args:
        games (pd.DataFrame): All games that should be used
        sql_db_connection (sqlite3.Connection): Connection to
        the SQL database to which the data should be saved
        feature_list (List[str]): important features defined in the config file
    """
    train_data = pd.DataFrame()
    logger.info(f"Transforming data into trainings data for {len(games)} games.")
    with Progress() as progress:
        task = progress.add_task("[green]Transforming game data...", total=len(games))
        for index, row in games.iterrows():
            tmp = get_train_data_from_game(row, feature_list)
            game_train_data = pd.DataFrame(tmp, index=[0])
            train_data = pd.concat([train_data, game_train_data], ignore_index=True)
            progress.update(task, advance=1)
    train_data.to_sql("train_data", sql_db_connection, if_exists="replace", index=False)


def get_game_data(sql_db_connection: sqlite3.Connection) -> pd.DataFrame:
    """Extract the game data from the SQL database created by the collect_game_data.py

    Args:
        sql_db_connection (sqlite3.Connection): Connection to the SQL database

    Returns:
        pd.DataFrame: Contains data for all games
    """
    games = pd.read_sql("SELECT * from NBA_games", sql_db_connection)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games = games.set_index("GAME_DATE")
    games = games.sort_index()
    games["WL_HOME"] = games.apply(
        lambda row: 1 if row["WL_HOME"] == "W" else 0, axis=1
    )
    games["WL_AWAY"] = games.apply(
        lambda row: 1 if row["WL_AWAY"] == "W" else 0, axis=1
    )
    return games


def main(config: Dict[str, Any]) -> None:
    """Creates the data used in the training and saves it to the SQL data base

    Args:
        config (Dict[str, Any]): config
    """
    connection = sqlite3.connect(config["sql_db_path"])
    trueskill.setup(mu=30, draw_probability=0)
    games = get_game_data(connection)
    # initialize teams:
    for team_name in set(
        np.concatenate(
            (games["TEAM_NAME_HOME"].unique(), games["TEAM_NAME_AWAY"].unique())
        )
    ):
        NBATeam(team_name)
    fill_team_game_data(games)
    create_train_data(games, connection, config["create_train_data"]["feature_list"])
    connection.close()


if __name__ == "__main__":
    main(config_modul.load_config(config_modul.get_comandline_arguments()["config"]))
