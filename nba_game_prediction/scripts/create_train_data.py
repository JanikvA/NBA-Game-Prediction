import datetime
import os
import sqlite3
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
import trueskill
from loguru import logger
from rich.progress import Progress

from nba_game_prediction import config_modul, elo_modul


class NBATeam:
    """Class to handle NBA teams and the data associated to them"""

    nba_teams_abbreviations: Dict[str, str] = {}
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

    def add_payroll_data(self, payroll_data: pd.DataFrame) -> None:
        """Adds payroll data to the self.games data

        Args:
            payroll_data (pd.DataFrame): payroll data collected from hoopshype.com
        """

        for attr in ["payroll", "inflation_adjusted_payroll", "fraction_total_payroll"]:
            self.games[attr] = self.games.apply(
                lambda row: payroll_data[
                    (payroll_data["team_name"] == self.name)
                    & (payroll_data["season"] == row["SEASON"])
                ][attr].iloc[0],
                axis=1,
            )

        self.games["payroll_oppo_ratio"] = self.games["payroll"] / self.games.apply(
            lambda row: payroll_data[
                (payroll_data["team_name"] == row["TEAM_NAME_opponent"])
                & (payroll_data["season"] == row["SEASON"])
            ]["payroll"].iloc[0],
            axis=1,
        )

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

    def get_mean_features(self, date, games_back):
        last_N_games = self.get_last_N_games(date, games_back).copy()
        last_N_games["FG2M"] = pd.to_numeric(
            (
                (last_N_games["PTS"] - 3 * last_N_games["FG3M"] - last_N_games["FTM"])
                / 2
            ),
            downcast="integer",
        )
        last_N_games["FG2A"] = last_N_games["FGA"] - last_N_games["FG3A"]
        last_N_games["FG2_PCT"] = last_N_games["FG2M"] / last_N_games["FG2A"]

        last_N_games["PTS1_frac"] = last_N_games["FTM"] / last_N_games["PTS"]
        last_N_games["PTS2_frac"] = last_N_games["FG2M"] * 2 / last_N_games["PTS"]
        last_N_games["PTS3_frac"] = last_N_games["FG3M"] * 3 / last_N_games["PTS"]
        data = (
            last_N_games[
                [
                    "WL",
                    "FT_PCT",
                    "FG2_PCT",
                    "FG3_PCT",
                    "PTS1_frac",
                    "PTS2_frac",
                    "PTS3_frac",
                    "PTS_oppo_ratio",
                ]
            ]
            .mean()
            .to_dict()
        )
        for k in list(data.keys()):
            data[k + f"_{games_back}G"] = data.pop(k)
        for algo in ["ELO", "FTE_ELO", "trueskill_mu"]:
            if len(last_N_games) > 0:
                # FIXME not quite correct. not including the elo of the current atm but it should
                data[f"{algo}_mean_change_{games_back}G"] = (
                    last_N_games.iloc[-1][algo] - last_N_games.iloc[0][algo]
                ) / len(last_N_games)
                data[f"{algo}_mean_{games_back}G"] = last_N_games[algo].mean()
            else:
                data[f"{algo}_mean_change_{games_back}G"] = np.nan
                data[f"{algo}_mean_{games_back}G"] = np.nan
        return data

    def get_features(self, date):
        data = {}
        previous_game = self.get_last_N_games(date, 1)
        if len(previous_game) > 0:
            data["won_last_game"] = previous_game["WL"].iloc[-1]
        else:
            data["won_last_game"] = np.nan
        if date not in self.games.index:
            logger.warning(f"No Game was played on the given date ({date})!")
        else:
            day_before = date - datetime.timedelta(days=1)
            data["is_back_to_back"] = int(day_before in previous_game)
            for key in [
                "ELO",
                "ELO_winprob",
                "trueskill_mu",
                "trueskill_sigma",
                "trueskill_winprob",
                "fraction_total_payroll",
                "payroll_oppo_ratio",
                "FTE_ELO",
                "FTE_ELO_winprob",
            ]:
                data[key] = self.games.loc[date][key]
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
        data = self.get_mean_features(date, games_back)
        data = {**data, **self.get_features(date)}
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
    def reset_trueskill_sigma(cls, new_sigma=4):
        for name, team in cls.nba_teams.items():
            team.trueskill = trueskill.Rating(team.trueskill.mu, new_sigma)

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
def extract_game_data(
    game_data: pd.DataFrame, FTE_data: pd.DataFrame = pd.DataFrame()
) -> None:
    """Extracts relevant data from the game and adds the
    information to the data of both the home and away team

    Args:
        game_data (pd.DataFrame): Data of the game to be analyzed
    """
    team_data_dic: Dict[str, Dict] = {ha: {} for ha in NBATeam.home_away}
    team_obj_dic: Dict[str, NBATeam] = {
        ha: NBATeam.nba_teams[game_data[f"TEAM_NAME_{ha}"]] for ha in NBATeam.home_away
    }

    for ha in NBATeam.home_away:
        opposite_ha = NBATeam.get_opposite_home_away(ha)
        team_data_dic[ha]["GAME_DATE"] = game_data.name
        team_data_dic[ha]["HOME_GAME"] = 1 if "HOME" == ha else 0
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
        team_data_dic[ha]["PTS_oppo_ratio"] = (
            game_data["PTS" + "_" + ha] / game_data["PTS" + "_" + opposite_ha]
        )

        if not FTE_data.empty:
            (
                team_data_dic[ha]["FTE_ELO"],
                team_data_dic[ha]["FTE_ELO_winprob"],
            ) = pick_FTE_game_data(
                game_data[f"TEAM_ABBREVIATION_{ha}"],
                game_data.name,
                FTE_data,
                game_data,
            )

        formatted_data = pd.DataFrame.from_records(team_data_dic[ha], index=[0])
        team_obj_dic[ha].games = pd.concat(
            [team_obj_dic[ha].games, formatted_data], ignore_index=True
        )

    update_elo_ratings(team_obj_dic, game_data)


def pick_FTE_game_data(
    team_abbrv, game_date, FTE_data, game_data=pd.Series(dtype="object")
):
    if team_abbrv == "CHA":
        team_abbrv = "CHO"  # don't know why but 538 uses CHO for charlotte
    if team_abbrv == "NOH":
        team_abbrv = "NOP"  # don't know why but 538 uses CHO for charlotte
    if team_abbrv == "PHX":
        team_abbrv = "PHO"  # don't know why but 538 uses CHO for charlotte
    if team_abbrv == "BKN":
        team_abbrv = "BRK"  # don't know why but 538 uses CHO for charlotte
    FTE_data_game_day = FTE_data[FTE_data["date"] == game_date]
    found_game = False
    for n, row in FTE_data_game_day.iterrows():
        if row["team1"] == team_abbrv:
            FTE_elo = row["elo1_pre"]
            FTE_elo_winprob = row["elo_prob1"]
            found_game = True
        elif row["team2"] == team_abbrv:
            FTE_elo = row["elo2_pre"]
            FTE_elo_winprob = row["elo_prob2"]
            found_game = True
    if found_game:
        return FTE_elo, FTE_elo_winprob
    else:
        logger.error(
            f"""Can't find the game in the FTE data.
            \n --------\n {game_date}, {team_abbrv}
            \n ------------ \n{FTE_data_game_day}"""
        )
        logger.info(game_data)
        raise Exception("This should not happen!")


def update_elo_ratings(team_obj_dic: Dict[str, NBATeam], game_data: pd.Series) -> None:
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


def fill_team_game_data(
    games: pd.DataFrame, payroll_data: pd.DataFrame, FTE_csv_path: str = ""
) -> None:
    """Adds the statistics from all games
    provided to their respective teams

    Args:
        games (pd.DataFrame): Games to be analyzed
        payroll_data (pd.DataFrame): payroll data
    """
    FTE_data = pd.DataFrame()
    if FTE_csv_path:
        logger.info(
            f"Adding FiveThirtyEight ELO data from {FTE_csv_path} to the games..."
        )
        FTE_data = get_FTE_data(FTE_csv_path)
    logger.info(f"Loading {len(games)} games for {len(NBATeam.nba_teams)} teams")
    with Progress() as progress:
        task = progress.add_task("[green]Loading game data...", total=len(games))
        current_season = games.iloc[0]["SEASON"]
        for index, row in games.iterrows():
            if current_season != row["SEASON"]:
                NBATeam.reset_trueskill_sigma(4)
                current_season = row["SEASON"]
            extract_game_data(row, FTE_data)
            progress.update(task, advance=1)
    for team_name, team in NBATeam.nba_teams.items():
        team.games = team.games.set_index("GAME_DATE")
        team.add_payroll_data(payroll_data)


def get_train_data_from_game(
    game: pd.DataFrame,
    feature_list: List[str],
    mean_features: List[str],
    n_games_back: int,
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
        tmp_dic[ha] = team.get_stats_for_date(game.name, games_back=n_games_back)
        for feature in feature_list:
            train_data_dict[ha + "_" + feature] = tmp_dic[ha][feature]
        for feature in mean_features:
            train_data_dict[f"{ha}_{feature}_{n_games_back}G"] = tmp_dic[ha][
                f"{feature}_{n_games_back}G"
            ]
    for simple_int_feature in ["SEASON_ID", "SEASON", "GAME_ID"]:
        train_data_dict[simple_int_feature] = int(game[simple_int_feature])
    train_data_dict["is_Playoffs"] = int(game["SEASON_TYPE"] == "Playoffs")
    train_data_dict["GAME_DATE"] = game.name
    return train_data_dict


def create_train_data(
    games: pd.DataFrame,
    sql_db_connection: sqlite3.Connection,
    feature_list: List[str],
    mean_features: List[str],
    n_games_back: int,
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
            tmp = get_train_data_from_game(
                row, feature_list, mean_features, n_games_back
            )
            game_train_data = pd.DataFrame(tmp, index=[0])
            train_data = pd.concat([train_data, game_train_data], ignore_index=True)
            progress.update(task, advance=1)
    logger.info(
        f"""Saving trainings data to the train_data table with
        the following columns:\n{train_data.columns.values}"""
    )
    train_data.to_sql("train_data", sql_db_connection, if_exists="replace", index=False)


def get_game_data(sql_db_connection: sqlite3.Connection) -> pd.DataFrame:
    """Extract the game data from the SQL database created by the collect_game_data.py

    Args:
        sql_db_connection (sqlite3.Connection): Connection to the SQL database

    Returns:
        pd.DataFrame: Contains data for all games
    """
    games = pd.read_sql("SELECT * from NBA_games LIMIT 200", sql_db_connection)
    games["SEASON"] = pd.to_numeric(games["SEASON"])
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


def get_team_payroll_data(sql_db_connection: sqlite3.Connection) -> pd.DataFrame:
    payroll_data = pd.read_sql("SELECT * from team_payroll", sql_db_connection)
    return payroll_data


def get_all_team_names(games: pd.DataFrame) -> Set:
    if "TEAM_NAME_HOME" in games.columns:
        home_key = "TEAM_NAME_HOME"
        away_key = "TEAM_NAME_AWAY"
    elif "HOME_TEAM_NAME" in games.columns:
        home_key = "HOME_TEAM_NAME"
        away_key = "AWAY_TEAM_NAME"
    else:
        raise Exception(
            f"Can't find the team name column in the data frame! {games.columns}"
        )

    return set(np.concatenate((games[home_key].unique(), games[away_key].unique())))


def validate_payroll_data(payroll_data: pd.DataFrame, games: pd.DataFrame) -> None:
    everything_is_ok = True
    for season in games["SEASON"].unique():
        teams_in_season = get_all_team_names(games[games["SEASON"] == season])
        for team in teams_in_season:
            if (
                team
                not in payroll_data[payroll_data["season"] == season][
                    "team_name"
                ].unique()
            ):
                logger.error(f"No payroll data for {team} in the {season} season!")
                everything_is_ok = False
    if not everything_is_ok:
        raise Exception("Payroll data is not available for all teams/seasons!")


def get_FTE_data(path_to_csv: str) -> pd.DataFrame:
    FTE_data = pd.read_csv(
        path_to_csv,
        parse_dates=["date"],
        usecols=[
            "date",
            "season",
            "team1",
            "team2",
            "elo1_pre",
            "elo2_pre",
            "elo_prob1",
            "elo_prob2",
        ],
    )
    FTE_data["season"] = FTE_data["season"] - 1  # Convention used in my code is
    # that the season is identified by the year it starts in NOT the year it ends in
    return FTE_data


def main(config: Dict[str, Any]) -> None:
    """Creates the data used in the training and saves it to the SQL data base

    Args:
        config (Dict[str, Any]): config
    """
    if os.path.isfile(config["sql_db_path"]):
        connection = sqlite3.connect(config["sql_db_path"])
    else:
        logger.error(
            f"{config['sql_db_path']} does not exist! need to run collect_game_data.py first!"
        )
    trueskill.setup(mu=30, draw_probability=0)
    games = get_game_data(connection)

    # FIXME, this game is missing in the FTE data
    games = games[games["GAME_ID"] != "0020700367"]

    # initialize teams:
    for team_name in get_all_team_names(games):
        NBATeam(team_name)
    payroll_data = get_team_payroll_data(connection)
    logger.info("Validating payroll and game data compatibility...")
    validate_payroll_data(payroll_data, games)
    fill_team_game_data(
        games, payroll_data, config["create_train_data"]["FTE_csv_path"]
    )
    create_train_data(
        games,
        connection,
        config["create_train_data"]["feature_list"],
        config["create_train_data"]["mean_features"],
        config["create_train_data"]["mean_over_last_N_games"],
    )
    connection.close()


if __name__ == "__main__":
    main(config_modul.load_config(config_modul.get_comandline_arguments()["config"]))
