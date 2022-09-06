import datetime
import sqlite3

import numpy as np
import pandas as pd
import trueskill
from loguru import logger
from rich.progress import Progress

from nba_game_prediction import config_modul, elo_modul


class NBATeam:
    nba_teams = {}
    home_away = ["HOME", "AWAY"]
    team_stats = [
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

    @classmethod
    def get_opposite_home_away(cls, home_away):
        if home_away not in cls.home_away:
            raise Exception(f"{home_away} has to be in {cls.home_away}")
        return "HOME" if home_away == "AWAY" else "AWAY"

    def __init__(self, name):
        self.name = name
        self.games = pd.DataFrame()
        self.elo = 1400
        self.trueskill = trueskill.Rating()
        if self.name in NBATeam.nba_teams:
            raise Exception(f"{self.name} is already in NBATeam.nba_teams!")
        else:
            NBATeam.nba_teams[self.name] = self

    def get_stats_between_dates(self, from_date, to_date):
        selected_games = self.games[
            (self.games["GAME_DATE"] < to_date) & (self.games["GAME_DATE"] > from_date)
        ]
        data = selected_games.mean(numeric_only=True).to_dict()
        for k in list(data.keys()):
            data[k + "_mean"] = data.pop(k)
        return data

    def get_stats_for_date(self, date, games_back=10):
        last_ten_games = self.get_last_N_games(date, games_back)
        data = last_ten_games.mean(numeric_only=True).to_dict()
        for k in list(data.keys()):
            data[k + f"_{games_back}G"] = data.pop(k)
        if self.games[self.games["GAME_DATE"] == date].empty:
            logger.warning(f"No Game was played on the given date ({date})!")
        else:
            day_before = date - datetime.timedelta(days=1)
            data["is_back_to_back"] = int(
                not self.games[self.games["GAME_DATE"] == day_before].empty
            )
            for key in [
                "ELO",
                "ELO_winprob",
                "trueskill_mu",
                "trueskill_sigma",
                "trueskill_winprob",
            ]:
                data[key] = self.games[self.games["GAME_DATE"] == date][key].values[0]
        return data

    # TODO this is inefficient
    def get_last_N_games(self, date, n_games=10):
        games_before = self.games[self.games["GAME_DATE"] < date]
        return games_before.head(n_games)


def extract_game_data(game_data):
    team_data_dic = {ha: {} for ha in NBATeam.home_away}
    team_obj_dic = {
        ha: NBATeam.nba_teams[game_data[f"TEAM_NAME_{ha}"]] for ha in NBATeam.home_away
    }

    team_data_dic["HOME"]["HOME_GAME"] = 1
    team_data_dic["AWAY"]["HOME_GAME"] = 0
    team_data_dic["HOME"]["TEAM_NAME_opponent"] = game_data["TEAM_NAME_AWAY"]
    team_data_dic["AWAY"]["TEAM_NAME_opponent"] = game_data["TEAM_NAME_HOME"]

    for ha in NBATeam.home_away:
        opposite_ha = NBATeam.get_opposite_home_away(ha)
        for uniq in ["SEASON_ID", "GAME_ID", "GAME_DATE", "SEASON_TYPE", "SEASON"]:
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


def fill_team_game_data(games):
    logger.info(f"Loading {len(games)} games for {len(NBATeam.nba_teams)} teams")
    with Progress() as progress:
        task = progress.add_task("[green]Loading game data...", total=len(games))
        for index, row in games.iterrows():
            extract_game_data(row)
            progress.update(task, advance=1)


def get_train_data_from_game(game, feature_list):
    tmp_dic = {}
    train_data_dict = {}
    for ha in NBATeam.home_away:
        team = NBATeam.nba_teams[game[f"TEAM_NAME_{ha}"]]
        train_data_dict[f"{ha}_TEAM_NAME"] = game[f"TEAM_NAME_{ha}"]
        train_data_dict[f"{ha}_WL"] = int(game[f"WL_{ha}"])
        tmp_dic[ha] = team.get_stats_for_date(game["GAME_DATE"])
        for feature in feature_list:

            train_data_dict[ha + "_" + feature] = tmp_dic[ha][feature]
    for simple_int_feature in ["SEASON_ID", "SEASON", "GAME_ID"]:
        train_data_dict[simple_int_feature] = int(game[simple_int_feature])
    train_data_dict["is_Playoffs"] = int(game["SEASON_TYPE"] == "Playoffs")
    train_data_dict["GAME_DATE"] = game["GAME_DATE"]
    return train_data_dict


def create_train_data(games, sql_db_connection, feature_list):
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


def get_game_data(sql_db_connection):
    games = pd.read_sql("SELECT * from NBA_games", sql_db_connection)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games["WL_HOME"] = games.apply(
        lambda row: 1 if row["WL_HOME"] == "W" else 0, axis=1
    )
    games["WL_AWAY"] = games.apply(
        lambda row: 1 if row["WL_AWAY"] == "W" else 0, axis=1
    )
    return games


def main(config):
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
