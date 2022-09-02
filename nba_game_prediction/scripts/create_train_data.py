import datetime
import sqlite3

import pandas as pd
import trueskill
from loguru import logger
from rich.progress import Progress

from nba_game_prediction import config_modul, elo_modul

# TODO make API for elo_modul similar to that of trueskill


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

    def __init__(self, name):
        self.name = name
        self.games = pd.DataFrame()
        self.elo = 1400
        self.trueskill = trueskill.Rating()
        if self.name in NBATeam.nba_teams:
            raise Exception(f"{self.name} is already in NBATeam.nba_teams!")
        else:
            NBATeam.nba_teams[self.name] = self

    def add_game(self, game):
        tmp_dic = {}
        for uniq in ["SEASON_ID", "GAME_ID", "GAME_DATE", "SEASON_TYPE", "SEASON"]:
            tmp_dic[uniq] = game[uniq]

        HOME_GAME = game["TEAM_NAME_HOME"] == self.name
        if not HOME_GAME and game["TEAM_NAME_AWAY"] != self.name:
            raise Exception(
                f"Something went wrong! {HOME_GAME=}, {self.name=}, {game=}"
            )

        if HOME_GAME:
            oppo = "_AWAY"
            this_team = "_HOME"
        else:
            oppo = "_HOME"
            this_team = "_AWAY"

        for team_stat in NBATeam.team_stats:
            tmp_dic[team_stat] = game[team_stat + this_team]
            tmp_dic[team_stat + "_opponent"] = game[team_stat + oppo]

        tmp_dic["HOME_GAME"] = int(HOME_GAME)
        tmp_dic["WL"] = int(game[f"WL{this_team}"])
        tmp_dic["TEAM_NAME_opponent"] = game[f"TEAM_NAME{oppo}"]

        # FIXME not quite correct because each game is analyzed twice (once for each team)
        # - but in the second run the elo/trueskill of the home team will
        # already be updated from the results of this game.
        oppo_elo = get_team_elo(tmp_dic["TEAM_NAME_opponent"])
        tmp_dic["ELO"] = self.elo
        tmp_dic["ELO_opponent"] = oppo_elo
        tmp_dic["ELO_winprob"] = elo_modul.expected_Ea(self.elo, oppo_elo)
        self.elo = elo_modul.calc_elo_change(self.elo, oppo_elo, tmp_dic["WL"])

        oppo_trueskill = get_team_trueskill(tmp_dic["TEAM_NAME_opponent"])
        tmp_dic["trueskill_mu"] = self.trueskill.mu
        tmp_dic["trueskill_sigma"] = self.trueskill.sigma
        tmp_dic["trueskill_mu_opponent"] = oppo_trueskill.mu
        tmp_dic["trueskill_sigma_opponent"] = oppo_trueskill.sigma
        tmp_dic["trueskill_winprob"] = elo_modul.trueskill_win_probability(
            self.trueskill, oppo_trueskill
        )
        if tmp_dic["WL"] == 1:
            self.trueskill, _ = trueskill.rate_1vs1(self.trueskill, oppo_trueskill)
        else:
            _, self.trueskill = trueskill.rate_1vs1(oppo_trueskill, self.trueskill)

        formatted_data = pd.DataFrame(tmp_dic, index=[0])
        self.games = pd.concat([self.games, formatted_data], ignore_index=True)

    def get_stats_between_dates(self, from_date, to_date):
        selected_games = self.games[
            (self.games["GAME_DATE"] < to_date) & (self.games["GAME_DATE"] > from_date)
        ]
        data = selected_games.mean(numeric_only=True).to_dict()
        for k in list(data.keys()):
            data[k + "_mean"] = data.pop(k)
        return data

    def get_stats_for_date(self, date, games_back=10):
        """
        WR_last_season
        WR_HOME_last_season
        WR_AWAY_last_season

        ELO
        ELO_uncertainty

        has_major_injury
        teams_market_size
        travel distance for away team
        """
        last_ten_games = self.get_last_N_games(date, games_back)
        data = last_ten_games.mean(numeric_only=True).to_dict()
        for k in list(data.keys()):
            data[k + f"_{games_back}G"] = data.pop(k)
        if self.games[self.games["GAME_DATE"] == date].empty:
            logger.warning("No Game was played on the given date!")
        else:
            day_before = date - datetime.timedelta(days=1)
            # TODO make this more DRY
            data["is_back_to_back"] = int(
                not self.games[self.games["GAME_DATE"] == day_before].empty
            )
            data["ELO"] = self.games[self.games["GAME_DATE"] == date]["ELO"].values[0]
            data["ELO_winprob"] = self.games[self.games["GAME_DATE"] == date][
                "ELO_winprob"
            ].values[0]
            data["trueskill_mu"] = self.games[self.games["GAME_DATE"] == date][
                "trueskill_mu"
            ].values[0]
            data["trueskill_sigma"] = self.games[self.games["GAME_DATE"] == date][
                "trueskill_sigma"
            ].values[0]
            data["trueskill_winprob"] = self.games[self.games["GAME_DATE"] == date][
                "trueskill_winprob"
            ].values[0]
        return data

    def get_last_N_games(self, date, n_games=10):
        games_before = self.games[self.games["GAME_DATE"] < date]
        return games_before.head(n_games)


# TODO the following two methods are not really needed.
# Just initalize all teams at the start after loading games
def get_team_elo(team_name):
    if team_name not in NBATeam.nba_teams:
        NBATeam(team_name)
    return NBATeam.nba_teams[team_name].elo


def get_team_trueskill(team_name):
    if team_name not in NBATeam.nba_teams:
        NBATeam(team_name)
    return NBATeam.nba_teams[team_name].trueskill


def fill_team_game_data(games):
    logger.info(f"Loading team game data for {len(games)} games")
    with Progress() as progress:
        task = progress.add_task("[green]Loading game data...", total=len(games))
        for index, row in games.iterrows():
            for ha in NBATeam.home_away:
                team_name = row[f"TEAM_NAME_{ha}"]
                if team_name not in NBATeam.nba_teams:
                    team = NBATeam(team_name)
                else:
                    team = NBATeam.nba_teams[team_name]
                team.add_game(row)
            progress.update(task, advance=1)


def get_train_data_from_game(game, feature_list):
    tmp_dic = {}
    for ha in NBATeam.home_away:
        team = NBATeam.nba_teams[game[f"TEAM_NAME_{ha}"]]
        tmp_dic[ha] = team.get_stats_for_date(game["GAME_DATE"])
    train_data_dict = {}
    for feature in feature_list:
        train_data_dict["HOME_" + feature] = tmp_dic["HOME"][feature]
        train_data_dict["AWAY_" + feature] = tmp_dic["AWAY"][feature]
    train_data_dict["HOME_WL"] = int(game["WL_HOME"])
    train_data_dict["SEASON_ID"] = int(game["SEASON_ID"])
    train_data_dict["SEASON"] = int(game["SEASON"])
    train_data_dict["GAME_ID"] = int(game["GAME_ID"])
    train_data_dict["is_Playoffs"] = int(game["SEASON_TYPE"] == "Playoffs")
    train_data_dict["GAME_DATE"] = game["GAME_DATE"]
    train_data_dict["HOME_TEAM_NAME"] = game["TEAM_NAME_HOME"]
    train_data_dict["AWAY_TEAM_NAME"] = game["TEAM_NAME_AWAY"]
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
    fill_team_game_data(games)
    create_train_data(games, connection, config["create_train_data"]["feature_list"])
    connection.close()


if __name__ == "__main__":
    main(config_modul.load_config(config_modul.get_comandline_arguments()["config"]))
