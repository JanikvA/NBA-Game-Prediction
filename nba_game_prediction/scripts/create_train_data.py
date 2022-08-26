import datetime

import pandas as pd
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

    def __init__(self, name):
        self.name = name
        self.games = pd.DataFrame()
        self.elo = 1400
        if self.name in NBATeam.nba_teams:
            raise Exception(f"{self.name} is already in NBATeam.nba_teams!")
        else:
            NBATeam.nba_teams[self.name] = self

    def add_game(self, game):
        tmp_dic = {}
        for uniq in ["SEASON_ID", "GAME_ID", "GAME_DATE"]:
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

        oppo_elo = get_team_elo(tmp_dic["TEAM_NAME_opponent"])
        tmp_dic["ELO"] = self.elo
        tmp_dic["ELO_opponent"] = oppo_elo
        self.elo = elo_modul.calc_elo_change(self.elo, oppo_elo, tmp_dic["WL"])

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
            data["is_back_to_back"] = not self.games[
                self.games["GAME_DATE"] == day_before
            ].empty
            data["ELO"] = self.games[self.games["GAME_DATE"] == date]["ELO"].values[0]
        return data

    def get_last_N_games(self, date, n_games=10):
        games_before = self.games[self.games["GAME_DATE"] < date]
        return games_before.head(n_games)


def get_team_elo(team_name):
    if team_name in NBATeam.nba_teams:
        return NBATeam.nba_teams[team_name].elo
    else:
        return 1400


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


def get_train_data_from_game(game):
    tmp_dic = {}
    for ha in NBATeam.home_away:
        team = NBATeam.nba_teams[game[f"TEAM_NAME_{ha}"]]
        tmp_dic[ha] = team.get_stats_for_date(game["GAME_DATE"])
    train_data_dict = {
        **{"HOME_" + k: v for k, v in tmp_dic["HOME"].items()},
        **{"AWAY_" + k: v for k, v in tmp_dic["AWAY"].items()},
    }
    train_data_dict["HOME_WL"] = int(game["WL_HOME"])
    return train_data_dict


def create_train_data(games, output_path):
    train_data = pd.DataFrame()
    logger.info(f"Transforming data into trainings data for {len(games)} games.")
    with Progress() as progress:
        task = progress.add_task("[green]Transforming game data...", total=len(games))
        for index, row in games.iterrows():
            tmp = get_train_data_from_game(row)
            game_train_data = pd.DataFrame(tmp, index=[0])
            train_data = pd.concat([train_data, game_train_data], ignore_index=True)
            progress.update(task, advance=1)
    train_data.to_csv(output_path)


def get_game_data(game_data_path):
    games = pd.read_csv(game_data_path)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games["WL_HOME"] = games.apply(
        lambda row: 1 if row["WL_HOME"] == "W" else 0, axis=1
    )
    games["WL_AWAY"] = games.apply(
        lambda row: 1 if row["WL_AWAY"] == "W" else 0, axis=1
    )
    return games


def main(config):
    games = get_game_data(config["combined_output_path"])
    fill_team_game_data(games)
    create_train_data(games, config["train_data_output_path"])


if __name__ == "__main__":
    main(config_modul.load_config(config_modul.get_comandline_arguments()["config"]))
