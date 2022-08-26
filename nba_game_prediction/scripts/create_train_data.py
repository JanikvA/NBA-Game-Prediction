import datetime

import pandas as pd
import progressbar


def expected_Ea(mmr_a, mmr_b):
    return 1 / (1 + 10 ** ((mmr_b - mmr_a) / 400))


# def calc_MMR_change(mmr_a, mmr_b, a_won, a_k=40, b_k=40):
#     Ea=expected_Ea(mmr_a, mmr_b)
#     new_mmr_a= mmr_a + a_k * (a_won - Ea)
#     new_mmr_b= mmr_b + b_k * ((1-a_won) - (1-Ea))
#     return new_mmr_a, new_mmr_b


def calc_MMR_change(mmr_a, mmr_b, a_won, a_k=40, b_k=40):
    Ea = expected_Ea(mmr_a, mmr_b)
    new_mmr_a = mmr_a + a_k * (a_won - Ea)
    return new_mmr_a


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
        self.mmr = 1400
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

        oppo_mmr = get_team_mmr(tmp_dic["TEAM_NAME_opponent"])
        tmp_dic["MMR"] = self.mmr
        tmp_dic["MMR_opponent"] = oppo_mmr
        self.mmr = calc_MMR_change(self.mmr, oppo_mmr, tmp_dic["WL"])

        formatted_data = pd.DataFrame(tmp_dic, index=[0])
        self.games = self.games.append(formatted_data, ignore_index=True)

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

        MMR
        MMR_uncertainty

        has_major_injury
        teams_market_size
        travel distance for away team
        """
        last_ten_games = self.get_last_N_games(date, games_back)
        data = last_ten_games.mean(numeric_only=True).to_dict()
        for k in list(data.keys()):
            data[k + f"_{games_back}G"] = data.pop(k)
        if self.games[self.games["GAME_DATE"] == date].empty:
            print("WARNING: No Game was played on the given date")
        else:
            day_before = date - datetime.timedelta(days=1)
            data["is_back_to_back"] = not self.games[
                self.games["GAME_DATE"] == day_before
            ].empty
            data["MMR"] = self.games[self.games["GAME_DATE"] == date]["MMR"].values[0]
        return data

    def get_last_N_games(self, date, n_games=10):
        games_before = self.games[self.games["GAME_DATE"] < date]
        return games_before.head(n_games)


def get_team_mmr(team_name):
    if team_name in NBATeam.nba_teams:
        return NBATeam.nba_teams[team_name].mmr
    else:
        return 1400


def fill_team_game_data(games):
    print("Loading team game data")
    for index, row in progressbar.progressbar(games.iterrows()):
        for ha in NBATeam.home_away:
            team_name = row[f"TEAM_NAME_{ha}"]
            if team_name not in NBATeam.nba_teams:
                team = NBATeam(team_name)
            else:
                team = NBATeam.nba_teams[team_name]
            team.add_game(row)


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


def create_train_data(games):
    train_data = pd.DataFrame()
    print("Transforming data into trainings data")
    for index, row in progressbar.progressbar(games.iterrows()):
        tmp = get_train_data_from_game(row)
        game_train_data = pd.DataFrame(tmp, index=[0])
        train_data = train_data.append(game_train_data, ignore_index=True)
    train_data.to_csv("train_data.csv")


def get_game_data():
    games = pd.read_csv("combined_game_data.csv")
    # games=pd.read_csv("test_data.csv")
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games["WL_HOME"] = games.apply(
        lambda row: 1 if row["WL_HOME"] == "W" else 0, axis=1
    )
    games["WL_AWAY"] = games.apply(
        lambda row: 1 if row["WL_AWAY"] == "W" else 0, axis=1
    )
    return games


def main():
    games = get_game_data()
    fill_team_game_data(games)
    create_train_data(games)


if __name__ == "__main__":
    main()
