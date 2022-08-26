from nba_game_prediction import collect_game_data, config_modul


def test_combine_team_games():
    pass


def test_main():
    collect_game_data.main(config_modul.load_config("data/test_config.yaml"))
