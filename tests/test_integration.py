import pytest

from nba_game_prediction import config_modul
from nba_game_prediction.scripts import (
    collect_game_data,
    create_train_data,
    plot_train_data,
    train_model,
)


@pytest.mark.integration
def test_collect_game_data():
    collect_game_data.main(config_modul.load_config("data/test_config.yaml"))


@pytest.mark.integration
def test_create_train_data():
    create_train_data.main(config_modul.load_config("data/test_config.yaml"))


@pytest.mark.integration
def test_plot_train_data():
    plot_train_data.main(config_modul.load_config("data/test_config.yaml"))


@pytest.mark.integration
def test_train_model():
    train_model.main(config_modul.load_config("data/test_config.yaml"))
