import pytest

from nba_game_prediction import config_modul
from nba_game_prediction.scripts import (
    collect_game_data,
    create_train_data,
    plot_train_data,
    train_model,
)


# TODO nba_api has blacklisted github actions so can't test this.
# The test needs to be run locally and the output has to be saved to the repo
# with git lfs. Should move SQL db to MS Azure cloud
@pytest.mark.not_with_ga
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
