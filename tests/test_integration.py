import pytest
import requests

from nba_game_prediction import config_modul
from nba_game_prediction.scripts import (
    collect_game_data,
    create_train_data,
    plot_train_data,
    train_model,
)


def test_pass_dummy():
    assert True


def test_fail_dummy():
    assert False


def test_request():
    r = requests.get("https://realpython.com/pytest-python-testing/")
    r_html = r.text
    assert r_html


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
