import pytest

from nba_game_prediction.scripts.plot_train_data import get_binominal_std_dev_on_prob


def test_get_binominal_std_dev_on_prob():
    unc = get_binominal_std_dev_on_prob(100, 0.5)
    assert pytest.approx(unc) == 0.05
