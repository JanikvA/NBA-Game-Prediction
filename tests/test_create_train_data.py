import pandas as pd
import pytest

from nba_game_prediction.scripts.create_train_data import NBATeam, update_elo_ratings


class TestNBATeam:
    cavs = NBATeam("Clevland Cavaliers")
    warriors = NBATeam("Golden State Warriors")

    @pytest.mark.parametrize("ha", ["HOME", "AWAY"])
    def test_get_opposite_home_away(self, ha):
        oppo = NBATeam.get_opposite_home_away(ha)
        if ha == "HOME":
            assert oppo == "AWAY"
        elif ha == "AWAY":
            assert oppo == "HOME"

    def test_reset_trueskill_sigma(self):
        NBATeam.reset_trueskill_sigma(4)
        assert self.cavs.trueskill.sigma == 4
        assert self.warriors.trueskill.sigma == 4

    def test_update_elo_ratings(self):
        team_obj_dic = {"HOME": self.cavs, "AWAY": self.warriors}
        game_data = pd.Series({"WL_HOME": 1})
        update_elo_ratings(team_obj_dic, game_data)
        assert pytest.approx(team_obj_dic["HOME"].elo) == 1410
        assert pytest.approx(team_obj_dic["AWAY"].elo) == 1390
        assert team_obj_dic["HOME"].trueskill.mu > 25
        assert team_obj_dic["AWAY"].trueskill.mu < 25
