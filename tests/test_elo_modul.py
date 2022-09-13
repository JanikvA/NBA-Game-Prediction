import pytest
import trueskill

from nba_game_prediction.elo_modul import ELO, trueskill_win_probability


def test_trueskill_win_probability():
    team1 = trueskill.Rating()
    team2 = trueskill.Rating()
    win_prob = trueskill_win_probability(team1, team2)
    assert pytest.approx(win_prob) == 0.5


class TestELO:
    elo_a = ELO(1400)
    elo_b = ELO(1400)

    def test_expected_Ea(cls):
        win_prob = ELO.expected_Ea(cls.elo_a.elo, cls.elo_b.elo)
        assert pytest.approx(win_prob) == 0.5

    def test_calc_elo_change(cls):
        new_elo_a = ELO.calc_elo_change(
            cls.elo_a.elo, cls.elo_b.elo, a_won=True, a_k=20
        )
        assert pytest.approx(new_elo_a) == 1410

    def test_rate_1vs1(cls):
        new_elo_a, new_elo_b = ELO.rate_1vs1(cls.elo_a.elo, cls.elo_b.elo)
        assert pytest.approx(new_elo_a) == 1410
        assert pytest.approx(new_elo_b) == 1390
