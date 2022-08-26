import elo


def test_expected_Ea():
    assert elo.expected_Ea(1400, 1400) == 0.5
