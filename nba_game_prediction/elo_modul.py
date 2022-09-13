import math
from typing import Tuple

import trueskill


def trueskill_win_probability(
    team1: trueskill.Rating, team2: trueskill.Rating
) -> float:
    """Calculate probability for team1 to win based on trueskill. For more
    information see https://trueskill.org/#trueskill.quality_1vs1 and
    https://github.com/sublee/trueskill/issues/1#issuecomment-149762508.

    Args:
        team1 (trueskill.Rating): trueskill.Rating of team1
        team2 (trueskill.Rating): trueskill.Rating of team2

    Returns:
        float: Returns win probability for team1
    """
    delta_mu = team1.mu - team2.mu
    sum_sigma = team1.sigma**2 + team2.sigma**2
    ts = trueskill.global_env()
    denom = math.sqrt(2 * (ts.beta * ts.beta) + sum_sigma)
    return ts.cdf(delta_mu / denom)


class ELO:
    def __init__(self, elo: float) -> None:
        self.elo = elo

    @classmethod
    def Rating(cls, elo: float = 1400) -> float:
        return elo

    @classmethod
    def expected_Ea(cls, elo_a: float, elo_b: float) -> float:
        """Represents the win probability for player a.

        Args:
            elo_a (float): ELO rating of player a
            elo_b (float): ELO rating of player b

        Returns:
            float: win probability for player a
        """
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    @classmethod
    def calc_elo_change(
        cls, elo_a: float, elo_b: float, a_won: bool, a_k: float = 20
    ) -> float:
        """Calculate elo change for player a base on the standard Elo rating used
        in chess https://en.wikipedia.org/wiki/Elo_rating_system.

        Args:
            elo_a (float): Elo rating of player a
            elo_b (float): Elo rating of player b
            a_won (bool): True if player a won
            a_k (float, optional): This number effects how big the impact of one
            particular game on the rating is. Large values mean high fluctuations
            but also rapid adaptation in case of change in skill (e.g. trades, injuries).
            Defaults to 10.

        Returns:
            float: New Elo rating after the match
        """
        Ea = cls.expected_Ea(elo_a, elo_b)
        new_elo_a = elo_a + a_k * (a_won - Ea)
        return new_elo_a

    @classmethod
    def rate_1vs1(cls, elo_a: float, elo_b: float) -> Tuple[float, float]:
        """Calculate updated elo ratings for both player where
        Player A, i.e. the first elo mentioned, won the match.

        Args:
            elo_a (float): elo rating of player A. The winning player
            elo_b (float): elo rating of player B. The losing player

        Returns:
            Tuple: the new elo of Player A, new elo of Player B
        """
        new_elo_a = ELO.calc_elo_change(elo_a, elo_b, a_won=True)
        new_elo_b = ELO.calc_elo_change(elo_b, elo_a, a_won=False)
        return new_elo_a, new_elo_b
