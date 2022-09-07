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
        cls, elo_a: float, elo_b: float, a_won: float, a_k: float = 20
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
    def rate_1vs1(cls, elo_obj_a: float, elo_obj_b: float) -> Tuple[float, float]:
        """Calculate updated elo ratings for both player where
        Player A, i.e. the first elo mentioned, won the match.

        Args:
            elo_obj_a (float): elo rating of player A. The winning player
            elo_obj_b (float): elo rating of player B. The losing player

        Returns:
            Tuple: the new elo of Player A, new elo of Player B
        """
        new_elo_a = ELO.calc_elo_change(elo_obj_a, elo_obj_b, a_won=True)
        new_elo_b = ELO.calc_elo_change(elo_obj_b, elo_obj_a, a_won=False)
        return new_elo_a, new_elo_b


""""
# TODO what to do with the stuff below?
class Player:
    def __init__(self, name, strength):
        self.name = name
        self.elo_history = []
        self.variability_history = []
        self.set_elo(1000)
        self.set_variability(80)
        self.strength = strength

    def set_elo(self, elo):
        self.elo = elo
        self.elo_history.append(elo)
        if len(self.elo_history) > 300 and self.strength == 1:
            self.strength = 4

    def set_variability(self, var):
        # print(var)
        min_var = 1
        if var < min_var:
            self.variability = min_var
        else:
            self.variability = var
        # print(self.variability)
        self.variability_history.append(self.variability)


def win_prob(player_a, player_b):
    return player_a.strength / (player_a.strength + player_b.strength)


def play_match(player_a, player_b):
    wr = win_prob(player_a, player_b)
    player_a_won = random.uniform(0, 1) < wr
    # if len(player_a.elo_history)>10:
    #     new_a_k=2+(player_a.elo_history[-1] - player_a.elo_history[-10]) / player_a.a_k
    #     new_b_k=2+(player_b.elo_history[-1] - player_b.elo_history[-10]) / player_b.a_k
    #     player_a.a_k=new_a_k
    #     player_b.a_k=new_b_k
    a_elo = calc_elo_change(
        player_a.elo, player_b.elo, player_a_won, player_a.variability
    )
    b_elo = calc_elo_change(
        player_b.elo, player_a.elo, not player_a_won, player_b.variability
    )
    player_a.set_elo(a_elo)
    player_b.set_elo(b_elo)
    win_streak_factor_a = 0
    win_streak_factor_b = 0

    # if len(player_a.elo_history) > 10:
    #     win_streak_factor_a = abs(
    #         player_a.elo_history[-1] - player_a.elo_history[-10]
    #     ) / (
    #         10
    #         * sum(player_a.variability_history[-10:])
    #         / len(player_a.variability_history[-10:])
    #     )
    # if len(player_b.elo_history) > 10:
    #     win_streak_factor_b = abs(
    #         player_b.elo_history[-1] - player_b.elo_history[-10]
    #     ) / (
    #         10
    #         * sum(player_b.variability_history[-10:])
    #         / len(player_b.variability_history[-10:])
    #     )

    # print("factor:",win_streak_factor_a)
    player_a.set_variability(
        player_a.variability * (1 - 1 / (len(player_a.elo_history) + 4) ** (1 / 1.01))
        + 0.1 * win_streak_factor_a
    )
    player_b.set_variability(
        player_b.variability * (1 - 1 / (len(player_b.elo_history) + 4) ** (1 / 1.01))
        + 0.1 * win_streak_factor_b
    )
    return player_a_won


def main():
    players = []
    n_players = 2
    for streng in range(n_players):
        players.append(Player(str(streng), streng + 1))

    for sim_n in range(20):
        for i in range(1000):
            for combi in list(itertools.combinations(players, 2)):
                play_match(combi[0], combi[1])

        data_dict = {"player": [], "game": [], "elo": [], "elo_change": []}
        for i in range(n_players):
            for game_n, elo in enumerate(players[i].elo_history):
                data_dict["player"].append(i)
                data_dict["game"].append(game_n)
                data_dict["elo"].append(elo)
                try:
                    data_dict["elo_change"].append(elo - data_dict["elo"][-2])
                except Exception as e:
                    data_dict["elo_change"].append(0)
                    print(e)
        data = pd.DataFrame(data_dict)
        data["elo_change_avg"] = data["elo_change"].rolling(10).sum()

        # data["elo_avg"]=data["elo"].rolling(100).mean()
        # sns.lineplot(data=data, x="game", y="elo_avg", hue="player")
        sns.lineplot(data=data, x="game", y="elo", hue="player")
        for p in players:
            p.__init__(p.name, int(p.name) + 1)

    # sns.lineplot(data=data, x="game", y="elo_change_avg", hue="player")
    plt.show()


def calc_avg_change(n_games, elo_a, elo_b, k_a):
    elo_change_a_won = calc_elo_change(elo_a, elo_b, 1, a_k=k_a) - elo_a
    elo_change_a_lost = calc_elo_change(elo_a, elo_b, 0, a_k=k_a) - elo_a
    possible_outcomes = []
    outcome_prob = []
    for comb in itertools.combinations_with_replacement([0, 1], n_games):
        counts = Counter(comb)
        # print("hi")
        # print(expected_Ea(elo_a,elo_b))
        # print(comb)
        # print(binom(n_games, counts[1]))
        # np.sqrt(np.cov(values, aweights=weights))
        possible_outcomes.append(
            counts[0] * elo_change_a_lost + counts[1] * elo_change_a_won
        )
        prob = (
            expected_Ea(elo_a, elo_b) ** counts[1]
            if counts[1]
            else (1 - expected_Ea(elo_a, elo_b)) ** counts[0]
        )
        outcome_prob.append(prob)
    # print(possible_outcomes)
    # print(outcome_prob)
    print(np.sqrt(np.cov(possible_outcomes, aweights=outcome_prob)))

    # print( statistics.mean(possible_outcomes) )
    # print( statistics.fmean(possible_outcomes) )
    # print( statistics.stdev(possible_outcomes) )


def test():
    a_won = calc_elo_change(1000, 1000, 1, a_k=1)
    a_lost = calc_elo_change(1000, 1000, 0, a_k=1)
    print(a_won, a_lost)
    calc_avg_change(10, 1000, 1400, 30)


if __name__ == "__main__":
    # main()
    test()

"""
