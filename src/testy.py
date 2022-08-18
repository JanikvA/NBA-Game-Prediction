import itertools
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def expected_Ea(mmr_a, mmr_b):
    return 1 / (1 + 10 ** ((mmr_b - mmr_a) / 400))


def calc_MMR_change(mmr_a, mmr_b, a_won, a_k=3, b_k=5):
    Ea = expected_Ea(mmr_a, mmr_b)
    new_mmr_a = mmr_a + a_k * (a_won - Ea)
    # print("MMR change:",a_k * (a_won - Ea))
    return new_mmr_a


class Player:
    def __init__(self, name, strength):
        self.name = name
        self.mmr_history = []
        self.variability_history = []
        self.set_mmr(1000)
        self.set_variability(80)
        self.strength = strength

    def set_mmr(self, mmr):
        self.mmr = mmr
        self.mmr_history.append(mmr)
        if len(self.mmr_history) > 300 and self.strength == 1:
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
    # if len(player_a.mmr_history)>10:
    #     new_a_k=2+(player_a.mmr_history[-1] - player_a.mmr_history[-10]) / player_a.a_k
    #     new_b_k=2+(player_b.mmr_history[-1] - player_b.mmr_history[-10]) / player_b.a_k
    #     player_a.a_k=new_a_k
    #     player_b.a_k=new_b_k
    a_mmr = calc_MMR_change(
        player_a.mmr, player_b.mmr, player_a_won, player_a.variability
    )
    b_mmr = calc_MMR_change(
        player_b.mmr, player_a.mmr, not player_a_won, player_b.variability
    )
    player_a.set_mmr(a_mmr)
    player_b.set_mmr(b_mmr)
    win_streak_factor_a = 0
    win_streak_factor_b = 0

    # if len(player_a.mmr_history) > 10:
    #     win_streak_factor_a = abs(
    #         player_a.mmr_history[-1] - player_a.mmr_history[-10]
    #     ) / (
    #         10
    #         * sum(player_a.variability_history[-10:])
    #         / len(player_a.variability_history[-10:])
    #     )
    # if len(player_b.mmr_history) > 10:
    #     win_streak_factor_b = abs(
    #         player_b.mmr_history[-1] - player_b.mmr_history[-10]
    #     ) / (
    #         10
    #         * sum(player_b.variability_history[-10:])
    #         / len(player_b.variability_history[-10:])
    #     )

    # print("factor:",win_streak_factor_a)
    player_a.set_variability(
        player_a.variability * (1 - 1 / (len(player_a.mmr_history) + 4) ** (1 / 1.01))
        + 0.1 * win_streak_factor_a
    )
    player_b.set_variability(
        player_b.variability * (1 - 1 / (len(player_b.mmr_history) + 4) ** (1 / 1.01))
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

        data_dict = {"player": [], "game": [], "mmr": [], "mmr_change": []}
        for i in range(n_players):
            for game_n, mmr in enumerate(players[i].mmr_history):
                data_dict["player"].append(i)
                data_dict["game"].append(game_n)
                data_dict["mmr"].append(mmr)
                try:
                    data_dict["mmr_change"].append(mmr - data_dict["mmr"][-2])
                except Exception as e:
                    data_dict["mmr_change"].append(0)
                    print(e)
        data = pd.DataFrame(data_dict)
        data["mmr_change_avg"] = data["mmr_change"].rolling(10).sum()

        # data["mmr_avg"]=data["mmr"].rolling(100).mean()
        # sns.lineplot(data=data, x="game", y="mmr_avg", hue="player")
        sns.lineplot(data=data, x="game", y="mmr", hue="player")
        for p in players:
            p.__init__(p.name, int(p.name) + 1)

    # sns.lineplot(data=data, x="game", y="mmr_change_avg", hue="player")
    plt.show()


def calc_avg_change(n_games, mmr_a, mmr_b, k_a):
    mmr_change_a_won = calc_MMR_change(mmr_a, mmr_b, 1, a_k=k_a) - mmr_a
    mmr_change_a_lost = calc_MMR_change(mmr_a, mmr_b, 0, a_k=k_a) - mmr_a
    possible_outcomes = []
    outcome_prob = []
    for comb in itertools.combinations_with_replacement([0, 1], n_games):
        counts = Counter(comb)
        # print("hi")
        # print(expected_Ea(mmr_a,mmr_b))
        # print(comb)
        # print(binom(n_games, counts[1]))
        # np.sqrt(np.cov(values, aweights=weights))
        possible_outcomes.append(
            counts[0] * mmr_change_a_lost + counts[1] * mmr_change_a_won
        )
        prob = (
            expected_Ea(mmr_a, mmr_b) ** counts[1]
            if counts[1]
            else (1 - expected_Ea(mmr_a, mmr_b)) ** counts[0]
        )
        outcome_prob.append(prob)
    # print(possible_outcomes)
    # print(outcome_prob)
    print(np.sqrt(np.cov(possible_outcomes, aweights=outcome_prob)))

    # print( statistics.mean(possible_outcomes) )
    # print( statistics.fmean(possible_outcomes) )
    # print( statistics.stdev(possible_outcomes) )


def test():
    a_won = calc_MMR_change(1000, 1000, 1, a_k=1)
    a_lost = calc_MMR_change(1000, 1000, 0, a_k=1)
    print(a_won, a_lost)
    calc_avg_change(10, 1000, 1400, 30)


if __name__ == "__main__":
    # main()
    test()
