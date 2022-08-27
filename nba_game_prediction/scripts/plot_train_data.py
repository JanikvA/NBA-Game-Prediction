import math

import matplotlib.pyplot as plt
import pandas as pd

from nba_game_prediction import config_modul


def main(config):
    train_data = pd.read_csv(config["train_data_path"])
    train_data = train_data.drop(
        [
            "GAME_ID",
            "SEASON_ID",
            "AWAY_TEAM_NAME",
            "HOME_TEAM_NAME",
            "GAME_DATE",
            "AWAY_is_back_to_back",
            "HOME_is_back_to_back",
        ],
        axis=1,
    )
    # sns.heatmap(train_data.corr(), vmin=-1, vmax=1, annot=True)
    # sns.pairplot(data=train_data, hue="HOME_WL", diag_kws={"common_norm":False})
    bins = [0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1]
    pd.DataFrame()
    x_data = []
    y_pred_data = []
    y_actual_data = []
    y_err = []
    ratio = []
    for n, lower_bound in enumerate(bins):
        if n + 1 == len(bins):
            break
        b1 = train_data.loc[
            (train_data["HOME_ELO_winprob"] > lower_bound)
            & (train_data["HOME_ELO_winprob"] < bins[n + 1])
        ]
        bin_center = lower_bound + (bins[n + 1] - lower_bound) / 2
        x_data.append(bin_center)
        y_pred_data.append(b1["HOME_ELO_winprob"].mean())
        y_actual_data.append(b1["HOME_WL"].mean())
        W = b1["HOME_WL"].count() * b1["HOME_WL"].mean()
        L = b1["HOME_WL"].count() * (1 - b1["HOME_WL"].mean())
        # from error propagation of W/(W+L)
        unc = math.sqrt(
            L**2 * math.sqrt(W) / (W + L) ** 4 + W**2 * math.sqrt(L) / (L + W) ** 4
        )
        y_err.append(unc / y_pred_data[-1])
        ratio.append(y_actual_data[-1] / y_pred_data[-1])
    plt.errorbar(x=x_data, y=ratio, yerr=y_err, color="black")
    plt.savefig("test.png")


if __name__ == "__main__":
    main(config_modul.load_config(config_modul.get_comandline_arguments()["config"]))
