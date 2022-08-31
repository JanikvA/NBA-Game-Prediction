import math
import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from nba_game_prediction import config_modul


def pred_vs_actual_prob_comparison(train_data, prob_key, out_dir):
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
            (train_data[prob_key] > lower_bound) & (train_data[prob_key] < bins[n + 1])
        ]
        bin_center = lower_bound + (bins[n + 1] - lower_bound) / 2
        x_data.append(bin_center)
        y_pred_data.append(b1[prob_key].mean())
        y_actual_data.append(b1["HOME_WL"].mean())
        W = b1["HOME_WL"].count() * b1["HOME_WL"].mean()
        L = b1["HOME_WL"].count() * (1 - b1["HOME_WL"].mean())
        # from error propagation of W/(W+L)
        unc = math.sqrt(
            L**2 * math.sqrt(W) / (W + L) ** 4 + W**2 * math.sqrt(L) / (L + W) ** 4
        )
        y_err.append(unc / y_pred_data[-1])
        ratio.append(y_actual_data[-1] / y_pred_data[-1])
    # TODO add plot for a clasifier that always assigns 50% wr probability and one
    # which assigns a random probability and one that assigns randomly but with pdf from data
    plt.errorbar(x=x_data, y=ratio, yerr=y_err, color="black")
    out_file_name = os.path.join(out_dir, "probability_comparison_" + prob_key + ".png")
    logger.info(f"Saving plot to: {out_file_name}")
    plt.savefig(out_file_name)
    plt.clf()


def feature_correlation(train_data, method, out_dir):
    """Produces heatmap from correlations

    Args:
        train_data (pandas.DataFrame): dataframe with data
        method (string): defines which method for calculating the
                         correlation is used (choices: pearson, kendall, spearman)
    """
    plot_data = train_data.drop(
        [
            "SEASON_ID",
            "AWAY_TEAM_NAME",
            "HOME_TEAM_NAME",
            "GAME_DATE",
        ],
        axis=1,
    )
    sns.heatmap(plot_data.corr(method=method), vmin=-1, vmax=1, annot=True)
    out_file_name = os.path.join(out_dir, "feature_correlation_" + method + ".png")
    logger.info(f"Saving plot to: {out_file_name}")
    plt.savefig(out_file_name)
    plt.clf()


def feature_pair_plot(train_data, out_dir):
    plot_data = train_data.drop(
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
    sns.pairplot(data=plot_data, hue="HOME_WL", diag_kws={"common_norm": False})
    out_file_name = os.path.join(out_dir, "feature_pair_plot.png")
    logger.info(f"Saving plot to: {out_file_name}")
    plt.savefig(out_file_name)
    plt.clf()


def add_random_probs(data):
    data["random_winprob"] = data.apply(lambda row: random.uniform(0, 1), axis=1)

    # data["random_winprob_with_data_pdf"] = data.apply(lambda row: random.,1), axis=1)
    # bins = [0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1]
    # distribution = (1./6, 2./6, 3./6)
    # np.random.choice(np.arange(len(distribution)), p=distribution)
    # np.random.choice(np.arange(len(distribution)), p=distribution, size=10)


def main(config):
    train_data = pd.read_csv(config["train_data_path"])
    add_random_probs(train_data)
    for prob in ["HOME_ELO_winprob", "HOME_trueskill_winprob", "random_winprob"]:
        pred_vs_actual_prob_comparison(train_data, prob, config["output_dir"])
    # for method in ["pearson", "kendall", "spearman"]:
    #     feature_correlation(train_data, method, config["output_dir"])
    # feature_pair_plot(train_data, config["output_dir"])


if __name__ == "__main__":
    main(config_modul.load_config(config_modul.get_comandline_arguments()["config"]))
