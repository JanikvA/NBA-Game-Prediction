import math
import os
import random
import sqlite3
from typing import Any, Dict, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import uncertainties
from aquarel import load_theme
from loguru import logger

from nba_game_prediction import config_modul


def add_random_probs(data: pd.DataFrame) -> None:
    data["random_winprob"] = data.apply(lambda row: random.uniform(0, 1), axis=1)


def get_binominal_std_dev_on_prob(n, p):
    return math.sqrt(n * p * (1 - p)) / n


def pred_vs_actual_prob_closure(
    data: pd.DataFrame, prob_key: str, result_key: str, out_dir: str
) -> None:
    """Sorts games into bins based on the probability that the home team wins.
    Then calculates the ratio of the actual win rate for these
    games and the mean of the predictions.
    For a perfect algo this ratio should be close to 1.
    The errorbars are calculated using Poisson error and the standard propagation of uncertainty

    Args:
        train_data (pd.DataFrame): Data for the training
        prob_key (str): Which probability to be plotted, e.g. HOME_ELO_winprob,
        HOME_trueskill_winprob or random_winprob
        out_dir (str): The plots will be saved to this directory
    """
    bins = [0, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 1]
    x_data = []
    y_pred_data = []
    y_actual_data = []
    y_err = []
    ratio = []
    for n, lower_bound in enumerate(bins):
        if n + 1 == len(bins):
            break
        bin_data = data.loc[
            (data[prob_key] > lower_bound) & (data[prob_key] < bins[n + 1])
        ]
        bin_center = lower_bound + (bins[n + 1] - lower_bound) / 2
        x_data.append(bin_center)
        y_pred_data.append(bin_data[prob_key].mean())
        y_actual_data.append(bin_data[result_key].mean())
        # Using binominal distribution and propagation of uncertainty
        # to calculate the ratio and it's uncertainty
        p = bin_data[result_key].mean()
        N = bin_data[result_key].count()
        y_data_unc_obj = uncertainties.ufloat(p, get_binominal_std_dev_on_prob(N, p))
        ratio_unc_obj = y_data_unc_obj / y_pred_data[-1]
        y_err.append(ratio_unc_obj.std_dev)
        ratio.append(ratio_unc_obj.nominal_value)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    sns.histplot(
        data=data,
        x=prob_key,
        hue=result_key,
        bins=bins,
        multiple="stack",
        ax=axs[0],
    )
    axs[0].set_title(f"{prob_key} closure")
    axs[1].errorbar(
        x=x_data,
        y=ratio,
        yerr=y_err,
        fmt="o",
        color="black",
        ecolor="grey",
        elinewidth=3,
        capsize=5,
    )
    axs[1].axhline(1, color="red", linestyle="--")
    axs[1].set_ylabel("Ratio of win probabilities (prediction/actual)")
    axs[1].set_xlabel("HOME win probability")
    out_file_name = os.path.join(out_dir, "probability_comparison_" + prob_key + ".png")
    logger.info(f"Saving plot to: {out_file_name}")
    fig.savefig(out_file_name)


def feature_correlation(
    train_data: pd.DataFrame, cols_to_plot: List[str], method: str, out_dir: str
) -> None:
    """Creates heatmap of correlation for input features

    Args:
        train_data (pd.DataFrame): trainings data
        cols_to_plot (List[str]): Which features to plot.
        method (str): Which correlation method to use. (Choices: "pearson", "kendall", "spearman")
        out_dir (str): plots will be saved to the path of this directory
    """
    plot_data = train_data[cols_to_plot]
    fig, ax = plt.subplots(figsize=(20, 15))
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    mask = np.triu(np.ones_like(plot_data.corr()))[1:, :-1]
    sns.heatmap(
        plot_data.corr(method=method).iloc[1:, :-1],
        vmin=-1,
        vmax=1,
        annot=True,
        ax=ax,
        mask=mask,
        fmt=".2f",
        cmap=cmap,
    )
    out_file_name = os.path.join(out_dir, "feature_correlation_" + method + ".png")
    logger.info(f"Saving plot to: {out_file_name}")
    fig.savefig(out_file_name)


# TODO this is slow. make it faster
def feature_pair_plot(
    train_data: pd.DataFrame, cols_to_plot: List[str], out_dir: str
) -> None:
    """Seaborn pairplot of features. Correlation between
    features and separation power for HOME_WL can be observed

    Args:
        train_data (pd.DataFrame): trainings data
        cols_to_plot (List[str]): Which features to plot.
        out_dir (str): plots will be saved to the path of this directory
    """
    plot_data = train_data[["HOME_WL"] + cols_to_plot]

    df = pd.melt(plot_data, plot_data.columns[0], plot_data.columns[1:])
    g = sns.FacetGrid(
        df, col="variable", hue="HOME_WL", col_wrap=3, sharex=False, sharey=False
    )
    g.map(sns.kdeplot, "value", shade=True)
    out_file_name = os.path.join(out_dir, "feature_pair_plot.png")
    logger.info(f"Saving plot to: {out_file_name}")
    g.savefig(out_file_name)


def plot_team_skill(
    connection: sqlite3.Connection,
    algo: str,
    teams_to_plot: List[str],
    cut_n_games: int,
    out_dir: str,
) -> None:
    """Plot the values for the elo {algo} for a selection of
    teams_to_plot with respect to time

    Args:
        connection (sqlite3.Connection): Connection to SQL database
        algo (str): Which rating to plot, e.g. ELO or trueskill_mu
        teams_to_plot (List[str]): List of team names which should be plotted
        cut_n_games (int): marks the position that corresponds to the cut_n_games
        variable. This variable ensures that only games are
        used after the elo scores have settled in.
        out_dir (str): plots will be saved to the path of this directory
    """
    data: pd.DataFrame = pd.DataFrame()
    for team_name in teams_to_plot:
        team_data = pd.read_sql(
            f"""
            SELECT GAME_DATE, HOME_{algo}, AWAY_{algo}, HOME_TEAM_NAME, AWAY_TEAM_NAME
            FROM train_data
            WHERE HOME_TEAM_NAME='{team_name}' OR AWAY_TEAM_NAME='{team_name}'""",
            connection,
        )
        team_data["team_name"] = team_name
        team_data[algo] = team_data.apply(
            lambda row: row[f"HOME_{algo}"]
            if row["HOME_TEAM_NAME"] == team_name
            else row[f"AWAY_{algo}"],
            axis=1,
        )
        if data.empty:
            data = team_data.loc[:, ["GAME_DATE", "team_name", algo]].copy()
        else:
            # TODO check what .copy does here
            data = pd.concat(
                [data, team_data.loc[:, ["GAME_DATE", "team_name", algo]].copy()],
                ignore_index=True,
            )
    data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data, x="GAME_DATE", y=algo, hue="team_name", ax=ax)
    myFmt = mdates.DateFormatter("%Y")
    ax.xaxis.set_major_formatter(myFmt)
    ax.axvline(x=data.iloc[cut_n_games]["GAME_DATE"], color="red")

    out_file_name = os.path.join(out_dir, f"team_{algo}_plot.png")
    logger.info(f"Saving plot to: {out_file_name}")
    fig.savefig(out_file_name)


def plot_league_skill_distribution(
    connection: sqlite3.Connection, algo: str, cut_n_games: int, out_dir: str
) -> None:
    """Plots the skill distribution for a given algo
    and each season in the trianings data. The median is
    indicated by a vertical red line.

    Args:
        connection (sqlite3.Connection): Connection to SQL database
        algo (str): Which rating to plot, e.g. ELO or trueskill_mu
        cut_n_games (int): ignores the first {cut_n_games} games in the data of
        out_dir (str): plots will be saved to the path of this directory
    """
    team_names = pd.read_sql(
        "SELECT DISTINCT HOME_TEAM_NAME FROM train_data", connection
    )
    seasons = pd.read_sql("SELECT DISTINCT SEASON FROM train_data", connection)[
        "SEASON"
    ].tolist()
    season_dict: Dict[str, List[Any]] = {"season": [], f"{algo}": []}
    for season in seasons:
        for team_name in team_names["HOME_TEAM_NAME"]:
            team_data = pd.read_sql(
                f"""
                SELECT SEASON, GAME_DATE, HOME_{algo}, AWAY_{algo}, HOME_TEAM_NAME, AWAY_TEAM_NAME
                FROM train_data
                WHERE HOME_TEAM_NAME='{team_name}' OR AWAY_TEAM_NAME='{team_name}'""",
                connection,
            )
            team_data = team_data[cut_n_games:]
            team_data["team_name"] = team_name
            team_data[algo] = team_data.apply(
                lambda row: row[f"HOME_{algo}"]
                if row["HOME_TEAM_NAME"] == team_name
                else row[f"AWAY_{algo}"],
                axis=1,
            )
            team_data["GAME_DATE"] = pd.to_datetime(team_data["GAME_DATE"])
            team_data.set_index("GAME_DATE")
            if team_data.loc[team_data["SEASON"] == season].empty:
                logger.warning(f"{team_name} has no games in the {season} season!")
            else:
                season_dict[f"{algo}"].append(
                    team_data.loc[team_data["SEASON"] == season].iloc[-1][algo]
                )
                season_dict["season"].append(season)
    data = pd.DataFrame.from_dict(season_dict)

    # Taken from https://seaborn.pydata.org/examples/kde_ridgeplot.html
    g = sns.FacetGrid(data, row="season", hue="season")
    g.map(sns.kdeplot, algo, fill=True)
    # TODO why does this work?
    # TODO add legend
    g.map(lambda x, **kw: plt.axvline(x.median(), color="red"), algo)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(
            0,
            0.2,
            label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    g.map(label, algo)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(left=True)

    out_file_name = os.path.join(out_dir, f"season_{algo}_plot.png")
    logger.info(f"Saving plot to: {out_file_name}")
    g.savefig(out_file_name)


def main(config: Dict[str, Any]) -> None:
    """Creates plots from the training data

    Args:
        config (Dict[str, Any]): config
    """
    theme = load_theme("arctic_dark").set_overrides({"font.family": "monospace"})
    theme.apply()
    connection = sqlite3.connect(config["sql_db_path"])
    train_data = pd.read_sql("SELECT * from train_data", connection)
    len_all_games = len(train_data)
    train_data = train_data.dropna()
    train_data["AWAY_is_back_to_back"] = pd.to_numeric(
        train_data["AWAY_is_back_to_back"]
    )
    logger.info(f"Dropped {len_all_games-len(train_data)} games because of NaNs")

    for algo in ["ELO", "trueskill_mu", "FTE_ELO"]:
        if "HOME_" + algo not in train_data.columns:
            logger.warning(
                f"""HOME_{algo} is not in the trainings data. Will not
                run plot_team_skill or plot_league_skill_distribution.
                To get this variable add it to
                the config[create_trainings_data][feature_list]"""
            )
            continue
        plot_team_skill(
            connection,
            algo,
            config["plot_train_data"]["teams_to_plot"],
            int(config["plot_train_data"]["cut_n_games"] / 15),  # dividing by 15
            # because for 15 games each team has played 1 game on average.
            config["output_dir"],
        )
        plot_league_skill_distribution(
            connection,
            algo,
            int(config["plot_train_data"]["cut_n_games"] / 15),
            config["output_dir"],
        )

    add_random_probs(train_data)

    for prob in [
        "HOME_ELO_winprob",
        "HOME_trueskill_winprob",
        "HOME_FTE_ELO_winprob",
        "random_winprob",
    ]:
        pred_vs_actual_prob_closure(
            train_data[config["plot_train_data"]["cut_n_games"] :],
            prob,
            "HOME_WL",
            config["output_dir"],
        )

    for method in ["pearson", "kendall", "spearman"]:
        feature_correlation(
            train_data,
            config["plot_train_data"]["correlation_features"],
            method,
            config["output_dir"],
        )

    feature_pair_plot(
        train_data,
        config["plot_train_data"]["pair_plot_features"],
        config["output_dir"],
    )
    connection.close()


if __name__ == "__main__":
    main(config_modul.load_config(config_modul.get_comandline_arguments()["config"]))
