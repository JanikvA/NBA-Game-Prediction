import math
import os
import random
import sqlite3
from typing import Any, Dict, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from aquarel import load_theme
from loguru import logger

from nba_game_prediction import config_modul


def add_random_probs(data: pd.DataFrame) -> None:
    data["random_winprob"] = data.apply(lambda row: random.uniform(0, 1), axis=1)


def pred_vs_actual_prob_comparison(
    train_data: pd.DataFrame, prob_key: str, out_dir: str
) -> None:
    """Sorts games into bins based on the probability that the home team wins.
    Then calculates the ratio of the actual win rate for these games and the bin center.
    For a perfect algo this ratio should be close to 1.
    The errorbars are calculated using Poisson error and the standard propagation of uncertainty

    Args:
        train_data (pd.DataFrame): Data for the training
        prob_key (str): Which probability to be plotted, e.g. HOME_ELO_winprob,
        HOME_trueskill_winprob or random_winprob
        out_dir (str): The plots will be saved to this directory
    """
    bins = [0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1]
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
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x=x_data, y=ratio, yerr=y_err, color="black")
    out_file_name = os.path.join(out_dir, "probability_comparison_" + prob_key + ".png")
    logger.info(f"Saving plot to: {out_file_name}")
    fig.savefig(out_file_name)


def feature_correlation(train_data: pd.DataFrame, method: str, out_dir: str) -> None:
    """Creates heatmap of correlation for input features

    Args:
        train_data (pd.DataFrame): trainings data
        method (str): Which correlation method to use. (Choices: "pearson", "kendall", "spearman")
        out_dir (str): plots will be saved to the path of this directory
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
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(plot_data.corr(method=method), vmin=-1, vmax=1, annot=True, ax=ax)
    out_file_name = os.path.join(out_dir, "feature_correlation_" + method + ".png")
    logger.info(f"Saving plot to: {out_file_name}")
    fig.savefig(out_file_name)


# TODO this is slow. make it faster
def feature_pair_plot(train_data: pd.DataFrame, out_dir: str) -> None:
    """Seaborn pairplot of features. Correlation between
    features and separation power for HOME_WL can be observed

    Args:
        train_data (pd.DataFrame): trainings data
        out_dir (str): plots will be saved to the path of this directory
    """
    plot_data = train_data.drop(
        [
            "GAME_ID",
            "SEASON_ID",
            "AWAY_TEAM_NAME",
            "HOME_TEAM_NAME",
            "GAME_DATE",
            "HOME_is_back_to_back",
            "AWAY_is_back_to_back",
            "is_Playoffs",
        ],
        axis=1,
    )
    pair_grid = sns.pairplot(
        data=plot_data,
        hue="HOME_WL",
        diag_kws={"common_norm": False},
        height=3,
        aspect=1,
    )
    out_file_name = os.path.join(out_dir, "feature_pair_plot.png")
    logger.info(f"Saving plot to: {out_file_name}")
    pair_grid.savefig(out_file_name)


def plot_team_skill(
    connection: sqlite3.Connection, algo: str, teams_to_plot: List[str], out_dir: str
) -> None:
    """_summary_

    Args:
        connection (sqlite3.Connection): Connection to SQL database
        algo (str): Which rating to plot, e.g. ELO or trueskill_mu
        teams_to_plot (List[str]): List of team names which should be plotted
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
            data = pd.concat(
                [data, team_data.loc[:, ["GAME_DATE", "team_name", algo]]],
                ignore_index=True,
            )
    data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data, x="GAME_DATE", y=algo, hue="team_name", ax=ax)
    myFmt = mdates.DateFormatter("%Y")
    ax.xaxis.set_major_formatter(myFmt)
    out_file_name = os.path.join(out_dir, f"team_{algo}_plot.png")
    logger.info(f"Saving plot to: {out_file_name}")
    fig.savefig(out_file_name)


def plot_league_skill_distribution(
    connection: sqlite3.Connection, algo: str, out_dir: str
) -> None:
    """Plots the skill distribution for a given algo
    and each season in the trianings data. The median is
    indicated by a vertical red line.

    Args:
        connection (sqlite3.Connection): Connection to SQL database
        algo (str): Which rating to plot, e.g. ELO or trueskill_mu
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

    for algo in ["ELO", "trueskill_mu"]:
        plot_team_skill(
            connection,
            algo,
            config["plot_train_data"]["teams_to_plot"],
            config["output_dir"],
        )
        plot_league_skill_distribution(connection, algo, config["output_dir"])

    add_random_probs(train_data)
    for prob in ["HOME_ELO_winprob", "HOME_trueskill_winprob", "random_winprob"]:
        pred_vs_actual_prob_comparison(train_data, prob, config["output_dir"])

    for method in ["pearson", "kendall", "spearman"]:
        feature_correlation(train_data, method, config["output_dir"])
    feature_pair_plot(train_data, config["output_dir"])
    connection.close()


if __name__ == "__main__":
    main(config_modul.load_config(config_modul.get_comandline_arguments()["config"]))
