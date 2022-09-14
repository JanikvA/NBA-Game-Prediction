import os
import sqlite3
from typing import Any, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from aquarel import load_theme
from labellines import labelLine
from loguru import logger
from scipy.stats.distributions import chi2
from sklearn import metrics, model_selection, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

from nba_game_prediction import config_modul
from nba_game_prediction.scripts.plot_train_data import (
    get_binominal_std_dev_on_prob,
    pred_vs_actual_prob_closure,
)


def plot_accuracy_per_season(y_test, test_pred, seasons_series, name, out_dir):
    acc_dict = {"season": [], "accuracy": [], "uncertainty": []}
    for season in seasons_series.unique():
        this_season_indices = seasons_series[seasons_series == season].index
        this_season_indices = this_season_indices[
            this_season_indices.isin(y_test.index)
        ]
        test_pred_season = pd.Series(data=test_pred, index=y_test.index)
        test_pred_season = test_pred_season[this_season_indices]
        y_test_season = y_test[this_season_indices]
        acc = metrics.accuracy_score(y_test_season, test_pred_season)
        acc_unc = get_binominal_std_dev_on_prob(len(y_test_season), acc)
        acc_dict["season"].append(season)
        acc_dict["accuracy"].append(acc)
        acc_dict["uncertainty"].append(acc_unc)
    season_accuracies = pd.DataFrame(acc_dict)

    # chi2 code from: https://www.astroml.org/book_figures/chapter4/fig_chi2_eval.html
    # compute the mean and the chi^2/dof
    mean_acc = metrics.accuracy_score(y_test, test_pred)
    z = (season_accuracies["accuracy"] - mean_acc) / season_accuracies["uncertainty"]
    chi2_sum = np.sum(z**2)
    chi2dof = chi2_sum / (len(season_accuracies) - 1)

    # compute the standard deviations of chi^2/dof
    sigma = np.sqrt(2.0 / (len(season_accuracies) - 1))
    nsig = (chi2dof - 1) / sigma

    p_value = chi2.sf(chi2_sum, len(season_accuracies) - 1)
    logger.info(f"{chi2_sum=:.2f}, {chi2dof=:.2f}, {sigma=:.2f}, {p_value=:.2f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        data=season_accuracies,
        x="season",
        y="accuracy",
        yerr="uncertainty",
        fmt="o",
        color="black",
        ecolor="grey",
        elinewidth=3,
        capsize=5,
    )
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Season")
    ax.set_ylabel("Accuracy")
    # labeLine doens't work wll with ax.axhline... so I have to do this
    x_beg, x_end = min(acc_dict["season"]) - 1, max(acc_dict["season"]) + 1
    x_middle = (x_beg + x_end) / 2
    mean_acc_line = ax.plot([x_beg, x_end], [mean_acc, mean_acc], color="red")
    labelLine(
        mean_acc_line[0],
        x=x_middle,
        label="Mean accuracy",
        color="red",
        fontweight="bold",
    )
    ax.set_xlim(x_beg, x_end)
    ax.text(
        0.02,
        0.02,
        r"$\hat{\mu} = %.2f$" % mean_acc,
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )
    ax.text(
        0.98,
        0.02,
        f"$\chi^2_{{\\rm dof}} = {chi2dof:.2f}\, ({nsig:.1f}\,\sigma), p = {p_value:.2f}$",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )
    # r'$\chi^2_{\rm dof} = %.2f\, (%.1f\,\sigma)$' % (chi2dof, nsig),
    out_file_name = os.path.join(out_dir, f"acc_per_season_{name}.png")
    logger.info(f"Saving plot to: {out_file_name}")
    fig.savefig(out_file_name)


def train_xgb(x_train, y_train, x_test, y_test):
    param_grid = {
        "learning_rate": np.linspace(0.1, 0.9, 3),
        "max_depth": list(range(4, 8, 2)),
        "reg_alpha": list(range(1, 2)),
        "reg_lambda": list(range(1, 2)),
        "n_estimators": [10, 100],
        "objective": ["binary:logistic"],
    }
    # TODO scalar transform with pipeline
    model = model_selection.GridSearchCV(
        xgb.XGBClassifier(), param_grid=param_grid, cv=10, verbose=True, n_jobs=-1
    )
    model.fit(x_train, y_train)
    cv_res = model.cv_results_
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_acc = metrics.accuracy_score(y_train, y_train_pred)
    test_acc = metrics.accuracy_score(y_test, y_test_pred)
    logger.info(f"xgboost accuracy: {test_acc} ({train_acc})")
    for n in range(len(cv_res["params"])):
        logger.debug("--------")
        logger.debug(cv_res["params"][n])
        logger.debug(
            f"{cv_res['mean_test_score'][n]:.3f} +- {cv_res['std_test_score'][n]:.3f}"
        )
        logger.debug("--------")
    logger.debug(f"Best parameters: {model.best_params_}")
    return model.best_estimator_


def plot_xgb(estimator, x_train, y_train, x_test, y_test, out_dir):
    test_prob = estimator.predict_proba(x_test)
    test_prob = test_prob[:, 1]
    dummy_df = pd.DataFrame.from_dict({"xgboost": test_prob, "HOME_WL": y_test})
    pred_vs_actual_prob_closure(dummy_df, "xgboost", "HOME_WL", out_dir)

    out_file_name = os.path.join(out_dir, "xgboost_importance.png")
    xgb.plot_importance(estimator)
    fig = plt.gcf()
    fig.savefig(out_file_name)
    plt.clf()

    # explain the model's predictions using SHAP
    explainer = shap.Explainer(estimator)
    shap_values = explainer(x_train)
    out_file_name = os.path.join(out_dir, "xgboost_shap_bar_summary.png")
    shap.summary_plot(shap_values, feature_names=x_train.columns, plot_type="bar")
    fig = plt.gcf()
    fig.savefig(out_file_name)
    plt.clf()


def get_data(sql_db_path, cut_n_games) -> pd.DataFrame:
    connection = sqlite3.connect(sql_db_path)
    data = pd.read_sql("SELECT * from train_data", connection)
    connection.close()
    # The ELO/trueskill ratings need some phase in the beginning to have meaningful values
    # which is why some of the first games of the training data are being removed here
    skip_first_n = cut_n_games
    if len(data) < skip_first_n:
        raise Exception(
            f"""Not enough games ({len(data)}) to skip {skip_first_n} games!
            Adjust cut_n_games under train_model in the config"""
        )
    logger.info(f"Dropping the first {skip_first_n} of the total {len(data)} games")
    data = data[skip_first_n:]
    n_left = len(data)
    data = data.dropna()
    logger.info(f"Dropping {n_left - len(data)} games because of NaNs")
    y = data["HOME_WL"]

    data["ELO_difference"] = data.apply(
        lambda row: row["HOME_ELO"] - row["AWAY_ELO"], axis=1
    )
    data["trueskill_mu_difference"] = data.apply(
        lambda row: row["HOME_trueskill_mu"] - row["AWAY_trueskill_mu"], axis=1
    )

    home_team_wr = y.mean()
    higher_elo_wr = data[
        ((data["ELO_difference"] >= 0) & (data["HOME_WL"] == 1))
        | ((data["ELO_difference"] < 0) & (data["HOME_WL"] == 0))
    ]["HOME_WL"].count() / len(data)
    higher_trueskill_wr = data[
        ((data["trueskill_mu_difference"] >= 0) & (data["HOME_WL"] == 1))
        | ((data["trueskill_mu_difference"] < 0) & (data["HOME_WL"] == 0))
    ]["HOME_WL"].count() / len(data)
    higher_FTE_elo_wr = data[
        ((data["HOME_FTE_ELO_winprob"] >= 0.5) & (data["HOME_WL"] == 1))
        | ((data["HOME_FTE_ELO_winprob"] < 0.5) & (data["HOME_WL"] == 0))
    ]["HOME_WL"].count() / len(data)
    win_rates = {
        "Win rate for the Home team": home_team_wr,
        "Win rate for team with higher ELO": higher_elo_wr,
        "Win rate for team with higher trueskill_mu": higher_trueskill_wr,
        "Win rate for team with higher FiveThirtyEight ELO": higher_FTE_elo_wr,
    }
    for description, wr in win_rates.items():
        logger.info(
            f"{description}: {wr:.1%} +- {get_binominal_std_dev_on_prob(len(data), wr):.1%}"
        )

    return data


def get_train_test(data: pd.DataFrame, feature_list) -> Dict[str, Any]:
    x = data.drop(["HOME_WL"], axis=1)
    y = data["HOME_WL"]
    x = x[feature_list]
    y = y[x.index]
    logger.info(f"Features used in the models: {', '.join(x.columns)}")
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2
    )
    logger.info(
        f"Using {len(y_train)} games for training and {len(y_test)} games for testing."
    )
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


def fit_and_plot_clf(
    name: str, clf, x_train, y_train, x_test, y_test, seasons, out_dir
):
    clf_transformed = make_pipeline(preprocessing.StandardScaler(), clf)
    clf_transformed.fit(x_train, y_train)
    vs = model_selection.cross_val_score(clf_transformed, x_train, y_train, cv=10)
    train_prediction = clf_transformed.predict(x_train)
    test_prediction = clf_transformed.predict(x_test)
    test_acc = metrics.accuracy_score(y_test, test_prediction)
    train_acc = metrics.accuracy_score(y_train, train_prediction)
    logger.info(
        f"{name} : {test_acc=:.1%} ({vs.mean()=:.1%} +- {vs.std():.1%}) | {train_acc=:.1%}"
    )
    test_prob = clf_transformed.predict_proba(x_test)
    test_prob = test_prob[:, 1]
    # TODO plot this for both test and trian data
    dummy_df = pd.DataFrame.from_dict({name: test_prob, "HOME_WL": y_test})
    pred_vs_actual_prob_closure(dummy_df, name, "HOME_WL", out_dir)
    plot_accuracy_per_season(y_test, test_prediction, seasons, name, out_dir)


def main(config: Dict[str, Any]) -> None:
    mpl.use("agg")
    theme = load_theme("arctic_dark").set_overrides({"font.family": "monospace"})
    theme.apply()
    data = get_data(
        config["sql_db_path"],
        config["train_model"]["cut_n_games"],
    )
    seasons = data["SEASON"]
    train_test_data = get_train_test(
        data,
        config["train_model"]["feature_list"],
    )

    classifier_dict = {
        "LinearRegression": LogisticRegression(),
        "KNeighborsClassifier(n_neighbors=40)": KNeighborsClassifier(n_neighbors=40),
        "RandomForestClassifier(max_depth=3)": RandomForestClassifier(max_depth=3),
        "MLPClassifier(alpha=1, max_iter=200, hidden_layer_sizes=(50,))": MLPClassifier(
            alpha=1, max_iter=200, hidden_layer_sizes=(50,)
        ),
    }
    for name, clf in classifier_dict.items():
        fit_and_plot_clf(
            name, clf, **train_test_data, seasons=seasons, out_dir=config["output_dir"]
        )

    estimator = train_xgb(**train_test_data)
    plot_xgb(estimator, **train_test_data, out_dir=config["output_dir"])
    plot_accuracy_per_season(
        train_test_data["y_test"],
        estimator.predict(train_test_data["x_test"]),
        seasons,
        "xgboost",
        config["output_dir"],
    )


if __name__ == "__main__":
    main(config_modul.load_config(config_modul.get_comandline_arguments()["config"]))
