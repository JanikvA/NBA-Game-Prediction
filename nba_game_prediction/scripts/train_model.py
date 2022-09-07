import os
import sqlite3
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from sklearn import metrics, model_selection, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from nba_game_prediction import config_modul


def main(config: Dict[str, Any]) -> None:
    connection = sqlite3.connect(config["sql_db_path"])
    data = pd.read_sql("SELECT * from train_data", connection)
    connection.close()
    data = data.dropna(how="any")
    y = data["HOME_WL"]
    data["ELO_difference"] = data.apply(
        lambda row: row["HOME_ELO"] - row["AWAY_ELO"], axis=1
    )
    data["trueskill_mu_difference"] = data.apply(
        lambda row: row["HOME_trueskill_mu"] - row["AWAY_trueskill_mu"], axis=1
    )
    data["trueskill_sigma_sum"] = data.apply(
        lambda row: row["HOME_trueskill_sigma"] + row["AWAY_trueskill_sigma"], axis=1
    )
    logger.info(f"Home team win rate: {y.mean()}")

    higher_elo_wr = data[
        ((data["ELO_difference"] > 0) & (data["HOME_WL"] == 1))
        | ((data["ELO_difference"] < 0) & (data["HOME_WL"] == 0))
    ]["HOME_WL"].count() / len(data)
    logger.info(f"Team with higher ELO win rate: {higher_elo_wr}")
    higher_trueskill_wr = data[
        ((data["trueskill_mu_difference"] > 0) & (data["HOME_WL"] == 1))
        | ((data["trueskill_mu_difference"] < 0) & (data["HOME_WL"] == 0))
    ]["HOME_WL"].count() / len(data)
    logger.info(f"Team with higher trueskill_mu win rate: {higher_trueskill_wr}")
    x = data.drop(["HOME_WL"], axis=1)
    x = x.loc[
        :,
        [
            "ELO_difference",
            "trueskill_mu_difference",
            "trueskill_sigma_sum",
            "HOME_is_back_to_back",
            "AWAY_is_back_to_back",
        ],
    ]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2
    )
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    classifiers = [
        RandomForestClassifier(max_depth=3),
        RandomForestClassifier(max_depth=5),
        RandomForestClassifier(max_depth=10),
        MLPClassifier(alpha=1, max_iter=200, hidden_layer_sizes=(50,)),
    ]

    name = [
        "RandomForestClassifier(max_depth=3)",
        "RandomForestClassifier(max_depth=5)",
        "RandomForestClassifier(max_depth=10)",
        "MLPClassifier(alpha=1, max_iter=1000)",
    ]
    for n, clf in enumerate(classifiers):
        clf.fit(x_train, y_train)
        prediction = clf.predict(x_test)
        prediction_prob = clf.predict_proba(x_test)
        logger.info(prediction_prob.shape)
        train_prediction = clf.predict(x_train)
        train_acc = metrics.accuracy_score(y_train, train_prediction)
        test_acc = metrics.accuracy_score(y_test, prediction)
        logger.info(f"{name[n]}: {test_acc} ({train_acc})")
        metrics.ConfusionMatrixDisplay.from_predictions(y_test, prediction)
        out_file_name = os.path.join(
            config["output_dir"], name[n] + "_confusion_matrix.png"
        )
        logger.info(f"Saving plot to: {out_file_name}")
        plt.savefig(out_file_name)
        plt.clf()


if __name__ == "__main__":
    main(config_modul.load_config(config_modul.get_comandline_arguments()["config"]))
