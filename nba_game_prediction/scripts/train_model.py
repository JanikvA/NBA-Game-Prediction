import os
import sqlite3
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb
from loguru import logger
from sklearn import metrics, model_selection, preprocessing

from nba_game_prediction import config_modul


def main(config: Dict[str, Any]) -> None:
    connection = sqlite3.connect(config["sql_db_path"])
    data = pd.read_sql("SELECT * from train_data", connection)
    connection.close()
    # data = data.dropna(how="any")
    # The ELO/trueskill ratings need some phase in the beginning to have meaningful values
    # which is why some of the first games of the training data are being removed here
    skip_first_n = config["train_model"]["cut_n_games"]
    if len(data) < skip_first_n:
        raise Exception(
            f"""Not enough games ({len(data)}) to skip {skip_first_n} games!
            Adjust cut_n_games under train_model in the {config['config_path']}"""
        )
    logger.info(f"Dropping the first {skip_first_n} of the total {len(data)} games")
    data = data[skip_first_n:]
    y = data["HOME_WL"]

    data["ELO_difference"] = data.apply(
        lambda row: row["HOME_ELO"] - row["AWAY_ELO"], axis=1
    )
    data["trueskill_mu_difference"] = data.apply(
        lambda row: row["HOME_trueskill_mu"] - row["AWAY_trueskill_mu"], axis=1
    )
    # data["trueskill_sigma_sum"] = data.apply(
    #     lambda row: row["HOME_trueskill_sigma"] + row["AWAY_trueskill_sigma"], axis=1
    # )
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
    x = x[config["train_model"]["feature_list"]]
    logger.info(f"Features used in the models: {', '.join(x.columns)}")
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2
    )
    logger.info(
        f"Using {len(y_train)} games for training and {len(y_test)} games for testing."
    )
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # classifiers = [
    #     RandomForestClassifier(max_depth=3),
    #     RandomForestClassifier(max_depth=5),
    #     RandomForestClassifier(max_depth=10),
    #     MLPClassifier(alpha=1, max_iter=200, hidden_layer_sizes=(50,)),
    # ]
    # name = [
    #     "RandomForestClassifier(max_depth=3)",
    #     "RandomForestClassifier(max_depth=5)",
    #     "RandomForestClassifier(max_depth=10)",
    #     "MLPClassifier(alpha=1, max_iter=1000)",
    # ]
    # for n, clf in enumerate(classifiers):
    #     clf.fit(x_train, y_train)
    #     prediction = clf.predict(x_test)
    #     train_prediction = clf.predict(x_train)
    #     train_acc = metrics.accuracy_score(y_train, train_prediction)
    #     test_acc = metrics.accuracy_score(y_test, prediction)
    #     logger.info(f"{name[n]}: {test_acc} ({train_acc})")
    #     metrics.ConfusionMatrixDisplay.from_predictions(y_test, prediction)
    #     out_file_name = os.path.join(
    #         config["output_dir"], name[n] + "_confusion_matrix.png"
    #     )
    #     logger.info(f"Saving plot to: {out_file_name}")
    #     plt.savefig(out_file_name)
    #     plt.clf()

    # train an XGBoost model
    # X, y = shap.datasets.boston()
    param = {"max_depth": 2, "eta": 0.3, "objective": "binary:logistic"}
    model = xgb.XGBClassifier(**param).fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_acc = metrics.accuracy_score(y_train, y_train_pred)
    test_acc = metrics.accuracy_score(y_test, y_test_pred)
    logger.info(f"xgboost accuracy: {test_acc} ({train_acc})")

    out_file_name = os.path.join(config["output_dir"], "xgboost_importance.png")
    xgb.plot_importance(model)
    fig = plt.gcf()
    fig.savefig(out_file_name)
    plt.clf()

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    explainer = shap.Explainer(model)
    shap_values = explainer(x_train)

    # visualize the first prediction's explanation
    out_file_name = os.path.join(config["output_dir"], "xgboost_shap_bar.png")
    # fig = shap.plots.waterfall(shap_values[0], show=False)
    shap.plots.bar(shap_values, show=False)
    fig = plt.gcf()
    fig.savefig(out_file_name)
    plt.clf()

    out_file_name = os.path.join(config["output_dir"], "xgboost_shap_dot_summary.png")
    shap.summary_plot(shap_values, feature_names=x.columns, plot_type="dot")
    fig = plt.gcf()
    fig.savefig(out_file_name)
    plt.clf()

    out_file_name = os.path.join(config["output_dir"], "xgboost_shap_bar_summary.png")
    shap.summary_plot(shap_values, feature_names=x.columns, plot_type="bar")
    fig = plt.gcf()
    fig.savefig(out_file_name)
    plt.clf()

    out_file_name = os.path.join(
        config["output_dir"], "xgboost_shap_violin_summary.png"
    )
    shap.summary_plot(shap_values, feature_names=x.columns, plot_type="violin")
    fig = plt.gcf()
    fig.savefig(out_file_name)
    plt.clf()


if __name__ == "__main__":
    main(config_modul.load_config(config_modul.get_comandline_arguments()["config"]))
