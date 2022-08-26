import matplotlib.pyplot as plt
import pandas as pd

from nba_game_prediction import config_modul


def read_data(train_data_path):
    return pd.read_csv(train_data_path)


def main(config):
    train_data = read_data(config["train_data_path"])
    # plt.matshow(train_data.corr())
    train_data["ELO_difference"] = train_data.apply(
        lambda row: row["HOME_ELO"] - row["AWAY_ELO"], axis=1
    )
    print(train_data.keys())
    exit(1)
    # sns.scatterplot(
    #     data=train_data, x="ELO_difference", y="HOME_PLUS_MINUS_10G", hue="HOME_WL"
    # )
    # sns.heatmap(train_data.corr(), vmin=-1, vmax=1, annot=True)
    # sns.heatmap(train_data.corr(), vmin=-1, vmax=1, annot=False)
    # sns.scatterplot(data=train_data, x="HOME_ELO", y="AWAY_ELO", hue="HOME_WL")
    # sns.kdeplot( data=train_data, x="HOME_ELO", y="AWAY_ELO", hue="HOME_WL")
    # sns.pairplot(data=train_data)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(config_modul.load_config(config_modul.get_comandline_arguments()["config"]))
