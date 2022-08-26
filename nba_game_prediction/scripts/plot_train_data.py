import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_data():
    return pd.read_csv("train_data.csv")


def main():
    train_data = read_data()
    # plt.matshow(train_data.corr())
    # print(train_data.mean())
    # train_data=train_data.loc[:,["HOME_WL", "HOME_is_back_to_back", "AWAY_is_back_to_back",
    # "HOME_WL_10G", "AWAY_WL_10G", "HOME_SEASON_ID_10G",
    # "HOME_MMR", "AWAY_MMR", "HOME_PLUS_MINUS"]]
    # train_data=train_data.loc[:,["HOME_WL",  "HOME_WL_10G",
    # "AWAY_WL_10G", "HOME_MMR", "AWAY_MMR", "HOME_PLUS_MINUS"]]
    train_data["MMR_difference"] = train_data.apply(
        lambda row: row["HOME_MMR"] - row["AWAY_MMR"], axis=1
    )
    # print(train_data.columns)
    # print(train_data.mean(numeric_only=True))
    # sns.heatmap(train_data.corr(), vmin=-1, vmax=1, annot=True)
    sns.scatterplot(
        data=train_data, x="MMR_difference", y="HOME_PLUS_MINUS", hue="HOME_WL"
    )
    # sns.scatterplot(data=train_data, x="HOME_MMR", y="AWAY_MMR", hue="HOME_WL")
    # sns.kdeplot( data=train_data, x="HOME_MMR", y="AWAY_MMR", hue="HOME_WL")
    # sns.histplot(data=train_data,
    # x="AWAY_is_back_to_back", hue="HOME_WL",
    # common_norm=False, common_bins=True, stat="probability")
    # sns.pairplot(data=train_data)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
