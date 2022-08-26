import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier


def read_data():
    return pd.read_csv("train_data.csv")


data = read_data()
data = data.dropna(how="any")
y = data["HOME_WL"]
data["MMR_difference"] = data.apply(
    lambda row: row["HOME_MMR"] - row["AWAY_MMR"], axis=1
)
print(f"Home team win rate: {y.mean()}")
higher_mmr_wr = data[
    ((data["MMR_difference"] > 0) & (data["HOME_WL"] == 1))
    | ((data["MMR_difference"] < 0) & (data["HOME_WL"] == 0))
]["HOME_WL"].count() / len(data)
print(f"Team with higher MMR win rate: {higher_mmr_wr}")
x = data.drop(["HOME_WL", "HOME_PLUS_MINUS", "AWAY_PLUS_MINUS"], axis=1)
x = x.loc[:, ["MMR_difference"]]
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.2
)
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

classifiers = [
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025),
    # SVC(gamma=2, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # MLPClassifier(alpha=1, max_iter=1000, hidden_layer_sizes=(100,)),
    # MLPClassifier(alpha=1, max_iter=1000, hidden_layer_sizes=(10,)),
    MLPClassifier(alpha=1, max_iter=200, hidden_layer_sizes=(50,)),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis()
]

name = [
    # "KNeighborsClassifier(3)",
    # "SVC(kernel='linear', C=0.025)",
    # "SVC(gamma=2, C=1)",
    # "GaussianProcessClassifier(1.0 * RBF(1.0))",
    # "DecisionTreeClassifier(max_depth=5)",
    # "RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)",
    "MLPClassifier(alpha=1, max_iter=1000)",
    # "AdaBoostClassifier()",
    # "GaussianNB()",
    # "QuadraticDiscriminantAnalysis()"
]
for n, clf in enumerate(classifiers):
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    prediction_prob = clf.predict_proba(x_test)
    print(prediction_prob.shape)
    train_prediction = clf.predict(x_train)
    train_acc = sklearn.metrics.accuracy_score(y_train, train_prediction)
    test_acc = sklearn.metrics.accuracy_score(y_test, prediction)
    print(f"{name[n]}: {test_acc} ({train_acc})")

    # sns.histplot(x=prediction_prob[:,1],hue=y_test, multiple="stack")
    # sns.histplot(x=prediction_prob[:,1],hue=y_test)
    # sns.displot(x=prediction_prob[:,1],hue=y_test, kind="kde", multiple="fill")
    # sns.scatterplot(x=prediction_prob[:,1],y=x_test[:,-1])
    plt.show()
