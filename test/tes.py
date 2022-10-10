from sklearn.ensemble import AdaBoostClassifier
from mlp import clf as clf_
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression


def test(clf):
    if clf:
        X, y = make_classification()
    else:
        X, y = make_regression()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    base = clf_(hidden_layer_sizes=(100,))
    model = AdaBoostClassifier(base_estimator=base,
                               n_estimators=100)

    model.fit(x_train, y_train)

    return model.score(x_test, y_test)


if __name__ == "__main__":
    print(test(True))
