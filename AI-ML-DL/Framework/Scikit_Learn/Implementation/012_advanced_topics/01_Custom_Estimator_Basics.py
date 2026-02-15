"""
Scikit-learn custom estimator basics: BaseEstimator, ClassifierMixin, RegressorMixin, check_is_fitted
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


class SimpleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.mean_ = np.mean(X[y == self.classes_[0]], axis=0)
        return self

    def predict(self, X):
        check_is_fitted(self, ["mean_"])
        dist = np.linalg.norm(X - self.mean_, axis=1)
        return np.where(dist < np.median(dist), self.classes_[0], self.classes_[1])


class SimpleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_"])
        return X @ self.coef_


def main():
    print("=" * 60)
    print("Custom Estimator Basics: BaseEstimator, Mixins, check_is_fitted")
    print("=" * 60)

    print("\n[1] SimpleClassifier with ClassifierMixin:")
    X, y = load_iris(return_X_y=True)
    X_bin = X[y != 2]
    y_bin = y[y != 2]
    X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, random_state=42)
    clf = SimpleClassifier()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(f"    Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(f"    classes_: {clf.classes_}")

    print("\n[2] SimpleRegressor with RegressorMixin:")
    X_r, y_r = make_regression(n_samples=100, n_features=5, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, random_state=42)
    reg = SimpleRegressor()
    reg.fit(X_tr, y_tr)
    pred_r = reg.predict(X_te)
    print(f"    RMSE: {np.sqrt(mean_squared_error(y_te, pred_r)):.4f}")

    print("\n[3] check_is_fitted prevents predict before fit:")
    clf2 = SimpleClassifier()
    try:
        clf2.predict(X_test)
    except Exception as e:
        print(f"    Expected error: {type(e).__name__}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
