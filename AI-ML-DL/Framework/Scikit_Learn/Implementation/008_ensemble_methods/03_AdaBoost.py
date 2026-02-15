"""
Scikit-learn AdaBoostClassifier and AdaBoostRegressor: n_estimators, learning_rate
"""

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


def main():
    print("=" * 60)
    print("AdaBoostClassifier and AdaBoostRegressor")
    print("=" * 60)

    X_clf, y_clf = load_iris(return_X_y=True)
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )

    print("\n[1] AdaBoostClassifier - n_estimators:")
    for n in [10, 50, 100]:
        ada = AdaBoostClassifier(n_estimators=n, random_state=42)
        ada.fit(X_clf_train, y_clf_train)
        acc = accuracy_score(y_clf_test, ada.predict(X_clf_test))
        print(f"    n_estimators={n}: Accuracy = {acc:.4f}")

    print("\n[2] AdaBoostClassifier - learning_rate:")
    for lr in [0.5, 1.0, 2.0]:
        ada = AdaBoostClassifier(n_estimators=50, learning_rate=lr, random_state=42)
        ada.fit(X_clf_train, y_clf_train)
        acc = accuracy_score(y_clf_test, ada.predict(X_clf_test))
        print(f"    learning_rate={lr}: Accuracy = {acc:.4f}")

    print("\n[3] AdaBoostClassifier with custom base estimator:")
    base = DecisionTreeClassifier(max_depth=1)
    ada = AdaBoostClassifier(estimator=base, n_estimators=50, random_state=42)
    ada.fit(X_clf_train, y_clf_train)
    acc = accuracy_score(y_clf_test, ada.predict(X_clf_test))
    print(f"    DecisionTreeClassifier(max_depth=1): Accuracy = {acc:.4f}")

    X_reg, y_reg = make_regression(n_samples=200, n_features=8, noise=12, random_state=42)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    print("\n[4] AdaBoostRegressor:")
    ada_reg = AdaBoostRegressor(n_estimators=50, random_state=42)
    ada_reg.fit(X_reg_train, y_reg_train)
    r2 = r2_score(y_reg_test, ada_reg.predict(X_reg_test))
    print(f"    n_estimators=50: R2 = {r2:.4f}")

    print("\n[5] estimator_weights_ and estimator_errors_:")
    ada = AdaBoostClassifier(n_estimators=5, random_state=42)
    ada.fit(X_clf_train, y_clf_train)
    print(f"    Weights (first 5): {ada.estimator_weights_}")
    print(f"    Errors (first 5): {ada.estimator_errors_}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()