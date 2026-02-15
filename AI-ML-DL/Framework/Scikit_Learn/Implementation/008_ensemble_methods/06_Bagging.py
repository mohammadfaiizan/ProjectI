"""
Scikit-learn BaggingClassifier and BaggingRegressor: max_samples, max_features, bootstrap
"""

import numpy as np
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


def main():
    print("=" * 60)
    print("BaggingClassifier and BaggingRegressor")
    print("=" * 60)

    X_clf, y_clf = load_iris(return_X_y=True)
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )

    print("\n[1] BaggingClassifier - max_samples:")
    for ms in [0.5, 0.8, 1.0]:
        bag = BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=50,
            max_samples=ms,
            random_state=42
        )
        bag.fit(X_clf_train, y_clf_train)
        acc = accuracy_score(y_clf_test, bag.predict(X_clf_test))
        print(f"    max_samples={ms}: Accuracy = {acc:.4f}")

    print("\n[2] BaggingClassifier - max_features:")
    for mf in [0.5, 0.8, 1.0]:
        bag = BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=50,
            max_features=mf,
            random_state=42
        )
        bag.fit(X_clf_train, y_clf_train)
        acc = accuracy_score(y_clf_test, bag.predict(X_clf_test))
        print(f"    max_features={mf}: Accuracy = {acc:.4f}")

    print("\n[3] BaggingClassifier - bootstrap vs bootstrap_features:")
    bag_both = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50,
        bootstrap=True,
        bootstrap_features=True,
        random_state=42
    )
    bag_both.fit(X_clf_train, y_clf_train)
    acc = accuracy_score(y_clf_test, bag_both.predict(X_clf_test))
    print(f"    bootstrap=True, bootstrap_features=True: Accuracy = {acc:.4f}")

    X_reg, y_reg = make_regression(n_samples=300, n_features=10, noise=15, random_state=42)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    print("\n[4] BaggingRegressor:")
    bag_reg = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        n_estimators=50,
        max_samples=0.8,
        random_state=42
    )
    bag_reg.fit(X_reg_train, y_reg_train)
    r2 = r2_score(y_reg_test, bag_reg.predict(X_reg_test))
    print(f"    n_estimators=50, max_samples=0.8: R2 = {r2:.4f}")

    print("\n[5] estimators_ - access base estimators:")
    bag = BaggingClassifier(n_estimators=5, random_state=42)
    bag.fit(X_clf_train, y_clf_train)
    print(f"    Number of estimators: {len(bag.estimators_)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
