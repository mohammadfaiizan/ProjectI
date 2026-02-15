"""
Scikit-learn HistGradientBoostingClassifier and HistGradientBoostingRegressor: native missing values, categorical
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


def main():
    print("=" * 60)
    print("HistGradientBoostingClassifier/Regressor")
    print("=" * 60)

    X_clf, y_clf = load_breast_cancer(return_X_y=True)
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )

    print("\n[1] HistGradientBoostingClassifier - basic usage:")
    hgb_clf = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    hgb_clf.fit(X_clf_train, y_clf_train)
    acc = accuracy_score(y_clf_test, hgb_clf.predict(X_clf_test))
    print(f"    max_iter=100: Accuracy = {acc:.4f}")

    print("\n[2] Native missing value support:")
    X_missing = X_clf_train.copy()
    X_missing[0, 0] = np.nan
    X_missing[5, 3] = np.nan
    X_test_missing = X_clf_test.copy()
    X_test_missing[0, 0] = np.nan
    hgb = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    hgb.fit(X_missing, y_clf_train)
    acc = accuracy_score(y_clf_test, hgb.predict(X_test_missing))
    print(f"    With NaN in data: Accuracy = {acc:.4f}")

    print("\n[3] Categorical features (categorical_features):")
    X_cat_train = np.column_stack([X_clf_train[:, :2], np.random.randint(0, 3, (X_clf_train.shape[0], 1))])
    X_cat_test = np.column_stack([X_clf_test[:, :2], np.random.randint(0, 3, (X_clf_test.shape[0], 1))])
    try:
        hgb_cat = HistGradientBoostingClassifier(
            max_iter=100, categorical_features=[2], random_state=42
        )
        hgb_cat.fit(X_cat_train, y_clf_train)
        acc = accuracy_score(y_clf_test, hgb_cat.predict(X_cat_test))
        print(f"    categorical_features=[2]: Accuracy = {acc:.4f}")
    except TypeError:
        hgb_cat = HistGradientBoostingClassifier(max_iter=100, random_state=42)
        hgb_cat.fit(X_cat_train, y_clf_train)
        acc = accuracy_score(y_clf_test, hgb_cat.predict(X_cat_test))
        print(f"    (categorical_features not in this sklearn): Accuracy = {acc:.4f}")

    X_reg, y_reg = make_regression(n_samples=500, n_features=15, noise=10, random_state=42)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    print("\n[4] HistGradientBoostingRegressor:")
    hgb_reg = HistGradientBoostingRegressor(max_iter=100, random_state=42)
    hgb_reg.fit(X_reg_train, y_reg_train)
    r2 = r2_score(y_reg_test, hgb_reg.predict(X_reg_test))
    print(f"    max_iter=100: R2 = {r2:.4f}")

    print("\n[5] learning_rate and max_depth:")
    for lr, depth in [(0.1, 5), (0.05, 10)]:
        hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=lr, max_depth=depth, random_state=42)
        hgb.fit(X_clf_train, y_clf_train)
        acc = accuracy_score(y_clf_test, hgb.predict(X_clf_test))
        print(f"    lr={lr}, max_depth={depth}: Accuracy = {acc:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
