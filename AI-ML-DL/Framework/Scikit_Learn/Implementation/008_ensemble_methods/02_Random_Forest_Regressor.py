"""
Scikit-learn RandomForestRegressor: n_estimators, max_depth, feature_importances_
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    print("=" * 60)
    print("RandomForestRegressor: n_estimators, max_depth, feature_importances_")
    print("=" * 60)

    X, y = make_regression(n_samples=300, n_features=10, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[1] n_estimators:")
    for n in [10, 50, 100, 200]:
        rf = RandomForestRegressor(n_estimators=n, random_state=42)
        rf.fit(X_train, y_train)
        r2 = r2_score(y_test, rf.predict(X_test))
        print(f"    n_estimators={n}: R2 = {r2:.4f}")

    print("\n[2] max_depth:")
    for depth in [3, 5, 10, None]:
        rf = RandomForestRegressor(n_estimators=50, max_depth=depth, random_state=42)
        rf.fit(X_train, y_train)
        r2 = r2_score(y_test, rf.predict(X_test))
        print(f"    max_depth={depth}: R2 = {r2:.4f}")

    print("\n[3] feature_importances_:")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    for i, imp in enumerate(rf.feature_importances_):
        print(f"    Feature {i}: {imp:.4f}")

    print("\n[4] min_samples_split and min_samples_leaf:")
    rf = RandomForestRegressor(n_estimators=50, min_samples_split=5, min_samples_leaf=2, random_state=42)
    rf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, rf.predict(X_test))
    print(f"    min_samples_split=5, min_samples_leaf=2: MSE = {mse:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()