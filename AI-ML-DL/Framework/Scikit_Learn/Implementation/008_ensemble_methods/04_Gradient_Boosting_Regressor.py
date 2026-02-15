"""
Scikit-learn GradientBoostingRegressor: n_estimators, learning_rate, max_depth, subsample
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    print("=" * 60)
    print("GradientBoostingRegressor: n_estimators, learning_rate, max_depth, subsample")
    print("=" * 60)

    X, y = make_regression(n_samples=500, n_features=15, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[1] n_estimators and learning_rate:")
    for n, lr in [(50, 0.1), (100, 0.1), (200, 0.05)]:
        gb = GradientBoostingRegressor(n_estimators=n, learning_rate=lr, random_state=42)
        gb.fit(X_train, y_train)
        mse = mean_squared_error(y_test, gb.predict(X_test))
        r2 = r2_score(y_test, gb.predict(X_test))
        print(f"    n={n}, lr={lr}: MSE={mse:.2f}, R2={r2:.4f}")

    print("\n[2] max_depth:")
    for depth in [2, 4, 6, 10]:
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=depth, random_state=42)
        gb.fit(X_train, y_train)
        r2 = r2_score(y_test, gb.predict(X_test))
        print(f"    max_depth={depth}: R2 = {r2:.4f}")

    print("\n[3] subsample:")
    for ss in [0.5, 0.8, 1.0]:
        gb = GradientBoostingRegressor(n_estimators=100, subsample=ss, random_state=42)
        gb.fit(X_train, y_train)
        r2 = r2_score(y_test, gb.predict(X_test))
        print(f"    subsample={ss}: R2 = {r2:.4f}")

    print("\n[4] staged_predict - incremental predictions:")
    gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
    gb.fit(X_train, y_train)
    staged = list(gb.staged_predict(X_test))
    for i in [0, 24, 49]:
        r2 = r2_score(y_test, staged[i])
        print(f"    After {i+1} trees: R2 = {r2:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
