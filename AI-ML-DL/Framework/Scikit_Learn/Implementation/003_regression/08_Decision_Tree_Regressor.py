"""
Scikit-learn DecisionTreeRegressor: max_depth, min_samples_split, pruning
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def main():
    print("=" * 60)
    print("DecisionTreeRegressor: max_depth, min_samples_split, pruning")
    print("=" * 60)

    X, y = make_regression(n_samples=200, n_features=4, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[1] Unpruned tree (full depth):")
    dt_full = DecisionTreeRegressor(random_state=42)
    dt_full.fit(X_train, y_train)
    print(f"    Tree depth: {dt_full.get_depth()}")
    print(f"    Test MSE: {mean_squared_error(y_test, dt_full.predict(X_test)):.4f}")

    print("\n[2] max_depth - pruning:")
    for depth in [2, 5, 10, None]:
        dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        d = dt.get_depth() if depth is None else depth
        mse = mean_squared_error(y_test, dt.predict(X_test))
        print(f"    max_depth={depth}: depth={d}, MSE = {mse:.4f}")

    print("\n[3] min_samples_split - leaf size:")
    for mss in [2, 5, 10, 20]:
        dt = DecisionTreeRegressor(min_samples_split=mss, max_depth=10, random_state=42)
        dt.fit(X_train, y_train)
        mse = mean_squared_error(y_test, dt.predict(X_test))
        print(f"    min_samples_split={mss}: MSE = {mse:.4f}")

    print("\n[4] min_samples_leaf for pruning:")
    dt = DecisionTreeRegressor(min_samples_leaf=10, random_state=42)
    dt.fit(X_train, y_train)
    print(f"    min_samples_leaf=10: depth={dt.get_depth()}")
    print(f"    Test MSE: {mean_squared_error(y_test, dt.predict(X_test)):.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
