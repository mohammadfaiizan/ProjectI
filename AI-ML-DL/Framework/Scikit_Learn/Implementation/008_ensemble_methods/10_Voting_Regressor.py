"""
Scikit-learn VotingRegressor: estimators, weights
"""

import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    print("=" * 60)
    print("VotingRegressor: estimators, weights")
    print("=" * 60)

    X, y = make_regression(n_samples=300, n_features=10, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    estimators = [
        ("lr", LinearRegression()),
        ("ridge", Ridge(alpha=1.0)),
        ("dt", DecisionTreeRegressor(max_depth=10, random_state=42)),
        ("svr", SVR(kernel="rbf", C=1.0)),
    ]

    print("\n[1] VotingRegressor - averaged predictions:")
    vr = VotingRegressor(estimators=estimators)
    vr.fit(X_train, y_train)
    y_pred = vr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"    MSE = {mse:.2f}, R2 = {r2:.4f}")

    print("\n[2] Individual estimator R2:")
    for name, est in estimators:
        est.fit(X_train, y_train)
        r2 = r2_score(y_test, est.predict(X_test))
        print(f"    {name}: R2 = {r2:.4f}")

    print("\n[3] Custom weights:")
    vr_weighted = VotingRegressor(estimators=estimators, weights=[1, 2, 1, 1])
    vr_weighted.fit(X_train, y_train)
    r2 = r2_score(y_test, vr_weighted.predict(X_test))
    print(f"    weights=[1,2,1,1]: R2 = {r2:.4f}")

    print("\n[4] named_estimators_:")
    vr.fit(X_train, y_train)
    print(f"    Estimator names: {list(vr.named_estimators_.keys())}")

    print("\n[5] Predictions from each estimator:")
    for name, est in vr.named_estimators_.items():
        pred = est.predict(X_test[:3])
        print(f"    {name} (first 3): {pred}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
