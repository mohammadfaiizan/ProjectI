"""
Scikit-learn Ridge Regression: alpha, solver, RidgeCV for cross-validation
"""

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def main():
    print("=" * 60)
    print("Ridge Regression: alpha, solver, RidgeCV")
    print("=" * 60)

    X, y = make_regression(n_samples=200, n_features=10, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] Ridge with fixed alpha:")
    ridge = Ridge(alpha=1.0, solver="cholesky", random_state=42)
    ridge.fit(X_train_scaled, y_train)
    y_pred = ridge.predict(X_test_scaled)
    print(f"    alpha=1.0, solver='cholesky'")
    print(f"    Test MSE: {mean_squared_error(y_test, y_pred):.4f}")

    print("\n[2] Ridge with different solvers:")
    for solver in ["auto", "svd", "cholesky", "lsqr", "sag", "saga"]:
        try:
            r = Ridge(alpha=1.0, solver=solver, random_state=42)
            r.fit(X_train_scaled, y_train)
            mse = mean_squared_error(y_test, r.predict(X_test_scaled))
            print(f"    {solver}: MSE = {mse:.4f}")
        except Exception as e:
            print(f"    {solver}: {e}")

    print("\n[3] RidgeCV - automatic alpha selection:")
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
    ridge_cv.fit(X_train_scaled, y_train)
    print(f"    Best alpha (CV): {ridge_cv.alpha_}")
    print(f"    Test MSE: {mean_squared_error(y_test, ridge_cv.predict(X_test_scaled)):.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
