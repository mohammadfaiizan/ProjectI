"""
Scikit-learn Bayesian Regression: BayesianRidge, ARDRegression
"""

import numpy as np
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def main():
    print("=" * 60)
    print("Bayesian Regression: BayesianRidge, ARDRegression")
    print("=" * 60)

    X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] BayesianRidge - automatic relevance determination:")
    br = BayesianRidge()
    br.fit(X_train_scaled, y_train)
    y_pred = br.predict(X_test_scaled)
    print(f"    Test MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print("    Coefficients:", br.coef_)
    print("    Lambda (precision):", br.lambda_)
    print("    Alpha (noise precision):", br.alpha_)

    print("\n[2] BayesianRidge - uncertainty estimates:")
    y_pred, y_std = br.predict(X_test_scaled, return_std=True)
    print(f"    Predictions (first 3): {y_pred[:3]}")
    print(f"    Std dev (first 3): {y_std[:3]}")

    print("\n[3] ARDRegression - Automatic Relevance Determination:")
    ard = ARDRegression()
    ard.fit(X_train_scaled, y_train)
    print("    Coefficients:", ard.coef_)
    print("    Lambda (per-feature):", ard.lambda_)
    print(f"    Test MSE: {mean_squared_error(y_test, ard.predict(X_test_scaled)):.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
