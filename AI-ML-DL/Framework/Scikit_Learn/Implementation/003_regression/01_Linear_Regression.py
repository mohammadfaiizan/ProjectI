"""
Scikit-learn Linear Regression: fit, predict, coef_, intercept_, score
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    print("=" * 60)
    print("Linear Regression: OLS fit, predict, coef_, intercept_, score")
    print("=" * 60)

    X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n[1] Coefficients (coef_):")
    for i, c in enumerate(model.coef_):
        print(f"    Feature {i}: {c:.4f}")

    print(f"\n[2] Intercept (intercept_): {model.intercept_:.4f}")

    print("\n[3] Predictions (first 5):")
    print(f"    {y_pred[:5]}")

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"\n[4] R^2 score - Train: {train_score:.4f}, Test: {test_score:.4f}")

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n[5] Test MSE: {mse:.4f}, R2: {r2:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
