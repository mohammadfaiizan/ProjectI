"""
Scikit-learn Polynomial Regression: PolynomialFeatures + LinearRegression pipeline
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def main():
    print("=" * 60)
    print("Polynomial Regression: PolynomialFeatures + LinearRegression")
    print("=" * 60)

    np.random.seed(42)
    X = np.sort(np.random.uniform(0, 5, 150)).reshape(-1, 1)
    y = 2 * X.ravel() ** 2 - 3 * X.ravel() + 1 + np.random.normal(0, 2, 150)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[1] PolynomialFeatures - degree expansion:")
    for degree in [1, 2, 3, 5]:
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly = poly.fit_transform(X_train)
        print(f"    degree={degree}: features {X_train.shape[1]} -> {X_poly.shape[1]}")

    print("\n[2] Pipeline - PolynomialFeatures + LinearRegression:")
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("linear", LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"    degree=2: MSE = {mean_squared_error(y_test, y_pred):.4f}")

    print("\n[3] Degree comparison (overfitting check):")
    for degree in [1, 2, 3, 5, 8]:
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=degree)),
            ("linear", LinearRegression())
        ])
        pipe.fit(X_train, y_train)
        train_mse = mean_squared_error(y_train, pipe.predict(X_train))
        test_mse = mean_squared_error(y_test, pipe.predict(X_test))
        print(f"    degree={degree}: Train MSE={train_mse:.4f}, Test MSE={test_mse:.4f}")

    print("\n[4] interaction_only and include_bias:")
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_inter = poly.fit_transform(np.array([[1, 2], [3, 4]]))
    print("    interaction_only=True:", X_inter)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
