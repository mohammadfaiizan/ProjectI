"""
Scikit-learn Robust Regressors: HuberRegressor, RANSACRegressor, TheilSenRegressor
"""

import numpy as np
from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def main():
    print("=" * 60)
    print("Robust Regressors: Huber, RANSAC, Theil-Sen")
    print("=" * 60)

    np.random.seed(42)
    X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train[10:15] = 500
    y_train[20:22] = -300

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] HuberRegressor - robust to outliers:")
    huber = HuberRegressor(epsilon=1.35)
    huber.fit(X_train_scaled, y_train)
    print(f"    epsilon=1.35 (transition point)")
    print(f"    Test MSE: {mean_squared_error(y_test, huber.predict(X_test_scaled)):.4f}")

    print("\n[2] RANSACRegressor - random sample consensus:")
    ransac = RANSACRegressor(random_state=42)
    ransac.fit(X_train_scaled, y_train)
    print(f"    Inliers: {np.sum(ransac.inlier_mask_)} / {len(y_train)}")
    print(f"    Test MSE: {mean_squared_error(y_test, ransac.predict(X_test_scaled)):.4f}")

    print("\n[3] TheilSenRegressor - median-based:")
    theil = TheilSenRegressor(random_state=42)
    theil.fit(X_train_scaled, y_train)
    print(f"    Test MSE: {mean_squared_error(y_test, theil.predict(X_test_scaled)):.4f}")

    print("\n[4] Comparison:")
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    print(f"    LinearRegression: MSE = {mean_squared_error(y_test, lr.predict(X_test_scaled)):.4f}")
    print(f"    HuberRegressor:   MSE = {mean_squared_error(y_test, huber.predict(X_test_scaled)):.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
