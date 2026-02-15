"""
Scikit-learn LassoLars: LARS path, LassoLarsCV
"""

import numpy as np
from sklearn.linear_model import LassoLars, LassoLarsCV
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def main():
    print("=" * 60)
    print("LassoLars: LARS-based Lasso, LassoLarsCV")
    print("=" * 60)

    X, y = make_regression(n_samples=150, n_features=8, noise=12, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] LassoLars with alpha:")
    ll = LassoLars(alpha=0.01, random_state=42)
    ll.fit(X_train_scaled, y_train)
    y_pred = ll.predict(X_test_scaled)
    print(f"    alpha=0.01, Test MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"    Non-zero coefs: {np.sum(ll.coef_ != 0)}")

    print("\n[2] LassoLarsCV - automatic alpha:")
    ll_cv = LassoLarsCV(cv=5)
    ll_cv.fit(X_train_scaled, y_train)
    print(f"    Best alpha: {ll_cv.alpha_:.6f}")
    print(f"    Test MSE: {mean_squared_error(y_test, ll_cv.predict(X_test_scaled)):.4f}")

    print("\n[3] alphas_ path (first 5):")
    print(f"    {ll_cv.alphas_[:5]}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()