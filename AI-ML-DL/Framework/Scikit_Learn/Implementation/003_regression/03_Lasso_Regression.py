"""
Scikit-learn Lasso Regression: alpha, LassoCV for cross-validation
"""

import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def main():
    print("=" * 60)
    print("Lasso Regression: alpha, LassoCV")
    print("=" * 60)

    X, y = make_regression(n_samples=200, n_features=10, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] Lasso with fixed alpha:")
    lasso = Lasso(alpha=0.1, random_state=42)
    lasso.fit(X_train_scaled, y_train)
    y_pred = lasso.predict(X_test_scaled)
    print(f"    alpha=0.1, Test MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"    Non-zero coefs: {np.sum(lasso.coef_ != 0)} / {len(lasso.coef_)}")

    print("\n[2] Effect of alpha on sparsity:")
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        l = Lasso(alpha=alpha, random_state=42)
        l.fit(X_train_scaled, y_train)
        nz = np.sum(l.coef_ != 0)
        mse = mean_squared_error(y_test, l.predict(X_test_scaled))
        print(f"    alpha={alpha}: non-zero={nz}, MSE={mse:.4f}")

    print("\n[3] LassoCV - automatic alpha selection:")
    lasso_cv = LassoCV(cv=5, random_state=42)
    lasso_cv.fit(X_train_scaled, y_train)
    print(f"    Best alpha (CV): {lasso_cv.alpha_:.4f}")
    print(f"    Test MSE: {mean_squared_error(y_test, lasso_cv.predict(X_test_scaled)):.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()