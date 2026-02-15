"""
Scikit-learn ElasticNet Regression: l1_ratio, ElasticNetCV
"""

import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def main():
    print("=" * 60)
    print("ElasticNet Regression: l1_ratio, ElasticNetCV")
    print("=" * 60)

    X, y = make_regression(n_samples=200, n_features=10, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] ElasticNet l1_ratio (0=Ridge, 1=Lasso):")
    for l1 in [0.0, 0.5, 1.0]:
        en = ElasticNet(alpha=0.1, l1_ratio=l1, random_state=42)
        en.fit(X_train_scaled, y_train)
        mse = mean_squared_error(y_test, en.predict(X_test_scaled))
        nz = np.sum(en.coef_ != 0)
        print(f"    l1_ratio={l1}: MSE={mse:.4f}, non-zero coefs={nz}")

    print("\n[2] ElasticNetCV - automatic alpha and l1_ratio:")
    en_cv = ElasticNetCV(cv=5, random_state=42)
    en_cv.fit(X_train_scaled, y_train)
    print(f"    Best alpha: {en_cv.alpha_:.4f}")
    print(f"    Best l1_ratio: {en_cv.l1_ratio_:.4f}")
    print(f"    Test MSE: {mean_squared_error(y_test, en_cv.predict(X_test_scaled)):.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()