"""
Scikit-learn Quantile Regression: QuantileRegressor (quantile, alpha)
"""

import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


def main():
    print("=" * 60)
    print("Quantile Regression: QuantileRegressor (quantile, alpha)")
    print("=" * 60)

    X, y = make_regression(n_samples=200, n_features=3, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] QuantileRegressor - different quantiles:")
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    models = {}
    for q in quantiles:
        qr = QuantileRegressor(quantile=q, alpha=0.5, solver="highs")
        qr.fit(X_train_scaled, y_train)
        models[q] = qr
        pred = qr.predict(X_test_scaled)
        print(f"    quantile={q}: MAE = {mean_absolute_error(y_test, pred):.4f}")

    print("\n[2] Prediction intervals (10th and 90th percentile):")
    pred_low = models[0.1].predict(X_test_scaled)
    pred_high = models[0.9].predict(X_test_scaled)
    coverage = np.mean((y_test >= pred_low) & (y_test <= pred_high))
    print(f"    80% interval coverage: {coverage:.2%}")

    print("\n[3] alpha (regularization) effect:")
    for alpha in [0.01, 0.5, 5.0]:
        qr = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
        qr.fit(X_train_scaled, y_train)
        mae = mean_absolute_error(y_test, qr.predict(X_test_scaled))
        print(f"    alpha={alpha}: MAE = {mae:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
