"""
Scikit-learn Regression Metrics: MSE, MAE, R2, MAPE, explained_variance_score
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
)


def main():
    print("=" * 60)
    print("Regression Metrics: MSE, MAE, R2, MAPE, Explained Variance")
    print("=" * 60)

    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print("\n[1] mean_squared_error:")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"    MSE:  {mse:.4f}")
    print(f"    RMSE: {rmse:.4f}")

    print("\n[2] squared=False for RMSE directly:")
    rmse_direct = mean_squared_error(y_test, y_pred, squared=False)
    print(f"    RMSE (squared=False): {rmse_direct:.4f}")

    print("\n[3] mean_absolute_error:")
    mae = mean_absolute_error(y_test, y_pred)
    print(f"    MAE: {mae:.4f}")

    print("\n[4] r2_score:")
    r2 = r2_score(y_test, y_pred)
    print(f"    R2: {r2:.4f}")

    print("\n[5] mean_absolute_percentage_error:")
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"    MAPE: {mape:.4f} (as decimal)")

    print("\n[6] explained_variance_score:")
    evs = explained_variance_score(y_test, y_pred)
    print(f"    Explained variance: {evs:.4f}")

    print("\n[7] Multi-output (sample):")
    y_multi = np.column_stack([y_test, y_test + np.random.randn(len(y_test)) * 5])
    y_pred_multi = np.column_stack([y_pred, y_pred + np.random.randn(len(y_pred)) * 5])
    r2_multi = r2_score(y_multi, y_pred_multi, multioutput="variance_weighted")
    print(f"    R2 (multioutput variance_weighted): {r2_multi:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
