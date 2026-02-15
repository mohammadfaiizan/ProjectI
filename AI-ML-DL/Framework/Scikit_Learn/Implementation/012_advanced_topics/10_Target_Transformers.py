"""
Scikit-learn TransformedTargetRegressor: transformer, func, inverse_func
"""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import QuantileTransformer
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def main():
    print("=" * 60)
    print("TransformedTargetRegressor: transformer, func, inverse_func")
    print("=" * 60)

    X, y = make_regression(n_samples=200, n_features=5, random_state=42)
    y = np.exp(y / 50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    print("\n[1] TransformedTargetRegressor with QuantileTransformer:")
    reg = TransformedTargetRegressor(
        regressor=Ridge(alpha=1.0),
        transformer=QuantileTransformer(output_distribution="normal"),
    )
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    print(f"    RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.4f}")

    print("\n[2] TransformedTargetRegressor with func/inverse_func:")
    reg_log = TransformedTargetRegressor(
        regressor=Ridge(alpha=1.0),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    reg_log.fit(X_train, y_train)
    pred_log = reg_log.predict(X_test)
    print(f"    RMSE (log transform): {np.sqrt(mean_squared_error(y_test, pred_log)):.4f}")

    print("\n[3] Without transformation (baseline):")
    ridge_raw = Ridge(alpha=1.0)
    ridge_raw.fit(X_train, y_train)
    pred_raw = ridge_raw.predict(X_test)
    print(f"    RMSE (no transform): {np.sqrt(mean_squared_error(y_test, pred_raw)):.4f}")

    print("\n[4] Access inner regressor and transformer:")
    print(f"    regressor_: {type(reg.regressor_).__name__}")
    print(f"    transformer_: {type(reg.transformer_).__name__}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
