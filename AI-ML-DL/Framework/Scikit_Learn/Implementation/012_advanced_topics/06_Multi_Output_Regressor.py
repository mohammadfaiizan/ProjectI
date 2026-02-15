"""
Scikit-learn MultiOutputRegressor and RegressorChain
"""

import numpy as np
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def main():
    print("=" * 60)
    print("MultiOutputRegressor and RegressorChain")
    print("=" * 60)

    X, y = make_regression(n_samples=200, n_features=10, n_targets=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    print("\n[1] MultiOutputRegressor (independent targets):")
    reg = MultiOutputRegressor(Ridge(alpha=1.0, random_state=42))
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    print(f"    MSE: {mse:.4f}")
    print(f"    Pred shape: {pred.shape}")

    print("\n[2] RegressorChain (chained, uses previous targets as features):")
    chain = RegressorChain(Ridge(alpha=1.0, random_state=42), order=[0, 1, 2])
    chain.fit(X_train, y_train)
    pred_chain = chain.predict(X_test)
    mse_chain = mean_squared_error(y_test, pred_chain)
    print(f"    MSE: {mse_chain:.4f}")
    print(f"    order: {chain.order}")

    print("\n[3] RegressorChain with random order:")
    chain_rand = RegressorChain(Ridge(alpha=1.0, random_state=42), order=None)
    chain_rand.fit(X_train, y_train)
    pred_rand = chain_rand.predict(X_test)
    print(f"    order (random): {chain_rand.order}")

    print("\n[4] Compare MultiOutput vs Chain:")
    print(f"    MultiOutput MSE: {mse:.4f}")
    print(f"    Chain MSE: {mse_chain:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
