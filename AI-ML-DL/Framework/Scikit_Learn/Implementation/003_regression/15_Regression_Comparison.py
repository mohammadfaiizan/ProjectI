"""
Scikit-learn Regression Comparison: Comparing all regressors on one dataset
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor,
    BayesianRidge, HuberRegressor, QuantileRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def main():
    print("=" * 60)
    print("Regression Model Comparison")
    print("=" * 60)

    X, y = make_regression(n_samples=300, n_features=10, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.5),
        "SGDRegressor": SGDRegressor(max_iter=1000),
        "BayesianRidge": BayesianRidge(),
        "HuberRegressor": HuberRegressor(),
        "SVR": SVR(kernel="rbf", C=1.0),
        "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=10),
        "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=10),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=50, max_depth=10),
    }

    print("\n[1] Test MSE and R2:")
    print("-" * 50)
    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append((name, mse, r2))
        print(f"    {name:25s}: MSE={mse:8.2f}, R2={r2:.4f}")

    print("\n[2] Cross-validation (5-fold) MSE:")
    print("-" * 50)
    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="neg_mean_squared_error")
        cv_mse = -scores.mean()
        print(f"    {name:25s}: CV MSE = {cv_mse:.2f}")

    print("\n[3] Best model by R2:")
    best = max(results, key=lambda x: x[2])
    print(f"    {best[0]}: R2 = {best[2]:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
