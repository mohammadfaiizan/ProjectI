"""
Scikit-learn KNeighborsRegressor: n_neighbors, weights, metric
"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def main():
    print("=" * 60)
    print("KNeighborsRegressor: n_neighbors, weights, metric")
    print("=" * 60)

    X, y = make_regression(n_samples=200, n_features=4, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] n_neighbors effect:")
    for k in [1, 5, 10, 20, 50]:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        mse = mean_squared_error(y_test, knn.predict(X_test_scaled))
        print(f"    n_neighbors={k}: MSE = {mse:.4f}")

    print("\n[2] weights options:")
    for w in ["uniform", "distance"]:
        knn = KNeighborsRegressor(n_neighbors=10, weights=w)
        knn.fit(X_train_scaled, y_train)
        mse = mean_squared_error(y_test, knn.predict(X_test_scaled))
        print(f"    weights='{w}': MSE = {mse:.4f}")

    print("\n[3] metric options:")
    for m in ["euclidean", "manhattan", "minkowski"]:
        knn = KNeighborsRegressor(n_neighbors=10, metric=m)
        knn.fit(X_train_scaled, y_train)
        mse = mean_squared_error(y_test, knn.predict(X_test_scaled))
        print(f"    metric='{m}': MSE = {mse:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
