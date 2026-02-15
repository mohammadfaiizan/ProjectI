"""
Scikit-learn Support Vector Regression: SVR (kernel, C, epsilon), LinearSVR
"""

import numpy as np
from sklearn.svm import SVR, LinearSVR
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def main():
    print("=" * 60)
    print("Support Vector Regression: SVR, LinearSVR")
    print("=" * 60)

    X, y = make_regression(n_samples=150, n_features=3, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] SVR with different kernels:")
    for kernel in ["linear", "rbf", "poly"]:
        svr = SVR(kernel=kernel, C=1.0, epsilon=0.1)
        svr.fit(X_train_scaled, y_train)
        mse = mean_squared_error(y_test, svr.predict(X_test_scaled))
        print(f"    kernel='{kernel}': MSE = {mse:.4f}")

    print("\n[2] SVR - C and epsilon parameters:")
    svr = SVR(kernel="rbf", C=10.0, epsilon=0.05)
    svr.fit(X_train_scaled, y_train)
    print(f"    C=10.0 (regularization), epsilon=0.05 (tube width)")
    print(f"    Test MSE: {mean_squared_error(y_test, svr.predict(X_test_scaled)):.4f}")

    print("\n[3] LinearSVR - faster for linear case:")
    lsvr = LinearSVR(C=1.0, epsilon=0.1, max_iter=5000)
    lsvr.fit(X_train_scaled, y_train)
    print(f"    LinearSVR (no kernel trick)")
    print(f"    Test MSE: {mean_squared_error(y_test, lsvr.predict(X_test_scaled)):.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
