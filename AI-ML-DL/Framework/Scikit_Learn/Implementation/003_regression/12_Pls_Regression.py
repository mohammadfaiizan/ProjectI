"""
Scikit-learn PLS Regression: PLSRegression (n_components)
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def main():
    print("=" * 60)
    print("PLS Regression: PLSRegression (n_components)")
    print("=" * 60)

    X, y = make_regression(n_samples=200, n_features=10, n_informative=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] PLSRegression - n_components (latent dimensions):")
    for n_comp in [1, 2, 3, 5, 10]:
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_train_scaled, y_train)
        mse = mean_squared_error(y_test, pls.predict(X_test_scaled).ravel())
        print(f"    n_components={n_comp}: MSE = {mse:.4f}")

    print("\n[2] PLS loadings and coefficients:")
    pls = PLSRegression(n_components=3)
    pls.fit(X_train_scaled, y_train)
    print("    x_weights_ shape:", pls.x_weights_.shape)
    print("    y_loadings_ shape:", pls.y_loadings_.shape)
    print("    coef_ shape:", pls.coef_.shape)

    print("\n[3] Transform to latent space:")
    X_train_pls = pls.transform(X_train_scaled)
    print(f"    Transformed X shape: {X_train_pls.shape}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
