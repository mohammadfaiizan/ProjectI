"""
Scikit-learn PCA: n_components, explained_variance_ratio_, components_, inverse_transform
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


def main():
    print("=" * 60)
    print("PCA Decomposition")
    print("=" * 60)

    X, _ = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)

    print("\n[1] PCA basic usage:")
    pca = PCA(n_components=2, random_state=42)
    X_transformed = pca.fit_transform(X)
    print(f"    Transformed shape: {X_transformed.shape}")
    print(f"    explained_variance_ratio_: {pca.explained_variance_ratio_}")

    print("\n[2] Cumulative variance:")
    pca_full = PCA(random_state=42)
    pca_full.fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    print(f"    Cumulative: {cumvar}")

    print("\n[3] components_ (loadings):")
    print(f"    components_ shape: {pca.components_.shape}")

    print("\n[4] inverse_transform (reconstruction):")
    X_reconstructed = pca.inverse_transform(X_transformed)
    mse = np.mean((X - X_reconstructed) ** 2)
    print(f"    Reconstruction MSE: {mse:.6f}")

    print("\n[5] n_components as fraction (0.95):")
    pca_95 = PCA(n_components=0.95, random_state=42)
    pca_95.fit(X)
    print(f"    Components for 95% variance: {pca_95.n_components_}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
