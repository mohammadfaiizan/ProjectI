"""
Scikit-learn KernelPCA: kernel, gamma, n_components
"""

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler


def main():
    print("=" * 60)
    print("Kernel PCA")
    print("=" * 60)

    X, _ = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=42)
    X = StandardScaler().fit_transform(X)

    print("\n[1] KernelPCA with RBF kernel:")
    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=15, random_state=42)
    X_transformed = kpca.fit_transform(X)
    print(f"    Transformed shape: {X_transformed.shape}")

    print("\n[2] kernel options:")
    for kernel in ["rbf", "poly", "cosine", "linear"]:
        kpca = KernelPCA(n_components=2, kernel=kernel, gamma=0.1 if kernel == "rbf" else None, random_state=42)
        X_t = kpca.fit_transform(X)
        print(f"    kernel='{kernel}': output range [{X_t.min():.2f}, {X_t.max():.2f}]")

    print("\n[3] gamma effect (RBF):")
    for gamma in [0.1, 1.0, 10.0]:
        kpca = KernelPCA(n_components=2, kernel="rbf", gamma=gamma, random_state=42)
        X_t = kpca.fit_transform(X)
        print(f"    gamma={gamma}: std={X_t.std():.2f}")

    print("\n[4] fit_inverse_transform:")
    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=15, fit_inverse_transform=True, random_state=42)
    kpca.fit(X)
    X_reconstructed = kpca.inverse_transform(kpca.transform(X))
    print(f"    Reconstruction MSE: {np.mean((X - X_reconstructed)**2):.6f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
