"""
Scikit-learn Manifold Learning: Isomap, LocallyLinearEmbedding, MDS
"""

import numpy as np
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler


def main():
    print("=" * 60)
    print("Manifold Learning: Isomap, LLE, MDS")
    print("=" * 60)

    X, _ = load_digits(return_X_y=True)
    X = StandardScaler().fit_transform(X[:300])

    print("\n[1] Isomap:")
    isomap = Isomap(n_components=2, n_neighbors=10)
    X_iso = isomap.fit_transform(X)
    print(f"    Transformed shape: {X_iso.shape}")
    print(f"    reconstruction_error(): {isomap.reconstruction_error():.4f}")

    print("\n[2] LocallyLinearEmbedding:")
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, method="standard")
    X_lle = lle.fit_transform(X)
    print(f"    Transformed shape: {X_lle.shape}")
    print(f"    reconstruction_error_: {lle.reconstruction_error_:.4f}")

    print("\n[3] LLE method options:")
    for method in ["standard", "ltsa", "hessian"]:
        lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, method=method)
        lle.fit(X)
        print(f"    method='{method}': err={lle.reconstruction_error_:.4f}")

    print("\n[4] MDS (metric and non-metric):")
    mds = MDS(n_components=2, random_state=42)
    X_mds = mds.fit_transform(X)
    print(f"    Transformed shape: {X_mds.shape}")
    print(f"    stress_: {mds.stress_:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
