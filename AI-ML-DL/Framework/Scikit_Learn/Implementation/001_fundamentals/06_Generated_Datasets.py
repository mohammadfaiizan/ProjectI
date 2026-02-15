"""
Scikit-learn Generated Datasets
make_classification, make_regression, make_blobs, make_moons, make_circles
"""

import numpy as np
from sklearn.datasets import (
    make_classification,
    make_regression,
    make_blobs,
    make_moons,
    make_circles,
)


def main():
    print("=" * 60)
    print("Generated Datasets: make_* functions")
    print("=" * 60)

    print("\n[1] make_classification:")
    X_clf, y_clf = make_classification(n_samples=100, n_features=20, n_informative=10, n_classes=3, random_state=42)
    print(f"    X shape: {X_clf.shape}, y shape: {y_clf.shape}")
    print(f"    Unique classes: {np.unique(y_clf)}")

    print("\n[2] make_regression:")
    X_reg, y_reg = make_regression(n_samples=100, n_features=5, n_informative=3, noise=10, random_state=42)
    print(f"    X shape: {X_reg.shape}, y shape: {y_reg.shape}")
    print(f"    y range: [{y_reg.min():.2f}, {y_reg.max():.2f}]")

    print("\n[3] make_blobs - Clustering:")
    X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
    print(f"    X shape: {X_blobs.shape}")
    print(f"    Centers: {len(np.unique(y_blobs))}")

    print("\n[4] make_moons - Non-linear classification:")
    X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)
    print(f"    X shape: {X_moons.shape}")
    print(f"    Two interleaving half circles")

    print("\n[5] make_circles - Concentric circles:")
    X_circles, y_circles = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)
    print(f"    X shape: {X_circles.shape}")
    print(f"    factor=0.5 controls inner/outer radius ratio")

    print("\n[6] Optional return_X_y:")
    X, y = make_classification(n_samples=50, random_state=42, return_X_y=True)
    print(f"    return_X_y=True: returns (X, y) tuple")

    print("\n[7] Custom parameters for make_classification:")
    X2, y2 = make_classification(n_samples=100, n_features=5, n_redundant=2, n_repeated=0, n_clusters_per_class=1)
    print(f"    n_redundant: redundant features")
    print(f"    n_clusters_per_class: clusters per class")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
