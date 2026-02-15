"""
Scikit-learn SpectralClustering: n_clusters, affinity, assign_labels
"""

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons


def main():
    print("=" * 60)
    print("Spectral Clustering")
    print("=" * 60)

    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

    print("\n[1] SpectralClustering basic usage:")
    sc = SpectralClustering(n_clusters=2, affinity="rbf", random_state=42)
    labels = sc.fit_predict(X)
    print(f"    labels_ (first 15): {labels[:15]}")
    print(f"    n_clusters: {len(set(labels))}")

    print("\n[2] affinity options:")
    for aff in ["rbf", "nearest_neighbors"]:
        sc = SpectralClustering(n_clusters=2, affinity=aff, random_state=42)
        labels = sc.fit_predict(X)
        print(f"    affinity='{aff}': clusters={len(set(labels))}")

    print("\n[3] assign_labels options:")
    for al in ["kmeans", "discretize"]:
        sc = SpectralClustering(n_clusters=2, assign_labels=al, random_state=42)
        labels = sc.fit_predict(X)
        print(f"    assign_labels='{al}': clusters={len(set(labels))}")

    print("\n[4] n_clusters sweep:")
    for k in [2, 3, 4]:
        sc = SpectralClustering(n_clusters=k, random_state=42)
        labels = sc.fit_predict(X)
        print(f"    n_clusters={k}: unique labels={len(set(labels))}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
