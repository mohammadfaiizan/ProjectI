"""
Scikit-learn Hierarchical Clustering: AgglomerativeClustering, linkage, distance_threshold, dendrogram
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import linkage


def main():
    print("=" * 60)
    print("Hierarchical (Agglomerative) Clustering")
    print("=" * 60)

    X, _ = make_blobs(n_samples=100, n_features=2, centers=4, random_state=42)

    print("\n[1] AgglomerativeClustering with n_clusters:")
    ac = AgglomerativeClustering(n_clusters=4, linkage="ward")
    ac.fit(X)
    print(f"    labels_ (first 15): {ac.labels_[:15]}")
    print(f"    n_leaves_: {ac.n_leaves_}")

    print("\n[2] linkage options:")
    for link in ["ward", "complete", "average", "single"]:
        ac = AgglomerativeClustering(n_clusters=4, linkage=link)
        ac.fit(X)
        print(f"    linkage='{link}': n_clusters={len(set(ac.labels_))}")

    print("\n[3] distance_threshold (flat clustering):")
    ac = AgglomerativeClustering(n_clusters=None, distance_threshold=5, linkage="ward")
    ac.fit(X)
    print(f"    n_clusters (auto): {ac.n_clusters_}")

    print("\n[4] Dendrogram linkage matrix:")
    Z = linkage(X, method="ward")
    print(f"    linkage matrix shape: {Z.shape}")
    print(f"    First 3 merges:\n{Z[:3]}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
