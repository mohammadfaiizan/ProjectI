"""
Scikit-learn Other Clustering: MeanShift, OPTICS, Birch; HDBSCAN (optional, requires hdbscan package)
"""

import numpy as np
from sklearn.cluster import MeanShift, OPTICS, Birch
from sklearn.datasets import make_blobs

try:
    from hdbscan import HDBSCAN
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


def main():
    print("=" * 60)
    print("Other Clustering: MeanShift, OPTICS, Birch, HDBSCAN")
    print("=" * 60)

    X, _ = make_blobs(n_samples=300, n_features=2, centers=4, random_state=42)

    print("\n[1] MeanShift (bandwidth auto):")
    ms = MeanShift()
    ms.fit(X)
    print(f"    n_clusters: {ms.labels_.max() + 1}")
    print(f"    cluster_centers_ shape: {ms.cluster_centers_.shape}")

    print("\n[2] OPTICS (min_samples, xi, min_cluster_size):")
    opt = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
    opt.fit(X)
    n_clusters = len(set(opt.labels_) - {-1})
    print(f"    n_clusters: {n_clusters}")
    print(f"    reachability_ (first 5): {opt.reachability_[:5]}")

    print("\n[3] Birch (n_clusters, threshold):")
    birch = Birch(n_clusters=4, threshold=0.5)
    birch.fit(X)
    print(f"    labels_ (first 10): {birch.labels_[:10]}")
    print(f"    subcluster_centers_ count: {len(birch.subcluster_centers_)}")

    if HAS_HDBSCAN:
        print("\n[4] HDBSCAN (min_cluster_size, min_samples):")
        hdb = HDBSCAN(min_cluster_size=5, min_samples=3)
        labels = hdb.fit_predict(X)
        n_clusters = len(set(labels) - {-1})
        n_noise = (labels == -1).sum()
        print(f"    n_clusters: {n_clusters}, noise: {n_noise}")
    else:
        print("\n[4] HDBSCAN: install 'hdbscan' package for HDBSCAN support")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
