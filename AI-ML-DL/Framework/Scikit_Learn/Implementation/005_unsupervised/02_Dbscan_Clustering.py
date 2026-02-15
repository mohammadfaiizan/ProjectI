"""
Scikit-learn DBSCAN: eps, min_samples, core_sample_indices_, labels_, -1 for noise
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons


def main():
    print("=" * 60)
    print("DBSCAN Clustering")
    print("=" * 60)

    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

    print("\n[1] DBSCAN basic usage:")
    db = DBSCAN(eps=0.3, min_samples=5)
    db.fit(X)
    n_clusters = len(set(db.labels_) - {-1})
    n_noise = (db.labels_ == -1).sum()
    print(f"    n_clusters found: {n_clusters}")
    print(f"    noise points (label=-1): {n_noise}")
    print(f"    core_sample_indices_ count: {len(db.core_sample_indices_)}")

    print("\n[2] eps effect:")
    for eps in [0.1, 0.2, 0.3, 0.5]:
        db = DBSCAN(eps=eps, min_samples=5)
        db.fit(X)
        n_clusters = len(set(db.labels_) - {-1})
        n_noise = (db.labels_ == -1).sum()
        print(f"    eps={eps}: clusters={n_clusters}, noise={n_noise}")

    print("\n[3] min_samples effect:")
    for ms in [3, 5, 10, 20]:
        db = DBSCAN(eps=0.3, min_samples=ms)
        db.fit(X)
        n_clusters = len(set(db.labels_) - {-1})
        print(f"    min_samples={ms}: clusters={n_clusters}")

    print("\n[4] labels_ breakdown:")
    unique, counts = np.unique(db.labels_, return_counts=True)
    for u, c in zip(unique, counts):
        label_str = "noise" if u == -1 else f"cluster {u}"
        print(f"    {label_str}: {c} points")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
