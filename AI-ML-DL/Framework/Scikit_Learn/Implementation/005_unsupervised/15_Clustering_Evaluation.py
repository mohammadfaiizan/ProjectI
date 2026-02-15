"""
Scikit-learn Clustering Evaluation: silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)


def main():
    print("=" * 60)
    print("Clustering Evaluation Metrics")
    print("=" * 60)

    X, y_true = make_blobs(n_samples=500, n_features=2, centers=4, random_state=42)

    print("\n[1] Internal metrics (no ground truth):")
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X)
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    print(f"    silhouette_score (higher better): {sil:.4f}")
    print(f"    calinski_harabasz_score (higher better): {ch:.2f}")
    print(f"    davies_bouldin_score (lower better): {db:.4f}")

    print("\n[2] External metric (with ground truth):")
    ari = adjusted_rand_score(y_true, labels)
    print(f"    adjusted_rand_score: {ari:.4f}")

    print("\n[3] Comparing k values (internal):")
    for k in [2, 3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=42)
        lab = km.fit_predict(X)
        sil = silhouette_score(X, lab)
        db = davies_bouldin_score(X, lab)
        print(f"    k={k}: silhouette={sil:.4f}, davies_bouldin={db:.4f}")

    print("\n[4] Perfect labels vs random:")
    ari_perfect = adjusted_rand_score(y_true, y_true)
    ari_random = adjusted_rand_score(y_true, np.random.permutation(labels))
    print(f"    ARI (perfect): {ari_perfect:.4f}")
    print(f"    ARI (random): {ari_random:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
