"""
Scikit-learn KMeans and MiniBatchKMeans: n_clusters, inertia_, labels_, cluster_centers_, silhouette
"""

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


def main():
    print("=" * 60)
    print("KMeans and MiniBatchKMeans Clustering")
    print("=" * 60)

    X, _ = make_blobs(n_samples=500, n_features=2, centers=4, random_state=42)

    print("\n[1] KMeans basic usage:")
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X)
    print(f"    inertia_: {kmeans.inertia_:.2f}")
    print(f"    labels_ (first 10): {kmeans.labels_[:10]}")
    print(f"    cluster_centers_ shape: {kmeans.cluster_centers_.shape}")

    print("\n[2] n_clusters effect on inertia:")
    for k in [2, 4, 6, 8]:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        sil = silhouette_score(X, km.labels_)
        print(f"    k={k}: inertia={km.inertia_:.2f}, silhouette={sil:.4f}")

    print("\n[3] MiniBatchKMeans (scalable):")
    mbk = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=100)
    mbk.fit(X)
    print(f"    inertia_: {mbk.inertia_:.2f}")
    print(f"    labels_ (first 10): {mbk.labels_[:10]}")

    print("\n[4] predict on new samples:")
    X_new = np.array([[0, 0], [10, 10]])
    pred = kmeans.predict(X_new)
    print(f"    Predictions for [[0,0], [10,10]]: {pred}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
