"""
Scikit-learn NearestCentroid: metric, shrink_threshold
"""

import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("NearestCentroid: metric, shrink_threshold")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] Basic NearestCentroid:")
    nc = NearestCentroid()
    nc.fit(X_train_scaled, y_train)
    print(f"    Accuracy: {accuracy_score(y_test, nc.predict(X_test_scaled)):.4f}")
    print("    Centroids shape:", nc.centroids_.shape)

    print("\n[2] metric options:")
    for m in ["euclidean", "manhattan"]:
        nc = NearestCentroid(metric=m)
        nc.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, nc.predict(X_test_scaled))
        print(f"    metric='{m}': Accuracy = {acc:.4f}")

    print("\n[3] shrink_threshold - centroid shrinkage:")
    for shrink in [0.0, 0.5, 1.0, 2.0]:
        nc = NearestCentroid(shrink_threshold=shrink)
        nc.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, nc.predict(X_test_scaled))
        print(f"    shrink_threshold={shrink}: Accuracy = {acc:.4f}")

    print("\n[4] centroids_ per class:")
    nc = NearestCentroid()
    nc.fit(X_train_scaled, y_train)
    for i, c in enumerate(nc.centroids_):
        print(f"    Class {i}: {c}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
