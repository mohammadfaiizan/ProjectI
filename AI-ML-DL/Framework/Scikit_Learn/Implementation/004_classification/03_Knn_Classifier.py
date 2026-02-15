"""
Scikit-learn KNeighborsClassifier: n_neighbors, weights, metric
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("KNeighborsClassifier: n_neighbors, weights, metric")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] n_neighbors effect:")
    for k in [1, 3, 5, 10, 20, 50]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test_scaled))
        print(f"    n_neighbors={k}: Accuracy = {acc:.4f}")

    print("\n[2] weights options:")
    for w in ["uniform", "distance"]:
        knn = KNeighborsClassifier(n_neighbors=10, weights=w)
        knn.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test_scaled))
        print(f"    weights='{w}': Accuracy = {acc:.4f}")

    print("\n[3] metric options:")
    for m in ["euclidean", "manhattan", "minkowski"]:
        knn = KNeighborsClassifier(n_neighbors=10, metric=m)
        knn.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test_scaled))
        print(f"    metric='{m}': Accuracy = {acc:.4f}")

    print("\n[4] predict_proba:")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    probs = knn.predict_proba(X_test_scaled[:3])
    print("    First 3 samples probabilities:\n", probs)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
