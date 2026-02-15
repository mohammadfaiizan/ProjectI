"""
Scikit-learn incremental learning: partial_fit for online learning
"""

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_iris, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def main():
    print("=" * 60)
    print("Incremental Learning: partial_fit for online learning")
    print("=" * 60)

    print("\n[1] SGDClassifier with partial_fit:")
    X, y = load_iris(return_X_y=True)
    X_bin = X[y != 2]
    y_bin = y[y != 2]
    classes = np.unique(y_bin)
    sgd = SGDClassifier(random_state=42)
    for _ in range(5):
        for i in range(0, len(X_bin), 20):
            batch_X = X_bin[i : i + 20]
            batch_y = y_bin[i : i + 20]
            sgd.partial_fit(batch_X, batch_y, classes=classes)
    pred = sgd.predict(X_bin)
    print(f"    Accuracy: {accuracy_score(y_bin, pred):.4f}")

    print("\n[2] MiniBatchKMeans with partial_fit:")
    X_blob, _ = make_blobs(n_samples=500, centers=3, random_state=42)
    kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=50)
    for i in range(0, len(X_blob), 50):
        kmeans.partial_fit(X_blob[i : i + 50])
    print(f"    Inertia: {kmeans.inertia_:.2f}")
    print(f"    Cluster centers shape: {kmeans.cluster_centers_.shape}")

    print("\n[3] MultinomialNB with partial_fit (for text):")
    X_text = np.random.randint(0, 10, (100, 20))
    y_text = np.random.randint(0, 2, 100)
    nb = MultinomialNB()
    classes_nb = np.unique(y_text)
    for i in range(0, len(X_text), 25):
        nb.partial_fit(X_text[i : i + 25], y_text[i : i + 25], classes=classes_nb)
    pred_nb = nb.predict(X_text)
    print(f"    Accuracy: {accuracy_score(y_text, pred_nb):.4f}")

    print("\n[4] Estimators supporting partial_fit:")
    print("    SGDClassifier, SGDRegressor, MiniBatchKMeans,")
    print("    MultinomialNB, BernoulliNB, PassiveAggressiveClassifier")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
