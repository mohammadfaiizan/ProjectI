"""
Scikit-learn large dataset strategies: partial_fit, out-of-core, HashingVectorizer
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA


def main():
    print("=" * 60)
    print("Large Dataset Strategies: partial_fit, out-of-core, HashingVectorizer")
    print("=" * 60)

    print("\n[1] SGDClassifier with partial_fit:")
    X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    sgd = SGDClassifier(max_iter=1, random_state=42, warm_start=False)
    batch_size = 500
    for i in range(0, len(X_train), batch_size):
        end = min(i + batch_size, len(X_train))
        sgd.partial_fit(X_train[i:end], y_train[i:end], classes=np.unique(y_train))
    score = sgd.score(X_test, y_test)
    print(f"    Batch size: {batch_size}, Batches: {len(X_train) // batch_size}")
    print(f"    Test accuracy: {score:.4f}")

    print("\n[2] HashingVectorizer for out-of-core text:")
    texts = ["hello world", "machine learning", "hello sklearn"] * 100
    hv = HashingVectorizer(n_features=2**10)
    X_h = hv.transform(texts)
    print(f"    Sparse shape: {X_h.shape}")
    print(f"    nnz: {X_h.nnz}")
    sgd_text = SGDClassifier(max_iter=1, random_state=42)
    for i in range(0, len(texts), 50):
        chunk = texts[i:i+50]
        X_chunk = hv.transform(chunk)
        y_chunk = np.array([0, 1, 0] * (len(chunk)//3))[:len(chunk)]
        if len(y_chunk) < len(chunk):
            y_chunk = np.resize(y_chunk, len(chunk))
        sgd_text.partial_fit(X_chunk, y_chunk, classes=np.array([0, 1]))
    print(f"    Partial fit on text chunks: OK")

    print("\n[3] MiniBatchKMeans:")
    X_k = np.random.randn(5000, 10)
    mbk = MiniBatchKMeans(n_clusters=5, batch_size=500, random_state=42)
    mbk.fit(X_k)
    print(f"    Inertia: {mbk.inertia_:.2f}")
    print(f"    Labels shape: {mbk.predict(X_k[:10]).shape}")

    print("\n[4] IncrementalPCA:")
    ipca = IncrementalPCA(n_components=5, batch_size=500)
    for i in range(0, len(X_k), 500):
        ipca.partial_fit(X_k[i:i+500])
    X_trans = ipca.transform(X_k)
    print(f"    Transformed shape: {X_trans.shape}")
    print(f"    Explained variance: {ipca.explained_variance_ratio_.sum():.4f}")

    print("\n[5] Summary of incremental estimators:")
    print("    - SGDClassifier, SGDRegressor: partial_fit")
    print("    - MiniBatchKMeans: fit with batches")
    print("    - IncrementalPCA: partial_fit")
    print("    - MultinomialNB: partial_fit")
    print("    - HashingVectorizer: no fit, transform only")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
