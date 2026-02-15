"""
Scikit-learn sparse data: scipy.sparse matrices, sparse-compatible estimators
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("Sparse Data: scipy.sparse, sparse-compatible estimators")
    print("=" * 60)

    print("\n[1] Create sparse matrix from dense:")
    X_dense = np.random.rand(100, 50)
    X_dense[X_dense < 0.9] = 0
    X_sparse = csr_matrix(X_dense)
    print(f"    Dense shape: {X_dense.shape}, size: {X_dense.nbytes} bytes")
    print(f"    Sparse shape: {X_sparse.shape}, nnz: {X_sparse.nnz}")

    print("\n[2] TfidfVectorizer returns sparse:")
    docs = ["hello world", "world of python", "python programming"]
    vec = TfidfVectorizer()
    X_tfidf = vec.fit_transform(docs)
    print(f"    Type: {type(X_tfidf).__name__}")
    print(f"    Shape: {X_tfidf.shape}")

    print("\n[3] Sparse-compatible estimators:")
    X, y = load_iris(return_X_y=True)
    X_sparse_iris = csr_matrix(X)
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_sparse_iris, y)
    pred = clf.predict(X_sparse_iris)
    print(f"    LogisticRegression on sparse: {accuracy_score(y, pred):.4f}")

    print("\n[4] LinearSVC and MultinomialNB (native sparse support):")
    lsvc = LinearSVC(random_state=42)
    lsvc.fit(X_tfidf, [0, 1, 0])
    nb = MultinomialNB()
    nb.fit(X_tfidf, [0, 1, 0])
    print(f"    LinearSVC and MultinomialNB accept sparse input")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
