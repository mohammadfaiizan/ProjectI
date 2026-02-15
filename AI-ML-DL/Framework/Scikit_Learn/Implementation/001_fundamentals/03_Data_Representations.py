"""
Scikit-learn Data Representations
Feature matrix X (2D), target vector y (1D), sparse matrices
"""

import numpy as np
from scipy import sparse
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def main():
    print("=" * 60)
    print("Data Representations: X, y, Sparse Matrices")
    print("=" * 60)

    print("\n[1] Feature matrix X - 2D array (n_samples x n_features):")
    iris = load_iris()
    X = iris.data
    y = iris.target
    print(f"    X shape: {X.shape}")
    print(f"    X dtype: {X.dtype}")
    print(f"    First row: {X[0]}")

    print("\n[2] Target vector y - 1D array (n_samples,):")
    print(f"    y shape: {y.shape}")
    print(f"    y unique values: {np.unique(y)}")
    print(f"    First 5 targets: {y[:5]}")

    print("\n[3] Dense 2D NumPy array:")
    X_dense = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"    Shape: {X_dense.shape}")
    print(f"    ndim: {X_dense.ndim}")

    print("\n[4] Sparse matrix - CSR format:")
    X_sparse = sparse.csr_matrix([[0, 1, 0], [2, 0, 3], [0, 0, 4]])
    print(f"    Type: {type(X_sparse)}")
    print(f"    Shape: {X_sparse.shape}")
    print(f"    nnz (non-zero elements): {X_sparse.nnz}")
    print(f"    To array:\n{X_sparse.toarray()}")

    print("\n[5] Estimator with dense data:")
    clf_dense = LogisticRegression(max_iter=200, random_state=42)
    clf_dense.fit(X[:100], y[:100])
    print(f"    LogisticRegression accepts dense X: OK")

    print("\n[6] Estimator with sparse data:")
    X_sparse_large = sparse.csr_matrix(np.random.rand(100, 50) > 0.9)
    from sklearn.linear_model import SGDClassifier
    sgd = SGDClassifier(random_state=42, max_iter=5)
    sgd.fit(X_sparse_large, np.random.randint(0, 2, 100))
    print(f"    SGDClassifier with sparse X: OK")

    print("\n[7] Row-major (C-order) convention:")
    print(f"    X is stored as samples in rows, features in columns")
    print(f"    X[0] gives first sample's features")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
