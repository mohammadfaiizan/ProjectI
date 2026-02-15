"""
Scikit-learn TruncatedSVD (sparse) and NMF (non-negative)
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF
from scipy.sparse import csr_matrix


def main():
    print("=" * 60)
    print("TruncatedSVD and NMF")
    print("=" * 60)

    print("\n[1] TruncatedSVD on dense data (sparse-friendly):")
    X_dense = np.random.rand(100, 50)
    svd = TruncatedSVD(n_components=10, random_state=42)
    X_svd = svd.fit_transform(X_dense)
    print(f"    Transformed shape: {X_svd.shape}")
    print(f"    explained_variance_ratio_ sum: {svd.explained_variance_ratio_.sum():.4f}")

    print("\n[2] TruncatedSVD on sparse matrix:")
    X_sparse = csr_matrix(X_dense)
    svd.fit(X_sparse)
    X_t = svd.transform(X_sparse)
    print(f"    Sparse transform shape: {X_t.shape}")

    print("\n[3] NMF (non-negative data):")
    X_pos = np.abs(np.random.rand(100, 50)) + 0.1
    nmf = NMF(n_components=10, random_state=42)
    W = nmf.fit_transform(X_pos)
    H = nmf.components_
    print(f"    W shape: {W.shape}, H shape: {H.shape}")
    print(f"    reconstruction_err_: {nmf.reconstruction_err_:.4f}")

    print("\n[4] NMF init options:")
    for init in ["nndsvda", "random"]:
        nmf = NMF(n_components=5, init=init, random_state=42)
        nmf.fit(X_pos)
        print(f"    init='{init}': err={nmf.reconstruction_err_:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
