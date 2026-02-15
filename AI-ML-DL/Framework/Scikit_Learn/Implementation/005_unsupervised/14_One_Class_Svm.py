"""
Scikit-learn OneClassSVM: kernel, nu, gamma
"""

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


def main():
    print("=" * 60)
    print("One-Class SVM Anomaly Detection")
    print("=" * 60)

    X, _ = make_blobs(n_samples=300, n_features=2, centers=1, random_state=42)
    X_outliers = np.random.uniform(low=-10, high=10, size=(20, 2))
    X = np.vstack([X, X_outliers])
    X = StandardScaler().fit_transform(X)

    print("\n[1] OneClassSVM basic usage:")
    ocsvm = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
    pred = ocsvm.fit_predict(X)
    n_anomalies = (pred == -1).sum()
    print(f"    Anomalies: {n_anomalies}")
    print(f"    n_support_: {ocsvm.n_support_}")

    print("\n[2] nu effect (upper bound on outliers):")
    for nu in [0.01, 0.1, 0.2, 0.5]:
        ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
        pred = ocsvm.fit_predict(X)
        print(f"    nu={nu}: anomalies={(pred==-1).sum()}")

    print("\n[3] kernel options:")
    for kernel in ["rbf", "linear", "poly"]:
        ocsvm = OneClassSVM(kernel=kernel, nu=0.1, gamma="scale")
        pred = ocsvm.fit_predict(X)
        print(f"    kernel='{kernel}': anomalies={(pred==-1).sum()}")

    print("\n[4] decision_function:")
    scores = ocsvm.decision_function(X)
    print(f"    decision_function range: [{scores.min():.2f}, {scores.max():.2f}]")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
