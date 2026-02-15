"""
Scikit-learn LocalOutlierFactor: n_neighbors, novelty, negative_outlier_factor_
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs


def main():
    print("=" * 60)
    print("Local Outlier Factor (LOF)")
    print("=" * 60)

    X, _ = make_blobs(n_samples=300, n_features=2, centers=1, random_state=42)
    X_outliers = np.random.uniform(low=-10, high=10, size=(20, 2))
    X = np.vstack([X, X_outliers])

    print("\n[1] LOF basic (unsupervised):")
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    pred = lof.fit_predict(X)
    n_anomalies = (pred == -1).sum()
    print(f"    Anomalies: {n_anomalies}")
    print(f"    negative_outlier_factor_ (first 5): {lof.negative_outlier_factor_[:5]}")

    print("\n[2] n_neighbors effect:")
    for k in [5, 10, 20, 50]:
        lof = LocalOutlierFactor(n_neighbors=k, contamination=0.1)
        pred = lof.fit_predict(X)
        print(f"    n_neighbors={k}: anomalies={(pred==-1).sum()}")

    print("\n[3] novelty=True (fit on train, predict on test):")
    X_train, X_test = X[:250], X[250:]
    lof_novelty = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
    lof_novelty.fit(X_train)
    pred_test = lof_novelty.predict(X_test)
    print(f"    Test anomalies: {(pred_test==-1).sum()}")

    print("\n[4] decision_function (novelty mode):")
    scores = lof_novelty.decision_function(X_test)
    print(f"    decision_function range: [{scores.min():.2f}, {scores.max():.2f}]")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
