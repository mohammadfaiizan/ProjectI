"""
Scikit-learn IsolationForest: n_estimators, contamination, decision_function
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs


def main():
    print("=" * 60)
    print("Isolation Forest Anomaly Detection")
    print("=" * 60)

    X, _ = make_blobs(n_samples=300, n_features=2, centers=1, random_state=42)
    X_outliers = np.random.uniform(low=-10, high=10, size=(20, 2))
    X = np.vstack([X, X_outliers])

    print("\n[1] IsolationForest basic usage:")
    iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    pred = iso.fit_predict(X)
    scores = iso.decision_function(X)
    n_anomalies = (pred == -1).sum()
    print(f"    Anomalies predicted: {n_anomalies}")
    print(f"    decision_function range: [{scores.min():.2f}, {scores.max():.2f}]")

    print("\n[2] contamination effect:")
    for cont in [0.05, 0.1, 0.2]:
        iso = IsolationForest(n_estimators=100, contamination=cont, random_state=42)
        pred = iso.fit_predict(X)
        print(f"    contamination={cont}: anomalies={(pred==-1).sum()}")

    print("\n[3] n_estimators effect:")
    for n in [10, 50, 100, 200]:
        iso = IsolationForest(n_estimators=n, contamination=0.1, random_state=42)
        iso.fit(X)
        print(f"    n_estimators={n}: avg_score={iso.decision_function(X).mean():.4f}")

    print("\n[4] score_samples (negative log anomaly score):")
    neg_scores = iso.score_samples(X)
    print(f"    score_samples range: [{neg_scores.min():.2f}, {neg_scores.max():.2f}]")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
