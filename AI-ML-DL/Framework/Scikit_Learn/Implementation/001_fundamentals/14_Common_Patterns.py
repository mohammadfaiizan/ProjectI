"""
Scikit-learn Common Patterns
Fit-predict, fit-transform, method chaining, warm_start
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris


def main():
    print("=" * 60)
    print("Common Patterns: fit-predict, fit-transform, chaining")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)

    print("\n[1] Fit-predict pattern (supervised):")
    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X[:100], y[:100])
    y_pred = clf.predict(X[100:])
    print(f"    fit(X_train, y_train) -> predict(X_test)")
    print(f"    Predictions count: {len(y_pred)}")

    print("\n[2] Fit-transform pattern (unsupervised/transformers):")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"    fit_transform(X) = fit(X) + transform(X)")
    print(f"    X_scaled mean: {X_scaled.mean():.6f}")

    print("\n[3] Transform only (after fit):")
    X_new = np.array([[5.1, 3.5, 1.4, 0.2]])
    X_new_scaled = scaler.transform(X_new)
    print(f"    transform(X_new) uses fitted parameters")
    print(f"    No refitting needed")

    print("\n[4] Method chaining:")
    clf2 = LogisticRegression(max_iter=200).set_params(C=0.5).fit(X[:100], y[:100])
    print(f"    .set_params().fit() chains")
    print(f"    clf2.C: {clf2.C}")

    print("\n[5] warm_start - incremental fitting:")
    sgd = LogisticRegression(max_iter=1, warm_start=True, random_state=42)
    for _ in range(5):
        sgd.fit(X[:100], y[:100])
    print(f"    warm_start=True reuses previous solution")
    print(f"    Useful for partial_fit-style iteration")

    print("\n[6] fit_predict (clustering):")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    print(f"    fit_predict = fit + predict in one call")
    print(f"    Labels (first 10): {labels[:10]}")

    print("\n[7] Pipeline chaining:")
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    pipe.fit(X[:100], y[:100]).predict(X[100:105])
    print("    pipe.fit().predict() - full chain")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
