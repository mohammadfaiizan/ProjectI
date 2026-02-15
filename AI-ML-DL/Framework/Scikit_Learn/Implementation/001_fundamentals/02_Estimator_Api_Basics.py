"""
Scikit-learn Estimator API Basics
Demonstrates fit(), predict(), transform(), score() interface
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris


def main():
    print("=" * 60)
    print("Estimator API Basics: fit, predict, transform, score")
    print("=" * 60)

    print("\n[1] fit() and predict() - Regression:")
    X_reg = np.array([[1], [2], [3], [4], [5]])
    y_reg = np.array([2, 4, 6, 8, 10])
    reg = LinearRegression()
    reg.fit(X_reg, y_reg)
    predictions = reg.predict([[6], [7]])
    print(f"    fit() called on 5 samples")
    print(f"    predict([[6], [7]]) = {predictions}")

    print("\n[2] fit() and predict() - Classification:")
    X, y = load_iris(return_X_y=True)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X[:100], y[:100])
    pred_class = clf.predict(X[100:105])
    print(f"    fit() on 100 samples, predict on 5 samples")
    print(f"    Predictions: {pred_class}")

    print("\n[3] transform() - Preprocessing:")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reg)
    print(f"    fit_transform() on X_reg")
    print(f"    Mean after scaling: {X_scaled.mean():.6f}")
    print(f"    Std after scaling: {X_scaled.std():.6f}")
    X_new_scaled = scaler.transform([[6], [7]])
    print(f"    transform([[6], [7]]) = {X_new_scaled.flatten()}")

    print("\n[4] score() - Regression:")
    reg_score = reg.score(X_reg, y_reg)
    print(f"    R^2 score on training data: {reg_score:.4f}")

    print("\n[5] score() - Classification:")
    clf_score = clf.score(X[:100], y[:100])
    print(f"    Accuracy on training data: {clf_score:.4f}")

    print("\n[6] fit_predict() - Convenience method:")
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X[:100])
    print(f"    KMeans fit_predict() labels (first 10): {labels[:10]}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
