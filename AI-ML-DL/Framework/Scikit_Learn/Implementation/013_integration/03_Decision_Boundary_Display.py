"""
Scikit-learn DecisionBoundaryDisplay.from_estimator for 2D decision boundaries
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def main():
    print("=" * 60)
    print("DecisionBoundaryDisplay.from_estimator")
    print("=" * 60)

    print("\n[1] Create 2D classification data:")
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                              n_informative=2, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    print(f"    Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    print("\n[2] SVC decision boundary display:")
    clf = SVC(kernel="rbf", C=1.0, random_state=42)
    clf.fit(X_train, y_train)
    disp = DecisionBoundaryDisplay.from_estimator(
        clf, X_train, response_method="predict",
        xlabel="Feature 0", ylabel="Feature 1", alpha=0.5
    )
    disp.ax_.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors="k")
    print(f"    Display created: {disp is not None}")
    print(f"    Decision function shape: {clf.decision_function(X_train).shape}")

    print("\n[3] LogisticRegression with response_method='predict_proba':")
    lr = LogisticRegression(max_iter=200, random_state=42)
    lr.fit(X_train, y_train)
    disp2 = DecisionBoundaryDisplay.from_estimator(
        lr, X_train, response_method="predict_proba",
        n_cols=2, grid_resolution=50
    )
    print(f"    Proba display: {len(disp2)} subplots")

    print("\n[4] Pipeline with DecisionBoundaryDisplay:")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", random_state=42))
    ])
    pipe.fit(X_train, y_train)
    disp3 = DecisionBoundaryDisplay.from_estimator(
        pipe, X_train, response_method="predict"
    )
    print(f"    Pipeline display: {disp3.ax_ is not None}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
