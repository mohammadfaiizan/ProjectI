"""
Scikit-learn pipeline cross-validation: cross_val_score with Pipeline
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.datasets import load_iris
from sklearn.metrics import make_scorer, accuracy_score


def main():
    print("=" * 60)
    print("Pipeline Cross-Validation: cross_val_score")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])

    print("\n[1] cross_val_score with pipeline:")
    scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    print(f"    CV scores: {np.round(scores, 4)}")
    print(f"    Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    print("\n[2] cross_validate with multiple metrics:")
    scoring = ["accuracy", "balanced_accuracy"]
    results = cross_validate(pipe, X, y, cv=5, scoring=scoring)
    print(f"    test_accuracy: {np.round(results['test_accuracy'], 4)}")
    print(f"    test_balanced_accuracy: {np.round(results['test_balanced_accuracy'], 4)}")

    print("\n[3] StratifiedKFold with pipeline:")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_skf = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
    print(f"    StratifiedKFold scores: {np.round(scores_skf, 4)}")

    print("\n[4] No data leakage (preprocessing inside CV):")
    print("    Each fold: fit scaler/PCA on train, transform val")
    print("    Prevents test data from influencing scaling/PCA")

    print("\n[5] Pipeline vs raw estimator in CV:")
    scores_pipe = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    scores_raw = cross_val_score(
        LogisticRegression(max_iter=500, random_state=42), X, y, cv=5, scoring="accuracy"
    )
    print(f"    Pipeline mean: {scores_pipe.mean():.4f}")
    print(f"    Raw (no scaling) mean: {scores_raw.mean():.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
