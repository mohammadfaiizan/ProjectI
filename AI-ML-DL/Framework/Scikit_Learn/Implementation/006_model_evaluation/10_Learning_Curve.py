"""
Scikit-learn Learning Curve: learning_curve, LearningCurveDisplay
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve, LearningCurveDisplay
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("Learning Curve")
    print("=" * 60)

    X, y = load_digits(return_X_y=True)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(gamma=0.001, random_state=42)),
    ])

    print("\n[1] learning_curve:")
    train_sizes, train_scores, test_scores = learning_curve(
        pipe, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
    )
    print(f"    train_sizes: {train_sizes}")
    print(f"    train_scores shape: {train_scores.shape}")
    print(f"    test_scores shape: {test_scores.shape}")

    print("\n[2] Mean train/test scores per size:")
    for i, size in enumerate(train_sizes):
        tr_mean = train_scores[i].mean()
        te_mean = test_scores[i].mean()
        print(f"    n={int(size)}: train={tr_mean:.4f}, test={te_mean:.4f}")

    print("\n[3] LearningCurveDisplay.from_estimator:")
    fig, ax = plt.subplots()
    LearningCurveDisplay.from_estimator(pipe, X, y, cv=5, ax=ax, n_jobs=-1)
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=100)
    plt.close()
    print("    Saved learning_curve.png")

    print("\n[4] With different train_sizes:")
    train_sizes, train_scores, test_scores = learning_curve(
        pipe, X, y, cv=3, train_sizes=[50, 200, 500, 1000, 1500],
    )
    print(f"    train_sizes: {train_sizes}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
