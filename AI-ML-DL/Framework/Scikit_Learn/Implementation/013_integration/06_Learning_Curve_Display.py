"""
Scikit-learn LearningCurveDisplay and ValidationCurveDisplay
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import LearningCurveDisplay, ValidationCurveDisplay


def main():
    print("=" * 60)
    print("LearningCurveDisplay and ValidationCurveDisplay")
    print("=" * 60)

    print("\n[1] Load digits dataset:")
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    print(f"    Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    print("\n[2] LearningCurveDisplay.from_estimator:")
    clf = SVC(kernel="rbf", gamma=0.001, random_state=42)
    disp = LearningCurveDisplay.from_estimator(
        clf, X_train, y_train,
        train_sizes=[50, 100, 200, 400, 600, 800],
        cv=3, n_jobs=-1
    )
    print(f"    Train sizes used: {disp.train_sizes_}")
    print(f"    Train scores shape: {disp.train_scores_.shape}")
    print(f"    Test scores shape: {disp.test_scores_.shape}")
    print(f"    Final train mean: {disp.train_scores_[-1].mean():.4f}")
    print(f"    Final test mean: {disp.test_scores_[-1].mean():.4f}")

    print("\n[3] ValidationCurveDisplay for gamma:")
    disp2 = ValidationCurveDisplay.from_estimator(
        clf, X_train, y_train,
        param_name="gamma", param_range=[1e-4, 1e-3, 1e-2, 1e-1],
        cv=3, n_jobs=-1
    )
    print(f"    Param range: {disp2.param_range}")
    print(f"    Train scores: {disp2.train_scores_.mean(axis=1)}")
    print(f"    Test scores: {disp2.test_scores_.mean(axis=1)}")

    print("\n[4] ValidationCurveDisplay for C:")
    disp3 = ValidationCurveDisplay.from_estimator(
        clf, X_train, y_train,
        param_name="C", param_range=[0.1, 1, 10, 100],
        cv=3, n_jobs=-1
    )
    best_idx = disp3.test_scores_.mean(axis=1).argmax()
    print(f"    Best C: {disp3.param_range[best_idx]}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
