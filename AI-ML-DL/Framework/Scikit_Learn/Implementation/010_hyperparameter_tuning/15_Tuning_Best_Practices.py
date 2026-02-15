"""
Tuning best practices: strategies, overfitting to validation, nested CV
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.svm import SVC


def main():
    print("=" * 60)
    print("Tuning best practices: strategies, nested CV")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[1] Tuning strategy: coarse-to-fine grid:")
    param_coarse = {"C": [0.1, 1.0, 100.0], "gamma": ["scale"]}
    grid1 = GridSearchCV(SVC(random_state=42), param_coarse, cv=3)
    grid1.fit(X_train, y_train)
    best_c = grid1.best_params_["C"]
    print(f"    Coarse best C: {best_c}")

    if best_c == 0.1:
        param_fine = {"C": [0.01, 0.05, 0.1, 0.5], "gamma": ["scale"]}
    elif best_c == 100.0:
        param_fine = {"C": [50.0, 100.0, 200.0], "gamma": ["scale"]}
    else:
        param_fine = {"C": [0.5, 1.0, 2.0], "gamma": ["scale"]}
    grid2 = GridSearchCV(SVC(random_state=42), param_fine, cv=3)
    grid2.fit(X_train, y_train)
    print(f"    Fine best C: {grid2.best_params_['C']}")

    print("\n[2] Overfitting to validation: train vs test gap:")
    grid = GridSearchCV(SVC(random_state=42), {"C": [0.1, 1.0, 10.0]}, cv=3, return_train_score=True)
    grid.fit(X_train, y_train)
    best_idx = grid.best_index_
    gap = grid.cv_results_["mean_train_score"][best_idx] - grid.cv_results_["mean_test_score"][best_idx]
    print(f"    Train-test gap for best: {gap:.4f}")
    print("    Large gap -> overfitting to validation")

    print("\n[3] Nested CV for unbiased performance estimate:")
    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=43)
    outer_scores = []
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_in, X_out = X[train_idx], X[test_idx]
        y_in, y_out = y[train_idx], y[test_idx]
        grid_inner = GridSearchCV(SVC(random_state=42), {"C": [0.1, 1.0, 10.0]}, cv=inner_cv)
        grid_inner.fit(X_in, y_in)
        score = grid_inner.score(X_out, y_out)
        outer_scores.append(score)
        print(f"    Fold {fold+1}: outer test score = {score:.4f}")
    print(f"    Nested CV mean: {np.mean(outer_scores):.4f} (+/- {np.std(outer_scores):.4f})")

    print("\n[4] Best practices summary:")
    print("    - Use RandomizedSearchCV for large spaces")
    print("    - Use Halving* for faster exploration")
    print("    - Nested CV when reporting final performance")
    print("    - Coarse-to-fine for expensive models")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
