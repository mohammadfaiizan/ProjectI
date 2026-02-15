"""
Search results analysis: cv_results_, best_params_, best_estimator_, visualizing
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


def main():
    print("=" * 60)
    print("Search results analysis: cv_results_, best_params_, best_estimator_")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.1], "kernel": ["rbf", "linear"]}
    grid = GridSearchCV(SVC(random_state=42), param_grid, cv=3, return_train_score=True)
    grid.fit(X_train, y_train)

    print("\n[1] cv_results_ keys:")
    print(f"    {list(grid.cv_results_.keys())[:10]}...")

    print("\n[2] best_params_ and best_score_:")
    print(f"    best_params_: {grid.best_params_}")
    print(f"    best_score_: {grid.best_score_:.4f}")

    print("\n[3] best_estimator_ (fitted model):")
    best_est = grid.best_estimator_
    print(f"    Type: {type(best_est).__name__}")
    print(f"    n_support_: {best_est.n_support_}")

    print("\n[4] Top 3 parameter combinations by mean_test_score:")
    idx = np.argsort(grid.cv_results_["mean_test_score"])[::-1][:3]
    for i, j in enumerate(idx):
        params = grid.cv_results_["params"][j]
        score = grid.cv_results_["mean_test_score"][j]
        std = grid.cv_results_["std_test_score"][j]
        print(f"    #{i+1}: {params} -> {score:.4f} (+/- {std:.4f})")

    print("\n[5] Train vs test (overfitting check):")
    best_idx = grid.best_index_
    train_mean = grid.cv_results_["mean_train_score"][best_idx]
    test_mean = grid.cv_results_["mean_test_score"][best_idx]
    print(f"    mean_train_score: {train_mean:.4f}")
    print(f"    mean_test_score:  {test_mean:.4f}")
    print(f"    Gap: {(train_mean - test_mean):.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
