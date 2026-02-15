"""
Multiple metrics in GridSearchCV, refit with callable
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


def main():
    print("=" * 60)
    print("Multi-metric search and refit callable")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01]}

    print("\n[1] Multiple scoring metrics:")
    grid = GridSearchCV(
        SVC(random_state=42),
        param_grid,
        cv=3,
        scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        refit="accuracy",
    )
    grid.fit(X_train, y_train)

    print("\n[2] cv_results_ keys for multi-metric:")
    metric_keys = [k for k in grid.cv_results_.keys() if k.startswith("mean_test_")]
    print(f"    {metric_keys}")

    print("\n[3] Best index per metric:")
    for metric in ["accuracy", "precision_macro", "recall_macro"]:
        key = f"mean_test_{metric}"
        if key in grid.cv_results_:
            idx = np.argmax(grid.cv_results_[key])
            print(f"    {metric}: best_index={idx}, score={grid.cv_results_[key][idx]:.4f}")

    print("\n[4] refit with callable (e.g., balance accuracy and recall):")
    def combined_refit(cv_results):
        acc = cv_results["mean_test_accuracy"]
        rec = cv_results["mean_test_recall_macro"]
        return np.argmax(0.6 * acc + 0.4 * rec)

    grid_callable = GridSearchCV(
        SVC(random_state=42),
        param_grid,
        cv=3,
        scoring=["accuracy", "recall_macro"],
        refit=combined_refit,
    )
    grid_callable.fit(X_train, y_train)
    print(f"    Best params from callable refit: {grid_callable.best_params_}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
