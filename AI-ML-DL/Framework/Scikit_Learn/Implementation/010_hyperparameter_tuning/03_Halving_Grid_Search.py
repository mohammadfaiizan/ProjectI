"""
Scikit-learn HalvingGridSearchCV: factor, resource, min_resources
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier


def main():
    print("=" * 60)
    print("HalvingGridSearchCV: factor, resource, min_resources")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 5, 10],
        "min_samples_split": [2, 5],
    }

    print("\n[1] HalvingGridSearchCV with factor=3:")
    halving_grid = HalvingGridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        factor=3,
        resource="n_samples",
        min_resources="exhaust",
        cv=3,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )
    halving_grid.fit(X_train, y_train)

    print(f"    best_params_: {halving_grid.best_params_}")
    print(f"    best_score_: {halving_grid.best_score_:.4f}")
    print(f"    n_iterations_: {halving_grid.n_iterations_}")

    print("\n[2] Iteration resources (samples per iteration):")
    for i, r in enumerate(halving_grid.n_resources_):
        print(f"    Iteration {i}: {r} samples")

    print("\n[3] Candidates per iteration:")
    print(f"    n_candidates_: {halving_grid.n_candidates_}")

    print("\n[4] Compare to full grid size:")
    full_combinations = 3 * 3 * 2
    print(f"    Full grid would evaluate: {full_combinations} combinations")
    print(f"    Halving evaluated fewer in early iterations")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
