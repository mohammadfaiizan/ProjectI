"""
Scikit-learn GridSearchCV: param_grid, cv, scoring, refit, best_params_, best_score_
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def main():
    print("=" * 60)
    print("GridSearchCV: param_grid, cv, scoring, refit")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(random_state=42)),
    ])

    param_grid = {
        "svc__C": [0.1, 1.0, 10.0],
        "svc__gamma": ["scale", "auto", 0.01, 0.1],
        "svc__kernel": ["rbf", "linear"],
    }

    print("\n[1] GridSearchCV with param_grid and cv=5:")
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        refit=True,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    print(f"    best_params_: {grid.best_params_}")
    print(f"    best_score_: {grid.best_score_:.4f}")
    print(f"    best_index_: {grid.best_index_}")

    print("\n[2] Refit behavior (best model retrained on full train data):")
    pred = grid.predict(X_test)
    print(f"    Test accuracy: {(pred == y_test).mean():.4f}")

    print("\n[3] Access best_estimator_:")
    best_estimator = grid.best_estimator_
    print(f"    Type: {type(best_estimator).__name__}")
    print(f"    Best C: {best_estimator.named_steps['svc'].C}")

    print("\n[4] Total combinations evaluated:")
    print(f"    {len(grid.cv_results_['params'])} parameter combinations")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
