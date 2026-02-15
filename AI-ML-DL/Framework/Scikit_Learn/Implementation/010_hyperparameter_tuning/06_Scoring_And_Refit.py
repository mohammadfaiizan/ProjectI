"""
Scoring and refit: scoring parameter, refit strategies, callable refit
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score


def main():
    print("=" * 60)
    print("Scoring and refit strategies")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01]}

    print("\n[1] scoring='accuracy' (default for classification):")
    grid = GridSearchCV(SVC(random_state=42), param_grid, cv=3, scoring="accuracy")
    grid.fit(X_train, y_train)
    print(f"    best_score_: {grid.best_score_:.4f}")

    print("\n[2] scoring='f1_macro' for imbalanced classes:")
    grid_f1 = GridSearchCV(SVC(random_state=42), param_grid, cv=3, scoring="f1_macro")
    grid_f1.fit(X_train, y_train)
    print(f"    best_score_: {grid_f1.best_score_:.4f}")

    print("\n[3] refit with specific metric when using multiple:")
    grid_multi = GridSearchCV(
        SVC(random_state=42),
        param_grid,
        cv=3,
        scoring=["accuracy", "f1_macro", "recall_macro"],
        refit="f1_macro",
    )
    grid_multi.fit(X_train, y_train)
    print(f"    Refit on: f1_macro")
    print(f"    best_params_: {grid_multi.best_params_}")

    print("\n[4] refit with callable (e.g., custom metric):")
    def neg_f1_macro(y_true, y_pred):
        return -f1_score(y_true, y_pred, average="macro")

    custom_scorer = make_scorer(neg_f1_macro, greater_is_better=False)
    grid_custom = GridSearchCV(SVC(random_state=42), param_grid, cv=3, scoring=custom_scorer)
    grid_custom.fit(X_train, y_train)
    print(f"    Custom scorer best_score_: {grid_custom.best_score_:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
