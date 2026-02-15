"""
Scikit-learn Multi-Metric Evaluation: cross_validate with multiple metrics, refit strategies
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def main():
    print("=" * 60)
    print("Multi-Metric Evaluation and Refit Strategies")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[1] cross_validate with multiple metrics:")
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    results = cross_validate(clf, X, y, cv=cv, scoring=scoring)
    print("    Keys:", list(results.keys()))
    for k in scoring:
        key = f"test_{k}"
        if key in results:
            print(f"    {k}: mean={results[key].mean():.4f}")

    print("\n[2] cross_validate with return_train_score:")
    results = cross_validate(clf, X, y, cv=cv, scoring=scoring, return_train_score=True)
    train_keys = [k for k in results.keys() if k.startswith("train_")]
    print("    Train keys:", train_keys)

    print("\n[3] GridSearchCV with refit to single metric:")
    param_grid = {"C": [0.1, 1.0, 10.0]}
    gs = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=cv, scoring="accuracy", refit=True)
    gs.fit(X, y)
    print(f"    Best params: {gs.best_params_}")
    print(f"    Best score: {gs.best_score_:.4f}")

    print("\n[4] GridSearchCV with refit to multiple metrics (refit=str):")
    gs_multi = GridSearchCV(
        LogisticRegression(random_state=42), param_grid, cv=cv,
        scoring={"accuracy": "accuracy", "f1": "f1_macro"},
        refit="f1",
    )
    gs_multi.fit(X, y)
    print(f"    Best params (refit=f1): {gs_multi.best_params_}")
    print(f"    Best f1: {gs_multi.best_score_:.4f}")
    print(f"    cv_results keys (sample): {list(gs_multi.cv_results_.keys())[:5]}")

    print("\n[5] Accessing all cv_results for multi-metric GridSearchCV:")
    for metric in ["accuracy", "f1"]:
        mean_key = f"mean_test_{metric}"
        if mean_key in gs_multi.cv_results_:
            print(f"    {metric}: {gs_multi.cv_results_[mean_key]}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
