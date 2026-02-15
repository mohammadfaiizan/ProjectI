"""
Scikit-learn cross_val_score and cross_validate: return_train_score, return_estimator
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression


def main():
    print("=" * 60)
    print("cross_val_score and cross_validate")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[1] cross_val_score (single metric):")
    scores = cross_val_score(clf, X, y, cv=cv)
    print(f"    Scores: {scores}")
    print(f"    Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")

    print("\n[2] cross_val_score with scoring parameter:")
    scores_precision = cross_val_score(clf, X, y, cv=cv, scoring="precision_macro")
    scores_recall = cross_val_score(clf, X, y, cv=cv, scoring="recall_macro")
    print(f"    Precision (macro): {scores_precision}")
    print(f"    Recall (macro): {scores_recall}")

    print("\n[3] cross_validate (multiple metrics):")
    results = cross_validate(clf, X, y, cv=cv, scoring=["accuracy", "precision_macro", "recall_macro"])
    print("    Keys:", list(results.keys()))
    for k, v in results.items():
        if hasattr(v, "mean"):
            print(f"    {k}: mean={v.mean():.4f}")

    print("\n[4] cross_validate with return_train_score=True:")
    results = cross_validate(clf, X, y, cv=cv, return_train_score=True)
    print("    test_score:", results["test_score"])
    print("    train_score:", results["train_score"])
    print(f"    Train mean: {results['train_score'].mean():.4f}")
    print(f"    Test mean:  {results['test_score'].mean():.4f}")

    print("\n[5] cross_validate with return_estimator=True:")
    results = cross_validate(clf, X, y, cv=cv, return_estimator=True)
    estimators = results["estimator"]
    print(f"    Number of estimators: {len(estimators)}")
    print(f"    First estimator coef_ shape: {estimators[0].coef_.shape}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
