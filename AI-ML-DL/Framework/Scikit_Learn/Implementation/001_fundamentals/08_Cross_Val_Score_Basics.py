"""
Scikit-learn Cross-Validation Basics
cross_val_score, cross_validate, cv parameter
"""

import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def main():
    print("=" * 60)
    print("Cross-Validation: cross_val_score, cross_validate")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(max_iter=200, random_state=42)

    print("\n[1] cross_val_score - single metric:")
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"    cv=5 (5-fold): {scores}")
    print(f"    Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")

    print("\n[2] cross_val_score with scoring parameter:")
    scores_f1 = cross_val_score(clf, X, y, cv=5, scoring="f1_macro")
    print(f"    scoring='f1_macro': {scores_f1}")
    print(f"    Mean F1: {scores_f1.mean():.4f}")

    print("\n[3] cross_validate - multiple metrics:")
    scoring = ["accuracy", "f1_macro", "precision_macro"]
    results = cross_validate(clf, X, y, cv=5, scoring=scoring)
    print(f"    Keys: {list(results.keys())}")
    print(f"    test_accuracy: {results['test_accuracy']}")
    print(f"    test_f1_macro: {results['test_f1_macro']}")
    print(f"    fit_time (sec): {results['fit_time']}")

    print("\n[4] cv parameter - integer vs splitter:")
    scores_k3 = cross_val_score(clf, X, y, cv=3)
    print(f"    cv=3: {scores_k3}")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_kf = cross_val_score(clf, X, y, cv=kf)
    print(f"    cv=KFold(5): {scores_kf}")

    print("\n[5] return_train_score in cross_validate:")
    results = cross_validate(clf, X, y, cv=3, return_train_score=True)
    print(f"    train_score: {results['train_score']}")
    print(f"    test_score: {results['test_score']}")

    print("\n[6] n_jobs for parallelization:")
    scores_par = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
    print(f"    n_jobs=-1 uses all CPUs")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
