"""
Scikit-learn Cross-Validation: KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold
"""

import numpy as np
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    cross_val_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression


def main():
    print("=" * 60)
    print("KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold")
    print("=" * 60)

    X_clf, y_clf = load_iris(return_X_y=True)
    X_reg, y_reg = make_regression(n_samples=100, n_features=5, random_state=42)

    print("\n[1] KFold (regression):")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(LinearRegression(), X_reg, y_reg, cv=kf, scoring="r2")
    print(f"    Splits: 5, Scores: {scores}")
    print(f"    Mean R2: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    print("\n[2] KFold split indices:")
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for i, (train_idx, test_idx) in enumerate(kf.split(X_reg)):
        print(f"    Fold {i}: train={len(train_idx)}, test={len(test_idx)}")

    print("\n[3] StratifiedKFold (classification):")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(LogisticRegression(random_state=42), X_clf, y_clf, cv=skf)
    print(f"    Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    print("\n[4] RepeatedKFold:")
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    scores = cross_val_score(LinearRegression(), X_reg, y_reg, cv=rkf, scoring="r2")
    print(f"    5 splits x 3 repeats = {len(scores)} scores")
    print(f"    Mean R2: {scores.mean():.4f}")

    print("\n[5] RepeatedStratifiedKFold:")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    scores = cross_val_score(LogisticRegression(random_state=42), X_clf, y_clf, cv=rskf)
    print(f"    5 splits x 2 repeats = {len(scores)} scores")
    print(f"    Mean accuracy: {scores.mean():.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
