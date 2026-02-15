"""
Scikit-learn Advanced CV: GroupKFold, TimeSeriesSplit, LeaveOneOut, LeavePOut
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import (
    GroupKFold,
    TimeSeriesSplit,
    LeaveOneOut,
    LeavePOut,
    cross_val_score,
)
from sklearn.linear_model import LinearRegression


def main():
    print("=" * 60)
    print("GroupKFold, TimeSeriesSplit, LeaveOneOut, LeavePOut")
    print("=" * 60)

    X, y = make_regression(n_samples=50, n_features=5, random_state=42)
    groups = np.array([i // 10 for i in range(50)])

    print("\n[1] GroupKFold (groups cannot be split):")
    gkf = GroupKFold(n_splits=5)
    scores = cross_val_score(LinearRegression(), X, y, groups=groups, cv=gkf, scoring="r2")
    print(f"    Groups: {np.unique(groups)}")
    print(f"    Scores: {scores}")
    print(f"    Mean R2: {scores.mean():.4f}")

    print("\n[2] TimeSeriesSplit:")
    tscv = TimeSeriesSplit(n_splits=5)
    n_splits = tscv.get_n_splits(X)
    print(f"    n_splits: {n_splits}")
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"    Fold {i}: train={len(train_idx)}, test={len(test_idx)}")

    print("\n[3] LeaveOneOut:")
    loo = LeaveOneOut()
    scores = cross_val_score(LinearRegression(), X[:20], y[:20], cv=loo, scoring="r2")
    print(f"    n_splits: {len(scores)}")
    print(f"    Mean R2: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    print("\n[4] LeavePOut (p=2):")
    lpo = LeavePOut(p=2)
    X_small, y_small = X[:10], y[:10]
    n_splits = lpo.get_n_splits(X_small)
    print(f"    n_splits (10 choose 2): {n_splits}")
    scores = cross_val_score(LinearRegression(), X_small, y_small, cv=lpo, scoring="r2")
    print(f"    Mean R2: {scores.mean():.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
