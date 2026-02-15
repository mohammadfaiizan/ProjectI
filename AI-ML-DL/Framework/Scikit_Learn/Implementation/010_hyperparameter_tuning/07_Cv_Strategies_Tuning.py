"""
Different CV strategies in search: KFold, StratifiedKFold, GroupKFold
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    GroupKFold,
    train_test_split,
)
from sklearn.svm import SVC


def main():
    print("=" * 60)
    print("CV strategies in hyperparameter search")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {"C": [0.1, 1.0], "gamma": ["scale", 0.01]}

    print("\n[1] KFold (default for regression-like):")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_kf = GridSearchCV(SVC(random_state=42), param_grid, cv=kf)
    grid_kf.fit(X_train, y_train)
    print(f"    best_score_: {grid_kf.best_score_:.4f}")

    print("\n[2] StratifiedKFold (preserve class proportions):")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_skf = GridSearchCV(SVC(random_state=42), param_grid, cv=skf)
    grid_skf.fit(X_train, y_train)
    print(f"    best_score_: {grid_skf.best_score_:.4f}")

    print("\n[3] GroupKFold (groups must not split across train/test):")
    groups = np.array([i // 10 for i in range(len(X_train))])
    gkf = GroupKFold(n_splits=5)
    grid_gkf = GridSearchCV(SVC(random_state=42), param_grid, cv=gkf)
    grid_gkf.fit(X_train, y_train, groups=groups)
    print(f"    best_score_: {grid_gkf.best_score_:.4f}")

    print("\n[4] cv as integer (StratifiedKFold for classifier):")
    grid_int = GridSearchCV(SVC(random_state=42), param_grid, cv=5)
    grid_int.fit(X_train, y_train)
    print(f"    cv=5 uses StratifiedKFold for SVC")
    print(f"    best_score_: {grid_int.best_score_:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
