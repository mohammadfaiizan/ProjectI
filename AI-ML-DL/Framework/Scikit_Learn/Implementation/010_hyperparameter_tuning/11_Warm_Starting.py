"""
Warm starting: warm_start for incremental training
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier


def main():
    print("=" * 60)
    print("Warm starting for incremental training")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[1] warm_start=True: fit() adds trees instead of replacing:")
    rf = RandomForestClassifier(n_estimators=10, warm_start=True, random_state=42)
    rf.fit(X_train, y_train)
    print(f"    After first fit: n_estimators={rf.n_estimators}")

    rf.n_estimators = 20
    rf.fit(X_train, y_train)
    print(f"    After second fit (n_estimators=20): {rf.n_estimators} trees")

    print("\n[2] GridSearchCV with warm_start (n_estimators as param):")
    param_grid = {"n_estimators": [10, 30, 50], "max_depth": [3, 5]}
    grid = GridSearchCV(
        RandomForestClassifier(warm_start=True, random_state=42),
        param_grid,
        cv=3,
    )
    grid.fit(X_train, y_train)
    print(f"    best_params_: {grid.best_params_}")

    print("\n[3] Caveat: warm_start in search resets between candidates:")
    print("    Each param combo trains from scratch; warm_start helps")
    print("    when manually iterating n_estimators on same model")

    print("\n[4] Manual warm start pattern (incremental n_estimators):")
    best_score = 0
    best_n = 0
    rf_ws = RandomForestClassifier(n_estimators=5, warm_start=True, random_state=42)
    for n in [5, 10, 20, 40]:
        rf_ws.n_estimators = n
        rf_ws.fit(X_train, y_train)
        score = rf_ws.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_n = n
        print(f"    n_estimators={n}: test_score={score:.4f}")
    print(f"    Best: n_estimators={best_n}, score={best_score:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
