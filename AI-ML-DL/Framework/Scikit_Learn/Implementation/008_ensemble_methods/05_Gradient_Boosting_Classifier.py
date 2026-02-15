"""
Scikit-learn GradientBoostingClassifier: n_estimators, learning_rate, max_depth
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("GradientBoostingClassifier: n_estimators, learning_rate, max_depth")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[1] n_estimators and learning_rate:")
    for n, lr in [(50, 0.1), (100, 0.1), (200, 0.05)]:
        gb = GradientBoostingClassifier(n_estimators=n, learning_rate=lr, random_state=42)
        gb.fit(X_train, y_train)
        acc = accuracy_score(y_test, gb.predict(X_test))
        print(f"    n={n}, lr={lr}: Accuracy = {acc:.4f}")

    print("\n[2] max_depth:")
    for depth in [2, 4, 6, 10]:
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=depth, random_state=42)
        gb.fit(X_train, y_train)
        acc = accuracy_score(y_test, gb.predict(X_test))
        print(f"    max_depth={depth}: Accuracy = {acc:.4f}")

    print("\n[3] subsample:")
    for ss in [0.5, 0.8, 1.0]:
        gb = GradientBoostingClassifier(n_estimators=100, subsample=ss, random_state=42)
        gb.fit(X_train, y_train)
        acc = accuracy_score(y_test, gb.predict(X_test))
        print(f"    subsample={ss}: Accuracy = {acc:.4f}")

    print("\n[4] predict_proba and staged_predict:")
    gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
    gb.fit(X_train, y_train)
    proba = gb.predict_proba(X_test[:3])
    print(f"    predict_proba shape: {proba.shape}")
    staged = list(gb.staged_predict(X_test))
    print(f"    staged_predict: {len(staged)} stages")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()