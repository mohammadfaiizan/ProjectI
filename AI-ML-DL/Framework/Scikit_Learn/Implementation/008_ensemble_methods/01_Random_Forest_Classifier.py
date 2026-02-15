"""
Scikit-learn RandomForestClassifier: n_estimators, max_depth, max_features, feature_importances_
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("RandomForestClassifier: n_estimators, max_depth, max_features")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[1] n_estimators - number of trees:")
    for n in [10, 50, 100, 200]:
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        rf.fit(X_train, y_train)
        acc = accuracy_score(y_test, rf.predict(X_test))
        print(f"    n_estimators={n}: Accuracy = {acc:.4f}")

    print("\n[2] max_depth - tree depth limit:")
    for depth in [3, 5, 10, None]:
        rf = RandomForestClassifier(n_estimators=50, max_depth=depth, random_state=42)
        rf.fit(X_train, y_train)
        acc = accuracy_score(y_test, rf.predict(X_test))
        print(f"    max_depth={depth}: Accuracy = {acc:.4f}")

    print("\n[3] max_features - features per split:")
    for mf in ["sqrt", "log2", 2, 4]:
        rf = RandomForestClassifier(n_estimators=50, max_features=mf, random_state=42)
        rf.fit(X_train, y_train)
        acc = accuracy_score(y_test, rf.predict(X_test))
        print(f"    max_features={mf}: Accuracy = {acc:.4f}")

    print("\n[4] feature_importances_:")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    for i, imp in enumerate(rf.feature_importances_):
        print(f"    Feature {i}: {imp:.4f}")
    print(f"    Sum: {rf.feature_importances_.sum():.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
