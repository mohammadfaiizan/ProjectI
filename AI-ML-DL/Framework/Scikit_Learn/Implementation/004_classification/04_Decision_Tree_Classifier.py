"""
Scikit-learn DecisionTreeClassifier: criterion, max_depth, min_samples
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("DecisionTreeClassifier: criterion, max_depth, min_samples")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[1] criterion options:")
    for crit in ["gini", "entropy", "log_loss"]:
        dt = DecisionTreeClassifier(criterion=crit, random_state=42)
        dt.fit(X_train, y_train)
        acc = accuracy_score(y_test, dt.predict(X_test))
        print(f"    criterion='{crit}': Accuracy = {acc:.4f}")

    print("\n[2] max_depth - pruning:")
    for depth in [2, 5, 10, None]:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        d = dt.get_depth() if depth is None else depth
        acc = accuracy_score(y_test, dt.predict(X_test))
        print(f"    max_depth={depth}: depth={d}, Accuracy = {acc:.4f}")

    print("\n[3] min_samples_split and min_samples_leaf:")
    dt = DecisionTreeClassifier(min_samples_split=10, min_samples_leaf=5, random_state=42)
    dt.fit(X_train, y_train)
    print(f"    min_samples_split=10, min_samples_leaf=5")
    print(f"    Tree depth: {dt.get_depth()}, Accuracy: {accuracy_score(y_test, dt.predict(X_test)):.4f}")

    print("\n[4] feature_importances_:")
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    for i, imp in enumerate(dt.feature_importances_):
        print(f"    Feature {i}: {imp:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
