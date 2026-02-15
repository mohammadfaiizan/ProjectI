"""
Scikit-learn Train-Test Split
train_test_split, test_size, stratify, random_state, shuffle
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


def main():
    print("=" * 60)
    print("Train-Test Split")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    print(f"\n[1] Original data: X {X.shape}, y {y.shape}")

    print("\n[2] Basic split (default 75% train, 25% test):")
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"    X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"    y_train: {y_train.shape}, y_test: {y_test.shape}")

    print("\n[3] Custom test_size:")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print(f"    test_size=0.3 -> X_test: {X_test.shape} ({len(X_test)/len(X)*100:.0f}%)")

    print("\n[4] random_state for reproducibility:")
    X_t1, X_t2, _, _ = train_test_split(X, y, random_state=42)
    X_t1b, X_t2b, _, _ = train_test_split(X, y, random_state=42)
    print(f"    Same random_state -> identical splits: {np.array_equal(X_t1, X_t1b)}")

    print("\n[5] stratify - preserve class distribution:")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    print(f"    Original class counts: {np.bincount(y.astype(int))}")
    print(f"    Train class counts: {np.bincount(y_train.astype(int))}")
    print(f"    Test class counts: {np.bincount(y_test.astype(int))}")

    print("\n[6] shuffle parameter:")
    X_noshuf, _, _, _ = train_test_split(X, y, shuffle=False, test_size=0.2)
    print(f"    shuffle=False: first samples go to train")
    print(f"    shuffle=True (default): random split")

    print("\n[7] Multiple arrays in one call:")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"    Returns: X_train, X_test, y_train, y_test")

    print("\n[8] train_size alternative:")
    X_train, X_test, _, _ = train_test_split(X, y, train_size=100, random_state=42)
    print(f"    train_size=100 -> X_train: {X_train.shape}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
