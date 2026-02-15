"""
Scikit-learn reproducibility: random_state usage, environment reproducibility, seed management
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def main():
    print("=" * 60)
    print("Reproducibility: random_state, seed management")
    print("=" * 60)

    print("\n[1] random_state in train_test_split:")
    X, y = make_classification(n_samples=200, random_state=0)
    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.2, random_state=42)
    X1b, X2b, y1b, y2b = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"    Same split: {np.array_equal(X1, X1b)}")

    print("\n[2] random_state in estimator:")
    clf1 = RandomForestClassifier(n_estimators=5, random_state=42)
    clf2 = RandomForestClassifier(n_estimators=5, random_state=42)
    clf1.fit(X1, y1)
    clf2.fit(X1, y1)
    pred1 = clf1.predict(X2)
    pred2 = clf2.predict(X2)
    print(f"    Same predictions: {np.array_equal(pred1, pred2)}")

    print("\n[3] Without random_state (non-reproducible):")
    clf3 = RandomForestClassifier(n_estimators=5)
    clf4 = RandomForestClassifier(n_estimators=5)
    clf3.fit(X1, y1)
    clf4.fit(X1, y1)
    pred3 = clf3.predict(X2)
    pred4 = clf4.predict(X2)
    print(f"    Same predictions: {np.array_equal(pred3, pred4)}")

    print("\n[4] Global numpy seed (affects numpy ops):")
    np.random.seed(42)
    a = np.random.rand(3)
    np.random.seed(42)
    b = np.random.rand(3)
    print(f"    Reproducible: {np.array_equal(a, b)}")

    print("\n[5] Environment reproducibility checklist:")
    print("    - Set random_state in all estimators")
    print("    - Set random_state in train_test_split, CV")
    print("    - Pin sklearn, numpy versions")
    print("    - Use pip freeze or conda export")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
