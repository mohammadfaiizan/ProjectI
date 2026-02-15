"""
Scikit-learn advanced patterns: clone, meta-estimators, duck typing
"""

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    print("=" * 60)
    print("Advanced Patterns: clone, meta-estimators, duck typing")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)

    print("\n[1] clone creates unfitted copy with same parameters:")
    clf = LogisticRegression(max_iter=500, C=0.5, random_state=42)
    clf.fit(X, y)
    clf_copy = clone(clf)
    print(f"    Original fitted: {hasattr(clf, 'classes_')}")
    print(f"    Clone fitted: {hasattr(clf_copy, 'classes_')}")
    print(f"    Same C: {clf_copy.C == clf.C}")

    print("\n[2] Meta-estimators (Bagging wraps base estimator):")
    base = DecisionTreeClassifier(max_depth=3, random_state=42)
    bag = BaggingClassifier(estimator=base, n_estimators=5, random_state=42)
    bag.fit(X, y)
    print(f"    Bagging uses {len(bag.estimators_)} clones of base")
    print(f"    Base type: {type(bag.estimators_[0]).__name__}")

    print("\n[3] Duck typing: any object with fit/predict works:")
    class DummyClassifier:
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self
        def predict(self, X):
            return np.random.choice(self.classes_, len(X))
    dummy = DummyClassifier()
    dummy.fit(X, y)
    pred = dummy.predict(X[:5])
    print(f"    Dummy predict: {pred}")

    print("\n[4] clone with Pipeline:")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])
    pipe.fit(X, y)
    pipe_clone = clone(pipe)
    print(f"    Pipeline clone steps: {[s for s, _ in pipe_clone.steps]}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
