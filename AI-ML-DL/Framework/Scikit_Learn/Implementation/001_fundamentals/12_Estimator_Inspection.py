"""
Scikit-learn Estimator Inspection
get_params(), set_params(), clone(), __repr__
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.datasets import load_iris


def main():
    print("=" * 60)
    print("Estimator Inspection: get_params, set_params, clone")
    print("=" * 60)

    clf = LogisticRegression(max_iter=200, C=1.0, random_state=42)

    print("\n[1] get_params() - all hyperparameters:")
    params = clf.get_params()
    print(f"    Keys (sample): {list(params.keys())[:5]}...")
    print(f"    max_iter: {params['max_iter']}")
    print(f"    C: {params['C']}")

    print("\n[2] set_params() - modify hyperparameters:")
    clf.set_params(C=0.5, max_iter=500)
    print(f"    After set_params(C=0.5): C={clf.C}")
    print(f"    After set_params(max_iter=500): max_iter={clf.max_iter}")

    print("\n[3] set_params() returns self (method chaining):")
    clf = LogisticRegression().set_params(C=0.1).set_params(max_iter=100)
    print(f"    Chained: C={clf.C}, max_iter={clf.max_iter}")

    print("\n[4] clone() - deep copy without fitted state:")
    clf_orig = LogisticRegression(max_iter=200, random_state=42)
    X, y = load_iris(return_X_y=True)
    clf_orig.fit(X[:100], y[:100])
    clf_new = clone(clf_orig)
    print(f"    clone() creates unfitted copy with same hyperparameters")
    print(f"    clf_orig fitted: has coef_ = {hasattr(clf_orig, 'coef_')}")
    print(f"    clf_new unfitted: ready for fresh fit")

    print("\n[5] __repr__ - string representation:")
    print(f"    {clf_orig}")

    print("\n[6] Nested params (e.g. in Pipeline):")
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([("s", StandardScaler()), ("c", LogisticRegression())])
    print(f"    pipe.get_params()['c__C']: {pipe.get_params()['c__C']}")
    pipe.set_params(c__C=0.01)
    print(f"    After set_params(c__C=0.01): {pipe['c'].C}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
