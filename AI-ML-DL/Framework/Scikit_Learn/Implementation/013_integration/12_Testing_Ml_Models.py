"""
Testing ML models: parametrize_with_checks, data-driven tests
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils import check_array


def main():
    print("=" * 60)
    print("Testing ML Models: parametrize_with_checks, data-driven tests")
    print("=" * 60)

    print("\n[1] parametrize_with_checks usage:")
    print("    @parametrize_with_checks([LogisticRegression(), RandomForestClassifier()])")
    print("    def test_sklearn_compatibility(estimator, check):")
    print("        check(estimator)")

    print("\n[2] Common checks run by parametrize_with_checks:")
    checks = [
        "check_estimator",
        "check_fit_score_takes_y",
        "check_no_attributes_set_in_init",
        "check_estimators_dtypes",
        "check_fit_idempotent",
    ]
    for c in checks:
        print(f"    - {c}")

    print("\n[3] Data-driven test pattern:")
    X, y = make_classification(n_samples=100, random_state=42)
    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape[0] == X.shape[0]
    assert set(pred).issubset(set(clf.classes_))
    print("    Assert pred shape and classes: OK")

    print("\n[4] Regression test for API stability:")
    lr = LogisticRegression()
    assert hasattr(lr, "fit") and hasattr(lr, "predict")
    assert hasattr(lr, "predict_proba")
    print("    API checks: OK")

    print("\n[5] Input validation test:")
    X_bad = np.array([[1, 2], [3, np.nan]])
    try:
        check_array(X_bad, allow_nan=False)
    except ValueError as e:
        print(f"    check_array catches NaN: {type(e).__name__}")

    print("\n[6] Run pytest with: pytest 12_Testing_Ml_Models.py -v")
    print("    Or: python -m pytest 12_Testing_Ml_Models.py -v")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
