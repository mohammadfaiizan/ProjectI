"""
Scikit-learn PrecisionRecallDisplay for imbalanced classification
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.dummy import DummyClassifier


def main():
    print("=" * 60)
    print("PrecisionRecallDisplay")
    print("=" * 60)

    print("\n[1] Imbalanced binary data:")
    X, y = make_classification(n_samples=500, weights=[0.9, 0.1], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    print(f"    Class distribution: {np.bincount(y_test)}")

    print("\n[2] PrecisionRecallDisplay.from_predictions:")
    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    disp = PrecisionRecallDisplay.from_predictions(y_test, y_proba)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    print(f"    Average precision: {ap:.4f}")
    print(f"    Precision at recall 0.5: {np.interp(0.5, rec[::-1], prec[::-1]):.4f}")

    print("\n[3] PrecisionRecallDisplay.from_estimator:")
    disp2 = PrecisionRecallDisplay.from_estimator(clf, X_test, y_test)
    print(f"    From estimator AP: {disp2.average_precision:.4f}")

    print("\n[4] Baseline (no skill) comparison:")
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(X_train, y_train)
    y_proba_d = dummy.predict_proba(X_test)[:, 1]
    disp3 = PrecisionRecallDisplay.from_predictions(y_test, y_proba_d, name="Dummy")
    ap_dummy = average_precision_score(y_test, y_proba_d)
    print(f"    Dummy AP: {ap_dummy:.4f}")
    print(f"    Model vs Dummy: {ap:.4f} vs {ap_dummy:.4f}")

    print("\n[5] Plot with positive class prevalence:")
    pos_ratio = y_test.mean()
    print(f"    Positive ratio: {pos_ratio:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
