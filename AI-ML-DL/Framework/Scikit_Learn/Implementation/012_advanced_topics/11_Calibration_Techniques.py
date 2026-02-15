"""
Scikit-learn calibration: CalibratedClassifierCV, calibration_curve
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss


def main():
    print("=" * 60)
    print("Calibration: CalibratedClassifierCV, calibration_curve")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_bin = X[y != 2]
    y_bin = y[y != 2]
    X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, random_state=42)

    print("\n[1] CalibratedClassifierCV method='isotonic':")
    base = GaussianNB()
    cal_iso = CalibratedClassifierCV(base, method="isotonic", cv=3)
    cal_iso.fit(X_train, y_train)
    prob_iso = cal_iso.predict_proba(X_test)[:, 1]
    print(f"    Brier score: {brier_score_loss(y_test, prob_iso):.4f}")

    print("\n[2] CalibratedClassifierCV method='sigmoid':")
    cal_sig = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    cal_sig.fit(X_train, y_train)
    prob_sig = cal_sig.predict_proba(X_test)[:, 1]
    print(f"    Brier score: {brier_score_loss(y_test, prob_sig):.4f}")

    print("\n[3] calibration_curve for visualization data:")
    prob_true, prob_pred = calibration_curve(y_test, prob_iso, n_bins=5)
    print(f"    prob_true: {prob_true}")
    print(f"    prob_pred: {prob_pred}")

    print("\n[4] Compare uncalibrated vs calibrated:")
    base.fit(X_train, y_train)
    prob_raw = base.predict_proba(X_test)[:, 1]
    print(f"    Uncalibrated Brier: {brier_score_loss(y_test, prob_raw):.4f}")
    print(f"    Calibrated (isotonic) Brier: {brier_score_loss(y_test, prob_iso):.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
