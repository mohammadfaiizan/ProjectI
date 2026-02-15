"""
Scikit-learn Probability Calibration: CalibratedClassifierCV (isotonic/sigmoid)
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss


def main():
    print("=" * 60)
    print("Probability Calibration: CalibratedClassifierCV")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base = SVC(probability=False, random_state=42)

    print("\n[1] Uncalibrated SVC (probability=False):")
    base.fit(X_train_scaled, y_train)
    print("    SVC does not support predict_proba when probability=False")

    print("\n[2] CalibratedClassifierCV - method='sigmoid' (Platt scaling):")
    cal_sigmoid = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    cal_sigmoid.fit(X_train_scaled, y_train)
    probs_s = cal_sigmoid.predict_proba(X_test_scaled)
    print(f"    Log loss: {log_loss(y_test, probs_s):.4f}")
    print(f"    Accuracy: {accuracy_score(y_test, cal_sigmoid.predict(X_test_scaled)):.4f}")

    print("\n[3] CalibratedClassifierCV - method='isotonic':")
    cal_iso = CalibratedClassifierCV(base, method="isotonic", cv=3)
    cal_iso.fit(X_train_scaled, y_train)
    probs_i = cal_iso.predict_proba(X_test_scaled)
    print(f"    Log loss: {log_loss(y_test, probs_i):.4f}")
    print(f"    Accuracy: {accuracy_score(y_test, cal_iso.predict(X_test_scaled)):.4f}")

    print("\n[4] Calibration with cv='prefit' (pre-fitted estimator):")
    base_fit = SVC(probability=False, random_state=42)
    base_fit.fit(X_train_scaled, y_train)
    cal_prefit = CalibratedClassifierCV(base_fit, method="sigmoid", cv="prefit")
    cal_prefit.fit(X_train_scaled, y_train)
    print(f"    Accuracy: {accuracy_score(y_test, cal_prefit.predict(X_test_scaled)):.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
