"""
Scikit-learn RidgeClassifier: alpha, RidgeClassifierCV
"""

import numpy as np
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("RidgeClassifier: alpha, RidgeClassifierCV")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] RidgeClassifier with different alpha:")
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        rc = RidgeClassifier(alpha=alpha, random_state=42)
        rc.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, rc.predict(X_test_scaled))
        print(f"    alpha={alpha}: Accuracy = {acc:.4f}")

    print("\n[2] RidgeClassifierCV - automatic alpha selection:")
    rc_cv = RidgeClassifierCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    rc_cv.fit(X_train_scaled, y_train)
    print(f"    Best alpha: {rc_cv.alpha_}")
    print(f"    Accuracy: {accuracy_score(y_test, rc_cv.predict(X_test_scaled)):.4f}")

    print("\n[3] decision_function:")
    rc = RidgeClassifier(alpha=1.0, random_state=42)
    rc.fit(X_train_scaled, y_train)
    dec = rc.decision_function(X_test_scaled[:3])
    print("    First 3 samples decision scores:\n", dec)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
