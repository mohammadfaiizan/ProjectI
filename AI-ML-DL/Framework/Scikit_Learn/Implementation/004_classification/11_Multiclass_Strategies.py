"""
Scikit-learn Multiclass Strategies: OneVsRestClassifier, OneVsOneClassifier
"""

import numpy as np
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def main():
    print("=" * 60)
    print("Multiclass Strategies: OneVsRest, OneVsOne")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] OneVsRestClassifier (OvR) - one binary classifier per class:")
    ovr = OneVsRestClassifier(SVC(kernel="linear", random_state=42))
    ovr.fit(X_train_scaled, y_train)
    print(f"    Number of estimators: {len(ovr.estimators_)}")
    print(f"    Accuracy: {accuracy_score(y_test, ovr.predict(X_test_scaled)):.4f}")

    print("\n[2] OneVsOneClassifier (OvO) - one binary classifier per pair:")
    ovo = OneVsOneClassifier(SVC(kernel="linear", random_state=42))
    ovo.fit(X_train_scaled, y_train)
    n_classes = len(np.unique(y_train))
    n_pairs = n_classes * (n_classes - 1) // 2
    print(f"    Number of estimators: {len(ovo.estimators_)} (expected {n_pairs})")
    print(f"    Accuracy: {accuracy_score(y_test, ovo.predict(X_test_scaled)):.4f}")

    print("\n[3] OvR with LogisticRegression:")
    ovr_lr = OneVsRestClassifier(LogisticRegression(random_state=42))
    ovr_lr.fit(X_train_scaled, y_train)
    print(f"    Accuracy: {accuracy_score(y_test, ovr_lr.predict(X_test_scaled)):.4f}")

    print("\n[4] decision_function (OvR - one score per class):")
    dec = ovr.decision_function(X_test_scaled[:3])
    print("    First 3 samples:\n", dec)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
