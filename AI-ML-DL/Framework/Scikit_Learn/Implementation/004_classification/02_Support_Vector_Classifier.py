"""
Scikit-learn Support Vector Classifier: SVC (kernel, C, gamma), LinearSVC
"""

import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("Support Vector Classifier: SVC, LinearSVC")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] SVC with different kernels:")
    for kernel in ["linear", "rbf", "poly"]:
        svc = SVC(kernel=kernel, C=1.0, random_state=42)
        svc.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, svc.predict(X_test_scaled))
        print(f"    kernel='{kernel}': Accuracy = {acc:.4f}")

    print("\n[2] C and gamma (RBF kernel):")
    for c, gamma in [(0.1, 0.1), (1.0, "scale"), (10.0, 0.01)]:
        svc = SVC(kernel="rbf", C=c, gamma=gamma, random_state=42)
        svc.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, svc.predict(X_test_scaled))
        print(f"    C={c}, gamma={gamma}: Accuracy = {acc:.4f}")

    print("\n[3] LinearSVC - faster for linear case:")
    lsvc = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    lsvc.fit(X_train_scaled, y_train)
    print(f"    LinearSVC Accuracy: {accuracy_score(y_test, lsvc.predict(X_test_scaled)):.4f}")

    print("\n[4] decision_function:")
    svc = SVC(kernel="rbf", random_state=42)
    svc.fit(X_train_scaled, y_train)
    dec = svc.decision_function(X_test_scaled[:3])
    print("    First 3 samples decision scores:\n", dec)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
