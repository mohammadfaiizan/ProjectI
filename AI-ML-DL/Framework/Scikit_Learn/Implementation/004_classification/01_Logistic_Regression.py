"""
Scikit-learn Logistic Regression: penalty, solver, multi_class, C
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def main():
    print("=" * 60)
    print("Logistic Regression: penalty, solver, multi_class, C")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] Basic LogisticRegression:")
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    print(f"    Accuracy: {accuracy_score(y_test, lr.predict(X_test_scaled)):.4f}")

    print("\n[2] penalty options:")
    for penalty in ["l2", "l1"]:
        solver = "saga" if penalty == "l1" else "lbfgs"
        lr = LogisticRegression(penalty=penalty, solver=solver, C=1.0, random_state=42)
        lr.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, lr.predict(X_test_scaled))
        print(f"    penalty='{penalty}': Accuracy = {acc:.4f}")

    print("\n[3] C (inverse regularization strength):")
    for c in [0.01, 0.1, 1.0, 10.0]:
        lr = LogisticRegression(C=c, random_state=42)
        lr.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, lr.predict(X_test_scaled))
        print(f"    C={c}: Accuracy = {acc:.4f}")

    print("\n[4] multi_class strategies:")
    for mc in ["auto", "ovr", "multinomial"]:
        lr = LogisticRegression(multi_class=mc, random_state=42)
        lr.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, lr.predict(X_test_scaled))
        print(f"    multi_class='{mc}': Accuracy = {acc:.4f}")

    print("\n[5] predict_proba:")
    probs = lr.predict_proba(X_test_scaled[:3])
    print("    First 3 samples probabilities:\n", probs)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
