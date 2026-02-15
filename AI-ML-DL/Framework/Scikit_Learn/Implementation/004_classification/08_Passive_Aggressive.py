"""
Scikit-learn PassiveAggressiveClassifier: C, loss
"""

import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("PassiveAggressiveClassifier: C, loss")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] Basic PassiveAggressiveClassifier:")
    pac = PassiveAggressiveClassifier(random_state=42)
    pac.fit(X_train_scaled, y_train)
    print(f"    Accuracy: {accuracy_score(y_test, pac.predict(X_test_scaled)):.4f}")

    print("\n[2] C (aggressiveness) parameter:")
    for c in [0.01, 0.1, 1.0, 10.0]:
        pac = PassiveAggressiveClassifier(C=c, random_state=42)
        pac.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, pac.predict(X_test_scaled))
        print(f"    C={c}: Accuracy = {acc:.4f}")

    print("\n[3] loss options:")
    for loss in ["hinge", "squared_hinge"]:
        pac = PassiveAggressiveClassifier(loss=loss, random_state=42)
        pac.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, pac.predict(X_test_scaled))
        print(f"    loss='{loss}': Accuracy = {acc:.4f}")

    print("\n[4] partial_fit for online learning:")
    pac = PassiveAggressiveClassifier(max_iter=1, random_state=42)
    classes = np.unique(y_train)
    for i in range(0, len(X_train_scaled), 20):
        pac.partial_fit(
            X_train_scaled[i : i + 20],
            y_train[i : i + 20],
            classes=classes
        )
    print(f"    Accuracy after partial_fit: {accuracy_score(y_test, pac.predict(X_test_scaled)):.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
