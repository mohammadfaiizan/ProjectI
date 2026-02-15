"""
Scikit-learn Perceptron: penalty, alpha, max_iter
"""

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("Perceptron: penalty, alpha, max_iter")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] Basic Perceptron:")
    perc = Perceptron(random_state=42)
    perc.fit(X_train_scaled, y_train)
    print(f"    Accuracy: {accuracy_score(y_test, perc.predict(X_test_scaled)):.4f}")

    print("\n[2] penalty options:")
    for penalty in [None, "l2", "l1", "elasticnet"]:
        perc = Perceptron(penalty=penalty, max_iter=1000, random_state=42)
        if penalty == "elasticnet":
            perc.set_params(eta0=0.01, l1_ratio=0.5)
        perc.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, perc.predict(X_test_scaled))
        print(f"    penalty={penalty}: Accuracy = {acc:.4f}")

    print("\n[3] alpha (regularization strength):")
    for alpha in [0.0001, 0.001, 0.01, 0.1]:
        perc = Perceptron(penalty="l2", alpha=alpha, max_iter=1000, random_state=42)
        perc.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, perc.predict(X_test_scaled))
        print(f"    alpha={alpha}: Accuracy = {acc:.4f}")

    print("\n[4] n_iter_ (epochs to converge):")
    perc = Perceptron(max_iter=1000, random_state=42)
    perc.fit(X_train_scaled, y_train)
    print(f"    n_iter_: {perc.n_iter_}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
