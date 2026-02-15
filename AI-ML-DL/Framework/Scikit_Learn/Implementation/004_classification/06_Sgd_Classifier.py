"""
Scikit-learn SGDClassifier: loss, penalty, partial_fit
"""

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("SGDClassifier: loss, penalty, partial_fit")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] loss options:")
    for loss in ["hinge", "log_loss", "perceptron", "squared_hinge"]:
        sgd = SGDClassifier(loss=loss, max_iter=1000, random_state=42)
        sgd.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, sgd.predict(X_test_scaled))
        print(f"    loss='{loss}': Accuracy = {acc:.4f}")

    print("\n[2] penalty options:")
    for penalty in ["l2", "l1", "elasticnet"]:
        sgd = SGDClassifier(penalty=penalty, max_iter=1000, random_state=42)
        if penalty == "elasticnet":
            sgd.set_params(l1_ratio=0.5)
        sgd.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, sgd.predict(X_test_scaled))
        print(f"    penalty='{penalty}': Accuracy = {acc:.4f}")

    print("\n[3] partial_fit - incremental learning:")
    sgd = SGDClassifier(max_iter=1, warm_start=False, random_state=42)
    classes = np.unique(y_train)
    batch_size = 20
    for i in range(0, len(X_train_scaled), batch_size):
        X_batch = X_train_scaled[i : i + batch_size]
        y_batch = y_train[i : i + batch_size]
        sgd.partial_fit(X_batch, y_batch, classes=classes)
    acc = accuracy_score(y_test, sgd.predict(X_test_scaled))
    print(f"    Trained with partial_fit (batch_size={batch_size})")
    print(f"    Accuracy: {acc:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
