"""
Scikit-learn Class Weight and Imbalance: class_weight='balanced', sample_weight
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    print("=" * 60)
    print("Class Weight and Imbalance: class_weight, sample_weight")
    print("=" * 60)

    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5,
        n_classes=2, weights=[0.9, 0.1], random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] Class distribution:")
    print(f"    Train: {np.bincount(y_train)}")
    print(f"    Test:  {np.bincount(y_test)}")

    print("\n[2] Without class_weight (default):")
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    pred = lr.predict(X_test_scaled)
    print(classification_report(y_test, pred))
    print("    Confusion matrix:\n", confusion_matrix(y_test, pred))

    print("\n[3] With class_weight='balanced':")
    lr_bal = LogisticRegression(class_weight="balanced", random_state=42)
    lr_bal.fit(X_train_scaled, y_train)
    pred_bal = lr_bal.predict(X_test_scaled)
    print(classification_report(y_test, pred_bal))
    print("    Confusion matrix:\n", confusion_matrix(y_test, pred_bal))

    print("\n[4] Custom class_weight dict:")
    lr_custom = LogisticRegression(
        class_weight={0: 0.5, 1: 2.0},
        random_state=42
    )
    lr_custom.fit(X_train_scaled, y_train)
    print(f"    Accuracy: {accuracy_score(y_test, lr_custom.predict(X_test_scaled)):.4f}")

    print("\n[5] sample_weight - per-sample weights:")
    sample_weights = np.ones(len(y_train))
    sample_weights[y_train == 1] = 5.0
    lr_sw = LogisticRegression(random_state=42)
    lr_sw.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    print(f"    Minority class weight=5.0")
    print(f"    Accuracy: {accuracy_score(y_test, lr_sw.predict(X_test_scaled)):.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
