"""
Scikit-learn Classification Metrics: accuracy_score, precision_score, recall_score, f1_score
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def main():
    print("=" * 60)
    print("Classification Metrics: accuracy, precision, recall, f1")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    print("\n[1] Binary-style (first 2 classes):")
    mask = y_test < 2
    y_test_bin = y_test[mask]
    y_pred_bin = y_pred[mask]
    print(f"    accuracy_score:  {accuracy_score(y_test_bin, y_pred_bin):.4f}")
    print(f"    precision_score: {precision_score(y_test_bin, y_pred_bin, zero_division=0):.4f}")
    print(f"    recall_score:    {recall_score(y_test_bin, y_pred_bin, zero_division=0):.4f}")
    print(f"    f1_score:        {f1_score(y_test_bin, y_pred_bin, zero_division=0):.4f}")

    print("\n[2] Multiclass (average parameter):")
    for avg in ["macro", "micro", "weighted"]:
        prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)
        print(f"    average='{avg}': precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}")

    print("\n[3] Per-class (average=None):")
    prec = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    for i in range(3):
        print(f"    Class {i}: precision={prec[i]:.4f}, recall={rec[i]:.4f}, f1={f1[i]:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
