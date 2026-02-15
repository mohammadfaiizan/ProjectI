"""
Scikit-learn Precision-Recall Curve: precision_recall_curve, average_precision_score, PrecisionRecallDisplay
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    PrecisionRecallDisplay,
)
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("Precision-Recall Curve and Average Precision")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    y_binary = (y == 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_score = clf.predict_proba(X_test_scaled)[:, 1]

    print("\n[1] precision_recall_curve:")
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    print(f"    Precision shape: {precision.shape}")
    print(f"    Recall shape: {recall.shape}")
    print(f"    First 3 precision: {precision[:3]}")
    print(f"    First 3 recall: {recall[:3]}")

    print("\n[2] average_precision_score:")
    ap = average_precision_score(y_test, y_score)
    print(f"    AP: {ap:.4f}")

    print("\n[3] PrecisionRecallDisplay.from_estimator:")
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_estimator(clf, X_test_scaled, y_test, ax=ax)
    plt.tight_layout()
    plt.savefig("precision_recall_curve.png", dpi=100)
    plt.close()
    print("    Saved precision_recall_curve.png")

    print("\n[4] PrecisionRecallDisplay.from_predictions:")
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_test, y_score, ax=ax)
    plt.tight_layout()
    plt.savefig("precision_recall_predictions.png", dpi=100)
    plt.close()
    print("    Saved precision_recall_predictions.png")

    print("\n[5] Multiclass average_precision_score (ovr):")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf.fit(X_train_scaled, y_train)
    y_proba = clf.predict_proba(X_test_scaled)
    ap_macro = average_precision_score(y_test, y_proba, average="macro")
    ap_micro = average_precision_score(y_test, y_proba, average="micro")
    print(f"    Macro: {ap_macro:.4f}, Micro: {ap_micro:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
