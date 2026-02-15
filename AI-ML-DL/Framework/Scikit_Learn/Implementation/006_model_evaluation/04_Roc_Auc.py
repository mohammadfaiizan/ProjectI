"""
Scikit-learn ROC and AUC: roc_curve, roc_auc_score, RocCurveDisplay, multi-class AUC
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("ROC Curve and AUC Score")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_proba = clf.predict_proba(X_test_scaled)

    print("\n[1] Binary ROC (class 0 vs rest):")
    y_test_bin = (y_test == 0).astype(int)
    y_score = y_proba[:, 0]
    fpr, tpr, thresholds = roc_curve(y_test_bin, y_score)
    auc = roc_auc_score(y_test_bin, y_score)
    print(f"    AUC: {auc:.4f}")
    print(f"    FPR (first 5): {fpr[:5]}")
    print(f"    TPR (first 5): {tpr[:5]}")

    print("\n[2] RocCurveDisplay.from_estimator (binary):")
    y_train_bin = (y_train == 0).astype(int)
    clf_bin = LogisticRegression(random_state=42)
    clf_bin.fit(X_train_scaled, y_train_bin)
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(clf_bin, X_test_scaled, y_test_bin, ax=ax)
    plt.tight_layout()
    plt.savefig("roc_curve_binary.png", dpi=100)
    plt.close()
    print("    Saved roc_curve_binary.png")

    print("\n[3] Multi-class AUC (ovr, ovr_weighted, macro, micro):")
    for multi_class in ["ovr", "ovr_weighted", "macro", "micro"]:
        auc_mc = roc_auc_score(y_test, y_proba, multi_class=multi_class)
        print(f"    {multi_class}: {auc_mc:.4f}")

    print("\n[4] One-vs-rest AUC per class:")
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    for i in range(3):
        auc_i = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        print(f"    Class {i}: AUC = {auc_i:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
