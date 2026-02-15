"""
Scikit-learn Confusion Matrix: confusion_matrix, ConfusionMatrixDisplay.from_estimator
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("Confusion Matrix and ConfusionMatrixDisplay")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    print("\n[1] confusion_matrix (raw counts):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\n[2] confusion_matrix with labels:")
    cm_labeled = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    print("    Labels [0,1,2]:\n", cm_labeled)

    print("\n[3] normalize parameter (pred, true, all):")
    for norm in ["pred", "true", "all"]:
        cm_norm = confusion_matrix(y_test, y_pred, normalize=norm)
        print(f"    normalize='{norm}':\n{cm_norm}\n")

    print("\n[4] ConfusionMatrixDisplay.from_estimator (saving to file):")
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay.from_estimator(clf, X_test_scaled, y_test, ax=ax)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100)
    plt.close()
    print("    Saved confusion_matrix.png")

    print("\n[5] ConfusionMatrixDisplay.from_predictions:")
    disp2 = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.tight_layout()
    plt.savefig("confusion_matrix_predictions.png", dpi=100)
    plt.close()
    print("    Saved confusion_matrix_predictions.png")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
