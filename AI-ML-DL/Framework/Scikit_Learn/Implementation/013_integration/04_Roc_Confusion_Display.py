"""
Scikit-learn RocCurveDisplay and ConfusionMatrixDisplay
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize


def main():
    print("=" * 60)
    print("RocCurveDisplay and ConfusionMatrixDisplay")
    print("=" * 60)

    print("\n[1] Binary ROC curve:")
    X, y = make_classification(n_samples=200, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    disp = RocCurveDisplay.from_predictions(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    print(f"    AUC: {auc(fpr, tpr):.4f}")
    print(f"    Display ax: {disp.ax_ is not None}")

    print("\n[2] RocCurveDisplay.from_estimator:")
    disp2 = RocCurveDisplay.from_estimator(clf, X_test, y_test)
    print(f"    From estimator AUC: {disp2.roc_auc:.4f}")

    print("\n[3] ConfusionMatrixDisplay:")
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    print(f"    Confusion matrix:\n{cm}")
    print(f"    Display created: {cm_disp is not None}")

    print("\n[4] ConfusionMatrixDisplay with display_labels:")
    cm_disp2 = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Neg", "Pos"]
    )
    print(f"    Labels: {cm_disp2.display_labels}")

    print("\n[5] Multiclass one-vs-rest ROC (3 classes):")
    X_m, y_m = make_classification(n_samples=300, n_classes=3, n_informative=4, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X_m, y_m, random_state=42)
    clf_m = LogisticRegression(max_iter=200, random_state=42)
    clf_m.fit(X_tr, y_tr)
    y_bin = label_binarize(y_te, classes=[0, 1, 2])
    y_proba_m = clf_m.predict_proba(X_te)
    for i in range(3):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_proba_m[:, i])
        print(f"    Class {i} AUC: {auc(fpr_i, tpr_i):.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
