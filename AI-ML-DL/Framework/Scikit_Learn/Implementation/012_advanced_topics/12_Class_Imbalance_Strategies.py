"""
Scikit-learn class imbalance: class_weight, sample_weight, SMOTE concepts, imblearn mention
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score


def main():
    print("=" * 60)
    print("Class Imbalance: class_weight, sample_weight, SMOTE concepts")
    print("=" * 60)

    X, y = make_classification(
        n_samples=1000, n_features=10, weights=[0.9, 0.1], random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    print(f"\n[0] Class distribution: {np.bincount(y_train)}")

    print("\n[1] class_weight='balanced':")
    clf_bal = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    clf_bal.fit(X_train, y_train)
    pred_bal = clf_bal.predict(X_test)
    print(classification_report(y_test, pred_bal, zero_division=0))

    print("\n[2] class_weight dict (custom weights):")
    clf_dict = LogisticRegression(
        class_weight={0: 0.5, 1: 2.0}, max_iter=500, random_state=42
    )
    clf_dict.fit(X_train, y_train)
    pred_dict = clf_dict.predict(X_test)
    print(f"    F1 macro: {f1_score(y_test, pred_dict, average='macro'):.4f}")

    print("\n[3] sample_weight (per-sample importance):")
    sample_weights = np.ones(len(y_train))
    sample_weights[y_train == 1] = 5.0
    clf_sw = LogisticRegression(max_iter=500, random_state=42)
    clf_sw.fit(X_train, y_train, sample_weight=sample_weights)
    pred_sw = clf_sw.predict(X_test)
    print(f"    F1 macro: {f1_score(y_test, pred_sw, average='macro'):.4f}")

    print("\n[4] SMOTE and imblearn:")
    print("    SMOTE (Synthetic Minority Over-sampling) not in sklearn.")
    print("    Use imbalanced-learn: pip install imbalanced-learn")
    print("    from imblearn.over_sampling import SMOTE")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
