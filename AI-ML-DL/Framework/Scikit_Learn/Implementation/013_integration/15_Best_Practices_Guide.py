"""
Comprehensive best practices: preprocessing, selection, evaluation
"""

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    print("=" * 60)
    print("Best Practices Guide: preprocessing, selection, evaluation")
    print("=" * 60)

    print("\n[1] Preprocessing in pipeline (avoid leakage):")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=2)),
        ("clf", LogisticRegression(max_iter=200, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    print(f"    Test accuracy: {score:.4f}")
    print(f"    Selected features: {pipe.named_steps['select'].get_support()}")

    print("\n[2] Cross-validation for robust evaluation:")
    scores = cross_val_score(pipe, X, y, cv=5)
    print(f"    CV scores: {scores}")
    print(f"    Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")

    print("\n[3] Random state for reproducibility:")
    clf1 = LogisticRegression(random_state=42)
    clf2 = LogisticRegression(random_state=42)
    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    print(f"    Deterministic: {np.allclose(clf1.predict(X_test), clf2.predict(X_test))}")

    print("\n[4] Model selection by problem type:")
    print("    - Binary/multiclass: LogisticRegression, RandomForest, SVC")
    print("    - Imbalanced: class_weight, SMOTE, Precision-Recall")
    print("    - High-dim: L1/L2, PCA, SelectKBest")

    print("\n[5] Evaluation beyond accuracy:")
    X_imb, y_imb = make_classification(weights=[0.9, 0.1], random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X_imb, y_imb, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    print(classification_report(y_te, pred, target_names=["Neg", "Pos"]))

    print("\n[6] Best practices summary:")
    print("    - Use Pipeline for preprocessing + model")
    print("    - Cross-validate, report mean and std")
    print("    - Set random_state for reproducibility")
    print("    - Use appropriate metrics (precision/recall for imbalanced)")
    print("    - Version models and track experiments")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
