"""
Scikit-learn Scoring Metrics Introduction
scoring parameter, accuracy, f1, r2, neg_mean_squared_error
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.datasets import load_iris, make_regression
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error


def main():
    print("=" * 60)
    print("Scoring Metrics Introduction")
    print("=" * 60)

    X_clf, y_clf = load_iris(return_X_y=True)
    X_reg, y_reg = make_regression(n_samples=100, n_features=5, random_state=42)
    clf = LogisticRegression(max_iter=200, random_state=42)
    reg = Ridge(random_state=42)

    print("\n[1] Classification - accuracy (default):")
    scores = cross_val_score(clf, X_clf, y_clf, cv=3, scoring="accuracy")
    print(f"    scoring='accuracy': {scores}")

    print("\n[2] Classification - f1_macro (multi-class):")
    scores_f1 = cross_val_score(clf, X_clf, y_clf, cv=3, scoring="f1_macro")
    print(f"    scoring='f1_macro': {scores_f1}")

    print("\n[3] Classification - f1_weighted:")
    scores_f1w = cross_val_score(clf, X_clf, y_clf, cv=3, scoring="f1_weighted")
    print(f"    scoring='f1_weighted': {scores_f1w}")

    print("\n[4] Regression - r2 (default):")
    scores_r2 = cross_val_score(reg, X_reg, y_reg, cv=3, scoring="r2")
    print(f"    scoring='r2': {scores_r2}")

    print("\n[5] Regression - neg_mean_squared_error:")
    scores_mse = cross_val_score(reg, X_reg, y_reg, cv=3, scoring="neg_mean_squared_error")
    print(f"    scoring='neg_mean_squared_error': {scores_mse}")
    print(f"    (Negated so higher is better)")

    print("\n[6] Direct metric computation:")
    clf.fit(X_clf[:100], y_clf[:100])
    y_pred = clf.predict(X_clf[100:])
    print(f"    accuracy_score: {accuracy_score(y_clf[100:], y_pred):.4f}")
    print(f"    f1_score(macro): {f1_score(y_clf[100:], y_pred, average='macro'):.4f}")

    print("\n[7] Common scoring strings:")
    print("    Classification: accuracy, f1, f1_macro, precision, recall")
    print("    Regression: r2, neg_mean_squared_error, neg_mean_absolute_error")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
