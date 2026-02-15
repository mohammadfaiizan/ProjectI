"""
Scikit-learn Evaluation Best Practices: data leakage, class imbalance, overfitting to test
"""

import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


def main():
    print("=" * 60)
    print("Evaluation Best Practices and Pitfalls")
    print("=" * 60)

    print("\n[1] Data leakage - WRONG: fit scaler on full data before split:")
    X, y = load_iris(return_X_y=True)
    scaler_wrong = StandardScaler()
    X_scaled_wrong = scaler_wrong.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_wrong, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    acc_wrong = accuracy_score(y_test, clf.predict(X_test))
    print(f"    Leaked accuracy: {acc_wrong:.4f}")

    print("\n[2] Data leakage - CORRECT: fit scaler only on train:")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler_correct = StandardScaler()
    X_train_scaled = scaler_correct.fit_transform(X_train)
    X_test_scaled = scaler_correct.transform(X_test)
    clf.fit(X_train_scaled, y_train)
    acc_correct = accuracy_score(y_test, clf.predict(X_test_scaled))
    print(f"    Correct accuracy: {acc_correct:.4f}")

    print("\n[3] Use Pipeline for correct CV (no leakage):")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42))])
    scores = cross_val_score(pipe, X, y, cv=5)
    print(f"    CV scores (Pipeline): {scores}")
    print(f"    Mean: {scores.mean():.4f}")

    print("\n[4] Class imbalance - accuracy can be misleading:")
    X_imb, y_imb = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X_imb, y_imb, test_size=0.2, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    print(f"    Accuracy: {accuracy_score(y_te, y_pred):.4f}")
    print("    Classification report (use precision/recall for minority):")
    print(classification_report(y_te, y_pred, target_names=["class0", "class1"]))

    print("\n[5] Overfitting to test - use holdout or nested CV:")
    print("    - Single train/test: test used once, risk of overfitting to test")
    print("    - Cross-validation: average over folds, more robust")
    print("    - Nested CV: outer loop for evaluation, inner for model selection")

    print("\n[6] Stratified split for imbalanced classification:")
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X_imb, y_imb):
        print(f"    Train class distribution: {np.bincount(y_imb[train_idx])}")
        print(f"    Test class distribution:  {np.bincount(y_imb[test_idx])}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
