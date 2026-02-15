"""
Scikit-learn MultiOutputClassifier wrapping any classifier
"""

import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def main():
    print("=" * 60)
    print("MultiOutputClassifier: wrap any classifier for multi-output")
    print("=" * 60)

    X, y = make_multilabel_classification(
        n_samples=200, n_features=20, n_labels=3, n_classes=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    print("\n[1] MultiOutputClassifier with LogisticRegression:")
    clf = MultiOutputClassifier(LogisticRegression(max_iter=500, random_state=42))
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(f"    y shape: {y_test.shape}, pred shape: {pred.shape}")
    acc = np.mean(pred == y_test)
    print(f"    Exact match accuracy: {acc:.4f}")
    f1 = f1_score(y_test, pred, average="samples")
    print(f"    F1 (samples): {f1:.4f}")

    print("\n[2] MultiOutputClassifier with DecisionTreeClassifier:")
    clf2 = MultiOutputClassifier(DecisionTreeClassifier(random_state=42))
    clf2.fit(X_train, y_train)
    pred2 = clf2.predict(X_test)
    print(f"    F1 (samples): {f1_score(y_test, pred2, average='samples'):.4f}")

    print("\n[3] Access individual estimators:")
    for i, est in enumerate(clf.estimators_):
        print(f"    Estimator {i}: {type(est).__name__}")

    print("\n[4] n_jobs for parallel fitting:")
    clf_par = MultiOutputClassifier(LogisticRegression(max_iter=500, random_state=42), n_jobs=2)
    clf_par.fit(X_train, y_train)
    print(f"    Fitted with n_jobs=2: {len(clf_par.estimators_)} estimators")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
