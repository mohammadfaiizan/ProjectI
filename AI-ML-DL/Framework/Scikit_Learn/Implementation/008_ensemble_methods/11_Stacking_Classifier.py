"""
Scikit-learn StackingClassifier: estimators, final_estimator, cv
"""

import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("StackingClassifier: estimators, final_estimator, cv")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    estimators = [
        ("lr", LogisticRegression(max_iter=500, random_state=42)),
        ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ("svc", SVC(probability=True, random_state=42)),
    ]

    print("\n[1] StackingClassifier - default final_estimator (LogisticRegression):")
    stk = StackingClassifier(estimators=estimators, cv=5)
    stk.fit(X_train, y_train)
    acc = accuracy_score(y_test, stk.predict(X_test))
    print(f"    cv=5: Accuracy = {acc:.4f}")

    print("\n[2] Custom final_estimator:")
    stk_custom = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=0.1, max_iter=500),
        cv=5
    )
    stk_custom.fit(X_train, y_train)
    acc = accuracy_score(y_test, stk_custom.predict(X_test))
    print(f"    final_estimator=LogisticRegression(C=0.1): Accuracy = {acc:.4f}")

    print("\n[3] cv - cross-validation folds:")
    for cv in [3, 5, 10]:
        stk = StackingClassifier(estimators=estimators, cv=cv)
        stk.fit(X_train, y_train)
        acc = accuracy_score(y_test, stk.predict(X_test))
        print(f"    cv={cv}: Accuracy = {acc:.4f}")

    print("\n[4] stack_method - for probability output:")
    stk_prob = StackingClassifier(
        estimators=estimators,
        stack_method="predict_proba",
        cv=5
    )
    stk_prob.fit(X_train, y_train)
    probs = stk_prob.predict_proba(X_test[:3])
    print(f"    stack_method='predict_proba': shape = {probs.shape}")

    print("\n[5] named_estimators_ and final_estimator_:")
    stk.fit(X_train, y_train)
    print(f"    Base estimators: {list(stk.named_estimators_.keys())}")
    print(f"    Final estimator type: {type(stk.final_estimator_).__name__}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
