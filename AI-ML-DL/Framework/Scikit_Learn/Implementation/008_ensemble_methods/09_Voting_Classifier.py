"""
Scikit-learn VotingClassifier: voting='hard'/'soft', weights
"""

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("VotingClassifier: voting='hard'/'soft', weights")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    estimators = [
        ("lr", LogisticRegression(max_iter=500, random_state=42)),
        ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ("svc", SVC(probability=True, random_state=42)),
    ]

    print("\n[1] Hard voting - majority class:")
    vc_hard = VotingClassifier(estimators=estimators, voting="hard")
    vc_hard.fit(X_train, y_train)
    acc = accuracy_score(y_test, vc_hard.predict(X_test))
    print(f"    voting='hard': Accuracy = {acc:.4f}")

    print("\n[2] Soft voting - averaged probabilities:")
    vc_soft = VotingClassifier(estimators=estimators, voting="soft")
    vc_soft.fit(X_train, y_train)
    acc = accuracy_score(y_test, vc_soft.predict(X_test))
    print(f"    voting='soft': Accuracy = {acc:.4f}")

    print("\n[3] Individual estimator accuracies:")
    for name, est in estimators:
        est.fit(X_train, y_train)
        acc = accuracy_score(y_test, est.predict(X_test))
        print(f"    {name}: Accuracy = {acc:.4f}")

    print("\n[4] Custom weights:")
    vc_weighted = VotingClassifier(estimators=estimators, voting="soft", weights=[2, 1, 1])
    vc_weighted.fit(X_train, y_train)
    acc = accuracy_score(y_test, vc_weighted.predict(X_test))
    print(f"    weights=[2,1,1]: Accuracy = {acc:.4f}")

    print("\n[5] named_estimators_ - access individual estimators:")
    vc = VotingClassifier(estimators=estimators, voting="hard")
    vc.fit(X_train, y_train)
    print(f"    Estimator names: {list(vc.named_estimators_.keys())}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
