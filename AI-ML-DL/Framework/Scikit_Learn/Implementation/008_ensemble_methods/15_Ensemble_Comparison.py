"""
Scikit-learn Comparing all ensemble methods on one dataset
"""

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score


def main():
    print("=" * 60)
    print("Ensemble Method Comparison on Breast Cancer Dataset")
    print("=" * 60)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=100,
            random_state=42
        ),
        "Voting": VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=500)),
                ("dt", DecisionTreeClassifier(max_depth=5)),
                ("svc", SVC(probability=True)),
            ],
            voting="soft"
        ),
        "Stacking": StackingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=500)),
                ("dt", DecisionTreeClassifier(max_depth=5)),
                ("svc", SVC(probability=True)),
            ],
            final_estimator=LogisticRegression(),
            cv=5
        ),
    }

    print("\n[1] Test set accuracy and F1:")
    print("-" * 50)
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        results.append((name, acc, f1))
        print(f"    {name:20s}: Acc={acc:.4f}, F1={f1:.4f}")

    print("\n[2] Cross-validation (5-fold) accuracy:")
    print("-" * 50)
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"    {name:20s}: CV Acc = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

    print("\n[3] Best model by test accuracy:")
    best = max(results, key=lambda x: x[1])
    print(f"    {best[0]}: Accuracy = {best[1]:.4f}")

    print("\n[4] Training time (relative):")
    import time
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        print(f"    {name:20s}: {elapsed:.3f}s")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
