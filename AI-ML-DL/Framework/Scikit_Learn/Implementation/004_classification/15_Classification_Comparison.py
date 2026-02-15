"""
Scikit-learn Classification Comparison: Comparing all classifiers on one dataset
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LogisticRegression, SGDClassifier, Perceptron,
    PassiveAggressiveClassifier, RidgeClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("Classification Model Comparison")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(random_state=42),
        "SVC": SVC(kernel="rbf", random_state=42),
        "LinearSVC": LinearSVC(max_iter=5000, random_state=42),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5),
        "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=5, random_state=42),
        "GaussianNB": GaussianNB(),
        "SGDClassifier": SGDClassifier(max_iter=1000, random_state=42),
        "Perceptron": Perceptron(max_iter=1000, random_state=42),
        "PassiveAggressiveClassifier": PassiveAggressiveClassifier(random_state=42),
        "NearestCentroid": NearestCentroid(),
        "RidgeClassifier": RidgeClassifier(random_state=42),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=50, random_state=42),
    }

    print("\n[1] Test Accuracy:")
    print("-" * 50)
    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_scaled))
        results.append((name, acc))
        print(f"    {name:30s}: {acc:.4f}")

    print("\n[2] Cross-validation (5-fold) Accuracy:")
    print("-" * 50)
    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        print(f"    {name:30s}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    print("\n[3] Best model by test accuracy:")
    best = max(results, key=lambda x: x[1])
    print(f"    {best[0]}: {best[1]:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
