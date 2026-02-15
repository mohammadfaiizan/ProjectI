"""
Scikit-learn HalvingRandomSearchCV: successive halving with random sampling
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import HalvingRandomSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint


def main():
    print("=" * 60)
    print("HalvingRandomSearchCV: successive halving + random params")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_distributions = {
        "n_estimators": randint(10, 150),
        "max_depth": randint(2, 15),
        "min_samples_split": randint(2, 15),
    }

    print("\n[1] HalvingRandomSearchCV with n_candidates and factor:")
    halving_random = HalvingRandomSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_distributions,
        n_candidates="exhaust",
        factor=3,
        resource="n_samples",
        min_resources="smallest",
        cv=3,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )
    halving_random.fit(X_train, y_train)

    print(f"    best_params_: {halving_random.best_params_}")
    print(f"    best_score_: {halving_random.best_score_:.4f}")
    print(f"    n_iterations_: {halving_random.n_iterations_}")

    print("\n[2] n_candidates per iteration:")
    print(f"    n_candidates_: {halving_random.n_candidates_}")

    print("\n[3] n_resources (samples) per iteration:")
    print(f"    n_resources_: {halving_random.n_resources_}")

    print("\n[4] Test performance:")
    pred = halving_random.predict(X_test)
    print(f"    Test accuracy: {(pred == y_test).mean():.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
