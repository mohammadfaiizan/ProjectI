"""
Scikit-learn RandomizedSearchCV: param_distributions, n_iter, random_state
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform


def main():
    print("=" * 60)
    print("RandomizedSearchCV: param_distributions, n_iter")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_distributions = {
        "n_estimators": randint(10, 200),
        "max_depth": randint(2, 20),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
    }

    print("\n[1] RandomizedSearchCV with n_iter=20:")
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )
    random_search.fit(X_train, y_train)

    print(f"    best_params_: {random_search.best_params_}")
    print(f"    best_score_: {random_search.best_score_:.4f}")

    print("\n[2] Fewer fits than full grid:")
    print(f"    Evaluated {len(random_search.cv_results_['params'])} combinations")

    print("\n[3] Reproducibility with random_state:")
    random_search2 = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )
    random_search2.fit(X_train, y_train)
    print(f"    Same best_score_ with same random_state: {random_search2.best_score_ == random_search.best_score_}")

    print("\n[4] Predict with best model:")
    pred = random_search.predict(X_test)
    print(f"    Test accuracy: {(pred == y_test).mean():.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
