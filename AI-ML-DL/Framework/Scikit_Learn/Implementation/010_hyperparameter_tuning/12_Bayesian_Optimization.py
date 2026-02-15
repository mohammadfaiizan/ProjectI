"""
Bayesian optimization: BayesSearchCV concepts (scikit-optimize), search spaces
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


def main():
    print("=" * 60)
    print("Bayesian optimization: BayesSearchCV (scikit-optimize)")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if not SKOPT_AVAILABLE:
        print("\n[!] scikit-optimize not installed. Install with: pip install scikit-optimize")
        print("    Demonstrating search space concepts with placeholder:")
        print("    Real(1e-3, 1e3, prior='log-uniform') for C")
        print("    Real(1e-4, 1e-1, prior='log-uniform') for gamma")
        print("    Categorical(['rbf', 'linear']) for kernel")
        print("\n" + "=" * 60)
        return

    search_spaces = {
        "C": Real(1e-2, 1e2, prior="log-uniform"),
        "gamma": Real(1e-4, 1e-1, prior="log-uniform"),
        "kernel": Categorical(["rbf", "linear"]),
    }

    print("\n[1] BayesSearchCV with skopt search spaces:")
    bayes_search = BayesSearchCV(
        SVC(random_state=42),
        search_spaces=search_spaces,
        n_iter=15,
        cv=3,
        random_state=42,
        n_jobs=-1,
    )
    bayes_search.fit(X_train, y_train)

    print(f"    best_params_: {bayes_search.best_params_}")
    print(f"    best_score_: {bayes_search.best_score_:.4f}")

    print("\n[2] Search space types:")
    print("    Real: continuous, prior='log-uniform' for scale params")
    print("    Integer: discrete integer range")
    print("    Categorical: list of options")

    print("\n[3] Fewer iterations than random search for similar quality:")
    pred = bayes_search.predict(X_test)
    print(f"    Test accuracy: {(pred == y_test).mean():.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
