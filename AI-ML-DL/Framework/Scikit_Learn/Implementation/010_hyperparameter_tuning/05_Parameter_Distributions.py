"""
Parameter distributions: scipy.stats (uniform, randint, loguniform), param_distributions
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from scipy.stats import uniform, randint, loguniform


def main():
    print("=" * 60)
    print("Parameter distributions: uniform, randint, loguniform")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[1] randint for discrete parameters:")
    dist = randint(1, 10)
    samples = dist.rvs(size=5, random_state=42)
    print(f"    randint(1, 10) samples: {samples}")

    print("\n[2] uniform for continuous parameters:")
    dist = uniform(0.0, 1.0)
    samples = dist.rvs(size=3, random_state=42)
    print(f"    uniform(0, 1) samples: {samples}")

    print("\n[3] loguniform for scale-sensitive params (e.g., C, gamma):")
    dist = loguniform(1e-3, 1e3)
    samples = dist.rvs(size=3, random_state=42)
    print(f"    loguniform(1e-3, 1e3) samples: {samples}")

    param_distributions = {
        "C": loguniform(1e-2, 1e2),
        "gamma": loguniform(1e-4, 1e-1),
        "kernel": ["rbf", "linear"],
    }

    print("\n[4] RandomizedSearchCV with mixed distributions:")
    search = RandomizedSearchCV(
        SVC(random_state=42),
        param_distributions=param_distributions,
        n_iter=15,
        cv=3,
        random_state=42,
    )
    search.fit(X_train, y_train)
    print(f"    best_params_: {search.best_params_}")
    print(f"    C sampled from log scale: {search.best_params_['C']:.6f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
