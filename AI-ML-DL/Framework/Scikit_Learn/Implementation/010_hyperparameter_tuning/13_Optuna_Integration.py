"""
Optuna integration patterns with sklearn
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def main():
    print("=" * 60)
    print("Optuna integration patterns with sklearn")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if not OPTUNA_AVAILABLE:
        print("\n[!] Optuna not installed. Install with: pip install optuna")
        print("    Pattern: define objective, suggest params, use cross_val_score")
        print("\n" + "=" * 60)
        return

    def objective(trial):
        C = trial.suggest_float("C", 1e-2, 1e2, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
        clf = SVC(C=C, gamma=gamma, kernel=kernel, random_state=42)
        scores = cross_val_score(clf, X_train, y_train, cv=3)
        return scores.mean()

    print("\n[1] Optuna study with TPE sampler:")
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42, n_startup_trials=5))
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    print(f"    Best value: {study.best_value:.4f}")
    print(f"    Best params: {study.best_params}")

    print("\n[2] Train final model with best params:")
    best = study.best_params
    clf = SVC(**best, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(f"    Test accuracy: {(pred == y_test).mean():.4f}")

    print("\n[3] Optuna + sklearn pattern:")
    print("    objective(trial) -> suggest params -> cross_val_score -> return metric")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
