"""
Scikit-learn bootstrap confidence intervals for metrics
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer


def main():
    print("=" * 60)
    print("Bootstrap Confidence Intervals for Model Evaluation")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(max_iter=500, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[1] cross_val_score - mean and std:")
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    print(f"    CV scores: {np.round(scores, 4)}")
    print(f"    Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")

    print("\n[2] Approximate 95% CI (mean +/- 1.96*std):")
    ci_low = scores.mean() - 1.96 * scores.std() / np.sqrt(len(scores))
    ci_high = scores.mean() + 1.96 * scores.std() / np.sqrt(len(scores))
    print(f"    95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    print("\n[3] Bootstrap resampling for CI:")
    n_bootstrap = 200
    rng = np.random.RandomState(42)
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = rng.choice(len(X), size=len(X), replace=True)
        X_boot, y_boot = X[indices], y[indices]
        scores_boot = cross_val_score(clf, X_boot, y_boot, cv=cv, scoring="accuracy")
        bootstrap_scores.append(scores_boot.mean())
    bootstrap_scores = np.array(bootstrap_scores)
    ci_2_5 = np.percentile(bootstrap_scores, 2.5)
    ci_97_5 = np.percentile(bootstrap_scores, 97.5)
    print(f"    Bootstrap 95% CI: [{ci_2_5:.4f}, {ci_97_5:.4f}]")

    print("\n[4] Percentile method (5-fold CV scores):")
    p2_5 = np.percentile(scores, 2.5)
    p97_5 = np.percentile(scores, 97.5)
    print(f"    Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()