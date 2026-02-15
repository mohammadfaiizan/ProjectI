"""
Scikit-learn Custom Scoring: make_scorer, custom scoring functions, greater_is_better
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error
from sklearn.metrics import recall_score


def my_custom_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def my_penalized_mse(y_true, y_pred, penalty=1.5):
    mse = np.mean((y_true - y_pred) ** 2)
    return -mse * penalty


def main():
    print("=" * 60)
    print("Custom Scoring with make_scorer")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=42)

    print("\n[1] make_scorer from custom function:")
    custom_scorer = make_scorer(my_custom_accuracy)
    scores = cross_val_score(clf, X, y, cv=5, scoring=custom_scorer)
    print(f"    Custom accuracy scores: {scores}")
    print(f"    Mean: {scores.mean():.4f}")

    print("\n[2] greater_is_better=False (for loss-like metrics):")
    neg_mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    from sklearn.linear_model import Ridge
    from sklearn.datasets import make_regression
    X_reg, y_reg = make_regression(n_samples=100, n_features=5, random_state=42)
    scores = cross_val_score(Ridge(), X_reg, y_reg, cv=5, scoring=neg_mse_scorer)
    print(f"    MSE scores (greater_is_better=False): {scores}")

    print("\n[3] make_scorer with extra parameters:")
    recall_macro = make_scorer(recall_score, average="macro", zero_division=0)
    scores = cross_val_score(clf, X, y, cv=5, scoring=recall_macro)
    print(f"    Recall (macro) scores: {scores}")

    print("\n[4] make_scorer with needs_threshold (for probability):")
    from sklearn.metrics import roc_auc_score
    roc_scorer = make_scorer(roc_auc_score, needs_threshold=True, multi_class="ovr")
    scores = cross_val_score(clf, X, y, cv=5, scoring=roc_scorer)
    print(f"    ROC AUC scores: {scores}")

    print("\n[5] needs_proba for predict_proba-based scoring:")
    from sklearn.metrics import log_loss
    logloss_scorer = make_scorer(log_loss, needs_proba=True)
    scores = cross_val_score(clf, X, y, cv=5, scoring=logloss_scorer)
    print(f"    Log loss scores: {scores}")

    print("\n[6] response_method (decision_function vs predict_proba):")
    from sklearn.svm import SVC
    svc = SVC(probability=True, random_state=42)
    roc_scorer_proba = make_scorer(roc_auc_score, response_method="predict_proba", multi_class="ovr")
    scores = cross_val_score(svc, X, y, cv=3, scoring=roc_scorer_proba)
    print(f"    SVC ROC AUC (predict_proba): {scores}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
