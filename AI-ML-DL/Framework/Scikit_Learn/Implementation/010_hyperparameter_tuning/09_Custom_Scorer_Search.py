"""
Custom scorer in search: make_scorer, greater_is_better, needs_proba
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, fbeta_score, roc_auc_score


def main():
    print("=" * 60)
    print("Custom scorer: make_scorer, greater_is_better, needs_proba")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {"C": [0.1, 1.0], "gamma": ["scale", 0.01]}

    print("\n[1] make_scorer from fbeta_score (F2: emphasize recall):")
    f2_scorer = make_scorer(fbeta_score, beta=2, average="macro")
    grid = GridSearchCV(SVC(random_state=42), param_grid, cv=3, scoring=f2_scorer)
    grid.fit(X_train, y_train)
    print(f"    best_score_: {grid.best_score_:.4f}")

    print("\n[2] greater_is_better=False (e.g., loss):")
    def custom_loss(y_true, y_pred):
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

    loss_scorer = make_scorer(custom_loss, greater_is_better=False)
    grid_loss = GridSearchCV(SVC(random_state=42), param_grid, cv=3, scoring=loss_scorer)
    grid_loss.fit(X_train, y_train)
    print(f"    Best minimizes loss: {grid_loss.best_score_:.4f}")

    print("\n[3] needs_proba for probability-based metrics (roc_auc):")
    lr_param_grid = {"C": [0.1, 1.0, 10.0]}
    auc_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr")
    grid_auc = GridSearchCV(
        LogisticRegression(max_iter=500, random_state=42),
        lr_param_grid,
        cv=3,
        scoring=auc_scorer,
    )
    grid_auc.fit(X_train, y_train)
    print(f"    best_score_ (roc_auc_ovr): {grid_auc.best_score_:.4f}")

    print("\n[4] Custom scorer with extra kwargs:")
    def weighted_accuracy(y_true, y_pred, weights=None):
        if weights is None:
            weights = np.ones_like(y_true)
        correct = (np.array(y_true) == np.array(y_pred)).astype(float)
        return np.average(correct, weights=weights)

    wa_scorer = make_scorer(weighted_accuracy)
    grid_wa = GridSearchCV(SVC(random_state=42), param_grid, cv=3, scoring=wa_scorer)
    grid_wa.fit(X_train, y_train)
    print(f"    Custom weighted accuracy best_score_: {grid_wa.best_score_:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
