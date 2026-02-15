"""
Scikit-learn custom scoring in pipeline context
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score
from sklearn.datasets import load_iris


def custom_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def main():
    print("=" * 60)
    print("Pipeline Custom Scoring")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    y_binary = (y == 0).astype(int)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])

    print("\n[1] make_scorer from existing metric:")
    f1_scorer = make_scorer(f1_score, average="macro")
    scores = cross_val_score(pipe, X, y, cv=3, scoring=f1_scorer)
    print(f"    F1 (macro) CV scores: {np.round(scores, 4)}")

    print("\n[2] Custom callable as scorer:")
    custom_scorer = make_scorer(custom_accuracy)
    scores_custom = cross_val_score(pipe, X, y, cv=3, scoring=custom_scorer)
    print(f"    Custom accuracy scores: {np.round(scores_custom, 4)}")

    print("\n[3] make_scorer with greater_is_better=False:")
    neg_recall = make_scorer(recall_score, average="macro", greater_is_better=False)
    scores_neg = cross_val_score(pipe, X, y, cv=3, scoring=neg_recall)
    print(f"    Negative recall (minimize): {np.round(scores_neg, 4)}")

    print("\n[4] GridSearchCV with custom scoring:")
    param_grid = {"clf__C": [0.1, 1.0, 10.0]}
    gs = GridSearchCV(
        pipe, param_grid, cv=3, scoring=make_scorer(precision_score, average="macro")
    )
    gs.fit(X, y)
    print(f"    Best precision score: {gs.best_score_:.4f}")
    print(f"    Best C: {gs.best_params_['clf__C']}")

    print("\n[5] Multiple scorers in cross_validate:")
    from sklearn.model_selection import cross_validate
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": make_scorer(f1_score, average="macro"),
        "recall": make_scorer(recall_score, average="macro"),
    }
    results = cross_validate(pipe, X, y, cv=3, scoring=scoring)
    print(f"    test_accuracy mean: {results['test_accuracy'].mean():.4f}")
    print(f"    test_f1_macro mean: {results['test_f1_macro'].mean():.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
