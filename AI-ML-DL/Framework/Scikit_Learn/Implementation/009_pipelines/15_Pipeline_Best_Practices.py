"""
Scikit-learn pipeline best practices: design patterns and practices
"""

import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("Pipeline Best Practices")
    print("=" * 60)

    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[1] Always use Pipeline for preprocessing + model:")
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=42))
    scores = cross_val_score(pipe, X_train, y_train, cv=3)
    print(f"    Prevents data leakage in CV: mean score {scores.mean():.4f}")

    print("\n[2] Use make_pipeline for brevity, Pipeline for explicit names:")
    pipe_explicit = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])
    print(f"    Explicit names help with get_params/set_params")

    print("\n[3] Param naming: stepname__param:")
    pipe_explicit.set_params(clf__C=0.1)
    print(f"    clf__C=0.1 -> LogisticRegression.C")

    print("\n[4] ColumnTransformer for mixed data types:")
    ct = make_column_transformer(
        (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), list(range(10))),
        remainder="drop",
    )
    pipe_ct = make_pipeline(ct, LogisticRegression(max_iter=500, random_state=42))
    pipe_ct.fit(X_train, y_train)
    print(f"    Numeric: impute -> scale; accuracy: {accuracy_score(y_test, pipe_ct.predict(X_test)):.4f}")

    print("\n[5] Keep pipelines in CV/GridSearch, not outside:")
    param_grid = {"logisticregression__C": [0.1, 1.0, 10.0]}
    gs = GridSearchCV(pipe, param_grid, cv=3)
    gs.fit(X_train, y_train)
    print(f"    GridSearch tunes pipeline as a whole")

    print("\n[6] Design patterns summary:")
    print("    - One pipeline = one reproducible model")
    print("    - Preprocessing before model in step order")
    print("    - Use joblib for persistence")
    print("    - Prefer cross_val_score(pipe, X, y) over manual splits")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
