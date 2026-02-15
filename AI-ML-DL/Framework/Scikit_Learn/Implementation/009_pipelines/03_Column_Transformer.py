"""
Scikit-learn ColumnTransformer: ColumnTransformer, make_column_transformer, remainder
"""

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("ColumnTransformer: make_column_transformer, remainder")
    print("=" * 60)

    np.random.seed(42)
    X_num, y = make_classification(n_samples=200, n_features=5, random_state=42)
    X_cat = np.random.choice(["A", "B", "C"], size=(200, 2))
    X = np.hstack([X_num, X_cat])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_cols = [0, 1, 2, 3, 4]
    categorical_cols = [5, 6]

    print("\n[1] ColumnTransformer with explicit transformers:")
    ct = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
    ], remainder="drop")
    X_transformed = ct.fit_transform(X_train)
    print(f"    Transformed shape: {X_transformed.shape}")

    print("\n[2] make_column_transformer:")
    ct_auto = make_column_transformer(
        (StandardScaler(), numeric_cols),
        (OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
        remainder="drop",
    )
    X_auto = ct_auto.fit_transform(X_train)
    print(f"    Transformed shape: {X_auto.shape}")
    print(f"    Transformer names: {[t[0] for t in ct_auto.transformers_]}")

    print("\n[3] remainder='passthrough':")
    ct_pass = ColumnTransformer([
        ("num", StandardScaler(), [0, 1]),
    ], remainder="passthrough")
    X_pass = ct_pass.fit_transform(X_train[:, :5])
    print(f"    Original cols 0-4, scaled 0-1, passthrough 2-4")
    print(f"    Transformed shape: {X_pass.shape}")

    print("\n[4] Full pipeline with ColumnTransformer:")
    pipe = Pipeline([
        ("preprocess", ct),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(f"    Pipeline accuracy: {accuracy_score(y_test, pred):.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
