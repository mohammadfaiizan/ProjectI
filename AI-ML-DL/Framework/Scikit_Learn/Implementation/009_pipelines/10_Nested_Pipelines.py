"""
Scikit-learn nested pipelines: nested and complex pipeline structures
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("Nested Pipelines: Complex Structures")
    print("=" * 60)

    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=20, n_informative=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[1] Nested preprocessing pipeline:")
    preprocess = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=5, random_state=42)),
    ])
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(f"    Accuracy: {accuracy_score(y_test, pred):.4f}")

    print("\n[2] Nested param access:")
    pipe.set_params(preprocess__pca__n_components=8)
    pipe.fit(X_train, y_train)
    print(f"    preprocess__pca__n_components=8")
    print(f"    Actual n_components: {pipe.named_steps['preprocess'].named_steps['pca'].n_components}")

    print("\n[3] ColumnTransformer with Pipeline inside:")
    ct = ColumnTransformer([
        ("num", Pipeline([
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=10)),
        ]), list(range(20))),
    ])
    pipe_ct = Pipeline([
        ("preprocess", ct),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])
    pipe_ct.fit(X_train, y_train)
    pred_ct = pipe_ct.predict(X_test)
    print(f"    ColumnTransformer + Pipeline accuracy: {accuracy_score(y_test, pred_ct):.4f}")

    print("\n[4] Deep nesting:")
    inner = Pipeline([("scale", StandardScaler())])
    mid = Pipeline([("inner", inner), ("pca", PCA(n_components=5, random_state=42))])
    outer = Pipeline([("mid", mid), ("clf", LogisticRegression(max_iter=500, random_state=42))])
    outer.fit(X_train, y_train)
    print(f"    mid__inner__scale, mid__pca__n_components")
    print(f"    Deep pipeline accuracy: {accuracy_score(y_test, outer.predict(X_test)):.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
