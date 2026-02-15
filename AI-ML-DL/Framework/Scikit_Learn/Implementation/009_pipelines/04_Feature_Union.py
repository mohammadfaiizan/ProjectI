"""
Scikit-learn FeatureUnion: parallel transformations
"""

import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("FeatureUnion: Parallel Transformations")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[1] FeatureUnion with PCA and PolynomialFeatures:")
    fu = FeatureUnion([
        ("pca", PCA(n_components=2, random_state=42)),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ])
    X_fu = fu.fit_transform(X_train)
    print(f"    Original features: {X_train.shape[1]}")
    print(f"    PCA output: 2, Poly output: {4 * 5 // 2}")
    print(f"    Union output shape: {X_fu.shape}")

    print("\n[2] Pipeline: FeatureUnion + Model:")
    pipe = Pipeline([
        ("union", FeatureUnion([
            ("pca", PCA(n_components=2, random_state=42)),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ])),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(f"    Pipeline accuracy: {accuracy_score(y_test, pred):.4f}")

    print("\n[3] Accessing union transformers:")
    pca_out = pipe.named_steps["union"].named_transformers_["pca"].transform(X_train)
    print(f"    PCA explained variance ratio: {pipe.named_steps['union'].named_transformers_['pca'].explained_variance_ratio_}")

    print("\n[4] Union with transformer_weights:")
    fu_weighted = FeatureUnion([
        ("pca", PCA(n_components=2, random_state=42)),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ], transformer_weights={"pca": 1.0, "poly": 0.5})
    X_weighted = fu_weighted.fit_transform(X_train)
    print(f"    Weighted union shape: {X_weighted.shape}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
