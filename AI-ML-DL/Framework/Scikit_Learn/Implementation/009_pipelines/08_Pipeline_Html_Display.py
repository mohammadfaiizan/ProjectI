"""
Scikit-learn pipeline HTML display: set_config(display='diagram'), HTML repr
"""

import numpy as np
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def main():
    print("=" * 60)
    print("Pipeline HTML Display: set_config(display='diagram')")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])

    print("\n[1] Default text representation:")
    set_config(display="text")
    print(pipe)

    print("\n[2] Diagram representation (HTML):")
    set_config(display="diagram")
    html_repr = str(pipe)
    print(f"    Output contains 'Pipeline' and HTML-like structure: {'Pipeline' in html_repr and '<' in html_repr}")

    print("\n[3] _repr_html_ in Jupyter:")
    if hasattr(pipe, "_repr_html_"):
        html = pipe._repr_html_()
        print(f"    _repr_html_ length: {len(html)} chars")
        print(f"    Contains svg/diagram: {'svg' in html.lower() or 'Pipeline' in html}")

    print("\n[4] Reset to text:")
    set_config(display="text")
    print("    Config reset to text display")

    print("\n[5] Nested pipeline display:")
    nested = Pipeline([
        ("preprocess", Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ])),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])
    set_config(display="diagram")
    nested_html = nested._repr_html_()
    print(f"    Nested pipeline HTML length: {len(nested_html)} chars")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
