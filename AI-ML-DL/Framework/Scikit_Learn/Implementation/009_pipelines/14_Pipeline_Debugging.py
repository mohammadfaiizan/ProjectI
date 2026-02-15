"""
Scikit-learn pipeline debugging: step inspection, intermediate outputs, error tracing
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


def main():
    print("=" * 60)
    print("Pipeline Debugging and Inspection")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2)),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])

    pipe.fit(X, y)

    print("\n[1] named_steps - access steps by name:")
    for name, step in pipe.named_steps.items():
        print(f"    {name}: {type(step).__name__}")

    print("\n[2] Step indices and names:")
    for i, (name, step) in enumerate(pipe.steps):
        print(f"    [{i}] {name}")

    print("\n[3] Inspect intermediate outputs - transform up to step:")
    X_scaled = pipe["scaler"].transform(X[:5])
    print(f"    After scaler (first 5 rows):\n{X_scaled}")

    X_pca = pipe["pca"].transform(X_scaled)
    print(f"\n    After PCA (first 5 rows):\n{X_pca}")

    print("\n[4] Pipeline slice - fit_transform up to step:")
    X_transformed = pipe[:2].fit_transform(X)
    print(f"    Shape after scaler+PCA: {X_transformed.shape}")

    print("\n[5] Check fitted attributes of steps:")
    print(f"    scaler.mean_: {pipe['scaler'].mean_[:2]}...")
    print(f"    pca.explained_variance_ratio_: {pipe['pca'].explained_variance_ratio_}")

    print("\n[6] get_params for pipeline:")
    params = pipe.get_params()
    print(f"    clf__C: {params.get('clf__C')}")
    print(f"    pca__n_components: {params.get('pca__n_components')}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()