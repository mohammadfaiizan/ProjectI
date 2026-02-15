"""
Scikit-learn pipeline caching: memory parameter, caching transforms
"""

import tempfile
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("Pipeline Caching: memory parameter")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n[1] Pipeline with memory='{tmpdir}':")
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("clf", LogisticRegression(max_iter=500, random_state=42)),
        ], memory=tmpdir)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        print(f"    Accuracy: {accuracy_score(y_test, pred):.4f}")

        print("\n[2] Cached transformers (joblib):")
        import os
        cache_files = [f for f in os.listdir(tmpdir) if f.endswith(".pkl")]
        print(f"    Cache directory contains: {len(cache_files)} joblib files")

    print("\n[3] Pipeline without memory (no caching):")
    pipe_no_cache = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])
    pipe_no_cache.fit(X_train, y_train)
    pred_nc = pipe_no_cache.predict(X_test)
    print(f"    Accuracy: {accuracy_score(y_test, pred_nc):.4f}")

    print("\n[4] memory with joblib.Memory:")
    from joblib import Memory
    mem = Memory(location=tempfile.mkdtemp(), verbose=0)
    pipe_mem = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ], memory=mem)
    pipe_mem.fit(X_train, y_train)
    print(f"    Memory location: {mem.location}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
