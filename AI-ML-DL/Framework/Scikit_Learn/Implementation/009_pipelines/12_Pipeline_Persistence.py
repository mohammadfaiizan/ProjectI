"""
Scikit-learn pipeline persistence: joblib dump/load pipelines
"""

import tempfile
import os
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("Pipeline Persistence: joblib dump/load")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    pred_orig = pipe.predict(X_test)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "pipeline.joblib")

        print("\n[1] joblib.dump:")
        joblib.dump(pipe, path)
        print(f"    Saved to: {path}")
        print(f"    File size: {os.path.getsize(path)} bytes")

        print("\n[2] joblib.load:")
        pipe_loaded = joblib.load(path)
        pred_loaded = pipe_loaded.predict(X_test)
        print(f"    Predictions match: {np.array_equal(pred_orig, pred_loaded)}")

        print("\n[3] compress option:")
        path_compressed = os.path.join(tmpdir, "pipeline_compressed.joblib")
        joblib.dump(pipe, path_compressed, compress=3)
        size_orig = os.path.getsize(path)
        size_comp = os.path.getsize(path_compressed)
        print(f"    Original size: {size_orig}, Compressed: {size_comp}")

        print("\n[4] Load and use in new process (simulated):")
        pipe_new = joblib.load(path_compressed)
        acc = accuracy_score(y_test, pipe_new.predict(X_test))
        print(f"    Loaded pipeline accuracy: {acc:.4f}")

    print("\n[5] Persistence of nested pipelines:")
    nested = Pipeline([
        ("preprocess", Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=42)),
        ])),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])
    nested.fit(X_train, y_train)
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        joblib.dump(nested, f.name)
        nested_loaded = joblib.load(f.name)
        print(f"    Nested pipeline loads correctly: {np.allclose(nested_loaded.predict(X_test), nested.predict(X_test))}")
        os.unlink(f.name)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
