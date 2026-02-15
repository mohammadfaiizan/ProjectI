"""
Scikit-learn model metadata: storing metadata with models, custom attributes
"""

import joblib
import json
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def main():
    print("=" * 60)
    print("Model Metadata: custom attributes")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)

    print("\n[1] Bundle model + metadata in dict:")
    bundle = {
        "model": model,
        "metadata": {
            "version": "1.0",
            "train_samples": len(X_train),
            "feature_names": ["sepal_len", "sepal_wid", "petal_len", "petal_wid"],
            "created": "2025-02-15",
        },
    }
    joblib.dump(bundle, "model_bundle.joblib")
    print("    Saved model + metadata bundle")

    print("\n[2] Load and access metadata:")
    loaded_bundle = joblib.load("model_bundle.joblib")
    meta = loaded_bundle["metadata"]
    print(f"    Version: {meta['version']}")
    print(f"    Features: {meta['feature_names']}")

    print("\n[3] Custom attributes on estimator (not recommended):")
    model.custom_attr = "my_value"
    joblib.dump(model, "model_custom.joblib")
    loaded = joblib.load("model_custom.joblib")
    print(f"    custom_attr: {getattr(loaded, 'custom_attr', 'N/A')}")

    print("\n[4] Metadata best practices:")
    print("    - Use separate dict/file for metadata")
    print("    - Include: version, features, metrics, timestamp")
    print("    - Avoid mutating estimator __dict__")

    print("\n[5] Schema for metadata:")
    schema = ["version", "model_type", "features", "metrics", "created_at"]
    print(f"    Recommended keys: {schema}")

    for f in ["model_bundle.joblib", "model_custom.joblib"]:
        if os.path.exists(f):
            os.remove(f)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
