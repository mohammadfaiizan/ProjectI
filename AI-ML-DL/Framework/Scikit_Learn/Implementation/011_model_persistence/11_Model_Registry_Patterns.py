"""
Scikit-learn model registry patterns: registry patterns for model management
"""

import json
import joblib
import os
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def main():
    print("=" * 60)
    print("Model Registry Patterns")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)

    print("\n[1] Simple file-based registry:")
    registry_dir = Path("model_registry")
    registry_dir.mkdir(exist_ok=True)
    version = "v1"
    model_path = registry_dir / f"model_{version}.joblib"
    meta_path = registry_dir / f"model_{version}_meta.json"
    joblib.dump(model, model_path)
    meta = {"version": version, "accuracy": 0.95}
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"    Saved to {registry_dir}/")

    print("\n[2] Registry index (list versions):")
    index = {"versions": ["v1"], "latest": "v1"}
    with open(registry_dir / "index.json", "w") as f:
        json.dump(index, f)
    print(f"    Index: {index}")

    print("\n[3] Load by version:")
    loaded = joblib.load(registry_dir / "model_v1.joblib")
    print(f"    Loaded model_{index['latest']}.joblib")

    print("\n[4] Registry pattern components:")
    print("    - Versioned model files")
    print("    - Metadata per version")
    print("    - Index pointing to latest")
    print("    - Optional: staging, production tags")

    print("\n[5] MLflow-style concepts:")
    print("    - run_id, experiment_id")
    print("    - model uri, artifact path")
    print("    - Use MLflow for full registry")

    if registry_dir.exists():
        for f in registry_dir.glob("*"):
            f.unlink()
        registry_dir.rmdir()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
