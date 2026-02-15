"""
Scikit-learn model versioning: strategies, metadata alongside models
"""

import json
import joblib
import os
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def main():
    print("=" * 60)
    print("Model Versioning: strategies, metadata")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    score = (model.predict(X_test) == y_test).mean()

    print("\n[1] Versioned save (model + metadata):")
    version = "v1.0.0"
    metadata = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "model_type": type(model).__name__,
        "train_samples": len(X_train),
        "test_accuracy": float(score),
        "sklearn_version": "1.x",
    }
    model_path = f"model_{version.replace('.', '_')}.joblib"
    meta_path = f"model_{version.replace('.', '_')}_meta.json"
    joblib.dump(model, model_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"    Saved {model_path} and {meta_path}")

    print("\n[2] Load metadata without loading model:")
    with open(meta_path, "r") as f:
        loaded_meta = json.load(f)
    print(f"    Version: {loaded_meta['version']}")
    print(f"    Test accuracy: {loaded_meta['test_accuracy']:.4f}")

    print("\n[3] Semantic versioning pattern:")
    versions = ["v1.0.0", "v1.1.0", "v2.0.0"]
    for v in versions:
        print(f"    {v} -> major.minor.patch")

    print("\n[4] Directory-based versioning:")
    dir_name = "models/20250215_001"
    os.makedirs(dir_name, exist_ok=True)
    joblib.dump(model, os.path.join(dir_name, "model.joblib"))
    with open(os.path.join(dir_name, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"    Created {dir_name}/")

    for f in [model_path, meta_path]:
        if os.path.exists(f):
            os.remove(f)
    import shutil
    if os.path.exists("models"):
        shutil.rmtree("models")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
