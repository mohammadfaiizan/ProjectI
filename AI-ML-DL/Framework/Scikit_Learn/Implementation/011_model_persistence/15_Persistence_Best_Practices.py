"""
Scikit-learn persistence best practices: model persistence and deployment
"""

import joblib
import json
import os
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def main():
    print("=" * 60)
    print("Persistence Best Practices")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[1] Save full pipeline (preprocessing + model):")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, "pipeline.joblib")
    print("    Pipeline includes scaler and classifier")

    print("\n[2] Version and metadata alongside:")
    meta = {
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "random_state": 42,
        "test_accuracy": float((pipe.predict(X_test) == y_test).mean()),
    }
    with open("pipeline_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[3] Use compress for smaller files:")
    joblib.dump(pipe, "pipeline_compressed.joblib", compress=3)
    s1 = os.path.getsize("pipeline.joblib")
    s2 = os.path.getsize("pipeline_compressed.joblib")
    print(f"    Uncompressed: {s1} bytes, compressed: {s2} bytes")

    print("\n[4] Validate after load:")
    loaded = joblib.load("pipeline_compressed.joblib")
    pred_orig = pipe.predict(X_test[:5])
    pred_loaded = loaded.predict(X_test[:5])
    print(f"    Predictions match: {(pred_orig == pred_loaded).all()}")

    print("\n[5] Best practices summary:")
    print("    - Save pipelines, not just models")
    print("    - Use random_state for reproducibility")
    print("    - Store metadata (version, metrics)")
    print("    - Compress (joblib compress=3)")
    print("    - Validate predictions after load")
    print("    - Pin dependency versions")

    for f in ["pipeline.joblib", "pipeline_compressed.joblib", "pipeline_meta.json"]:
        if os.path.exists(f):
            os.remove(f)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
