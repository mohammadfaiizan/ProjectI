"""
Scikit-learn large models: handling large models, memory-mapped loading
"""

import joblib
import numpy as np
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def main():
    print("=" * 60)
    print("Streaming Large Models: memory-mapped loading")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "model_large.joblib")

    print("\n[1] Standard load (full into memory):")
    loaded = joblib.load("model_large.joblib")
    print(f"    Model type: {type(loaded).__name__}")
    print(f"    n_estimators: {loaded.n_estimators}")

    print("\n[2] mmap_mode for large arrays (joblib):")
    try:
        loaded_mmap = joblib.load("model_large.joblib", mmap_mode="r")
        pred = loaded_mmap.predict(X_test[:5])
        print(f"    mmap_mode='r' works: {len(pred) == 5}")
    except Exception as e:
        print(f"    mmap note: {type(e).__name__}")

    print("\n[3] Large model strategies:")
    print("    - mmap_mode='r' for read-only, shared memory")
    print("    - Chunked prediction for very large datasets")
    print("    - Consider model compression (smaller n_estimators)")

    print("\n[4] Chunked prediction pattern:")
    chunk_size = 10
    preds = []
    for i in range(0, len(X_test), chunk_size):
        chunk = X_test[i : i + chunk_size]
        preds.extend(loaded.predict(chunk))
    print(f"    Chunked preds count: {len(preds)}")

    print("\n[5] Memory considerations:")
    print("    - joblib uses pickle; large models = large files")
    print("    - ONNX can reduce size for deployment")
    print("    - Use compress=3+ for disk space")

    if os.path.exists("model_large.joblib"):
        os.remove("model_large.joblib")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
