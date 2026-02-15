"""
Production-ready sklearn: validation, monitoring, versioning
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array


def validate_input(X, expected_features=None):
    X = check_array(X, accept_sparse=False, dtype=np.float64)
    if expected_features is not None and X.shape[1] != expected_features:
        raise ValueError(f"Expected {expected_features} features, got {X.shape[1]}")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or Inf")
    return X


def main():
    print("=" * 60)
    print("Production Patterns: validation, monitoring, versioning")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    print("\n[1] Pipeline for production:")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    n_features = X_train.shape[1]
    print(f"    Expected features: {n_features}")

    print("\n[2] Input validation before predict:")
    X_valid = validate_input(X_test[:5], expected_features=n_features)
    pred = pipe.predict(X_valid)
    print(f"    Valid input: {X_valid.shape}, pred: {pred.shape}")

    print("\n[3] Model versioning with joblib:")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    version = "v1"
    path = model_dir / f"pipe_{version}.joblib"
    joblib.dump({"pipe": pipe, "version": version, "n_features": n_features}, path)
    loaded = joblib.load(path)
    print(f"    Saved: {path}")
    print(f"    Loaded version: {loaded['version']}")

    print("\n[4] Monitoring metrics:")
    pred_all = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred_all)
    print(f"    Test accuracy: {acc:.4f}")
    print("    Log: latency, throughput, prediction distribution")

    print("\n[5] Graceful degradation:")
    try:
        bad_X = np.array([[1, 2, np.nan, 4]])
        validate_input(bad_X)
    except ValueError as e:
        print(f"    Caught: {type(e).__name__}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
