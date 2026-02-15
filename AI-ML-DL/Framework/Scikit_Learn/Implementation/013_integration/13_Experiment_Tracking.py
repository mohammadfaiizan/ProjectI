"""
Experiment tracking patterns: MLflow, manual logging
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def manual_log_run(config, metrics, artifacts_dir="runs"):
    Path(artifacts_dir).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(artifacts_dir) / ts
    run_dir.mkdir()
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return str(run_dir)


def main():
    print("=" * 60)
    print("Experiment Tracking: MLflow, manual logging")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    print("\n[1] Manual experiment logging:")
    config = {"model": "RandomForest", "n_estimators": 50, "random_state": 42}
    clf = RandomForestClassifier(**config)
    clf.fit(X_train, y_train)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=3)
    pred = clf.predict(X_test)
    metrics = {
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "test_accuracy": float(accuracy_score(y_test, pred))
    }
    run_path = manual_log_run(config, metrics)
    print(f"    Run saved to: {run_path}")
    print(f"    Metrics: {metrics}")

    print("\n[2] Compare multiple configs:")
    results = []
    for n_est in [10, 50, 100]:
        clf2 = RandomForestClassifier(n_estimators=n_est, random_state=42)
        scores = cross_val_score(clf2, X_train, y_train, cv=3)
        results.append({"n_estimators": n_est, "cv_mean": scores.mean()})
    best = max(results, key=lambda r: r["cv_mean"])
    print(f"    Best n_estimators: {best['n_estimators']} (cv_mean={best['cv_mean']:.4f})")

    print("\n[3] MLflow pattern (when installed):")
    print("    with mlflow.start_run():")
    print("        mlflow.log_params(config)")
    print("        mlflow.log_metrics(metrics)")
    print("        mlflow.sklearn.log_model(clf, 'model')")

    print("\n[4] Structured logging for reproducibility:")
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "data_hash": hash(X_train.tobytes()) % 10**8,
        "config": config,
        "metrics": metrics
    }
    print(f"    Log entry keys: {list(log_entry.keys())}")

    print("\n[5] Version tracking:")
    print("    - sklearn.__version__")
    print("    - Model class name")
    print("    - Config hash for deduplication")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
