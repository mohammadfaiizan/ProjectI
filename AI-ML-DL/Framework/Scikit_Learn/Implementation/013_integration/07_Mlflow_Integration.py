"""
Scikit-learn MLflow integration: autolog, log_model, log_metric patterns
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def main():
    print("=" * 60)
    print("MLflow Integration: autolog, log_model, log_metric")
    print("=" * 60)

    if not MLFLOW_AVAILABLE:
        print("\n[!] MLflow not installed. Install with: pip install mlflow")
        print("    Simulating MLflow patterns without actual logging.")
        simulate_mlflow_patterns()
        return

    print("\n[1] mlflow.sklearn.autolog:")
    mlflow.sklearn.autolog()
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    with mlflow.start_run(run_name="sklearn_rf_demo"):
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")

    print(f"    Test accuracy: {acc:.4f}")
    print(f"    Run logged with autolog")

    print("\n[2] Manual log_metric pattern:")
    with mlflow.start_run(run_name="manual_metrics"):
        clf2 = RandomForestClassifier(n_estimators=20, random_state=42)
        cv_scores = cross_val_score(clf2, X_train, y_train, cv=3)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())
        clf2.fit(X_train, y_train)
        mlflow.sklearn.log_model(clf2, "model")
    print(f"    CV mean: {cv_scores.mean():.4f}, std: {cv_scores.std():.4f}")

    print("\n[3] log_param and log_params:")
    with mlflow.start_run():
        mlflow.log_params({"n_estimators": 15, "max_depth": 5})
        clf3 = RandomForestClassifier(n_estimators=15, max_depth=5, random_state=42)
        clf3.fit(X_train, y_train)
        mlflow.sklearn.log_model(clf3, "model")
    print("    Params logged")

    print("\n" + "=" * 60)


def simulate_mlflow_patterns():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"\n[Simulated] Test accuracy: {acc:.4f}")
    print("[Simulated] Would call: mlflow.sklearn.autolog(), mlflow.log_metric(), mlflow.sklearn.log_model()")


if __name__ == "__main__":
    main()
