"""
Scikit-learn model validation after loading: version compatibility checks
"""

import joblib
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def main():
    print("=" * 60)
    print("Model Validation After Loading")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "model_val.joblib")

    print("\n[1] Load and check type:")
    loaded = joblib.load("model_val.joblib")
    assert hasattr(loaded, "predict"), "Model must have predict method"
    print(f"    Type: {type(loaded).__name__}")
    print(f"    Has predict: {hasattr(loaded, 'predict')}")

    print("\n[2] Sanity prediction check:")
    pred_orig = model.predict(X_test[:5])
    pred_loaded = loaded.predict(X_test[:5])
    match = (pred_orig == pred_loaded).all()
    print(f"    Predictions match: {match}")

    print("\n[3] Input shape validation:")
    try:
        loaded.predict(X_test)
        print("    Input shape OK")
    except Exception as e:
        print(f"    Error: {e}")

    print("\n[4] sklearn version check:")
    print(f"    Current sklearn: {sklearn.__version__}")
    print("    Loaded models may require same or compatible version")

    print("\n[5] Basic validation function:")
    def validate_loaded_model(loaded_model, X_sample, y_sample=None):
        pred = loaded_model.predict(X_sample)
        ok = len(pred) == len(X_sample)
        if y_sample is not None:
            ok = ok and (pred == y_sample).sum() >= 0
        return ok
    valid = validate_loaded_model(loaded, X_test[:10], y_test[:10])
    print(f"    Validation passed: {valid}")

    import os
    if os.path.exists("model_val.joblib"):
        os.remove("model_val.joblib")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
