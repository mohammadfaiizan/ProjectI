"""
Scikit-learn model persistence: joblib.dump, joblib.load, compress parameter
"""

import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def main():
    print("=" * 60)
    print("Joblib Save/Load: dump, load, compress")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    acc_before = (model.predict(X_test) == y_test).mean()
    print(f"\n[1] Model trained, test accuracy: {acc_before:.4f}")

    print("\n[2] joblib.dump (no compression):")
    joblib.dump(model, "model_no_compress.joblib")
    import os
    size_no = os.path.getsize("model_no_compress.joblib")
    print(f"    File size: {size_no} bytes")

    print("\n[3] joblib.dump with compress=3:")
    joblib.dump(model, "model_compress.joblib", compress=3)
    size_comp = os.path.getsize("model_compress.joblib")
    print(f"    File size: {size_comp} bytes")
    print(f"    Compression ratio: {size_no / size_comp:.2f}x")

    print("\n[4] joblib.load and verify:")
    loaded = joblib.load("model_compress.joblib")
    acc_after = (loaded.predict(X_test) == y_test).mean()
    print(f"    Loaded model type: {type(loaded).__name__}")
    print(f"    Test accuracy after load: {acc_after:.4f}")
    print(f"    Predictions match: {(model.predict(X_test) == loaded.predict(X_test)).all()}")

    print("\n[5] compress parameter range (0-9):")
    for c in [0, 3, 6, 9]:
        path = f"model_c{c}.joblib"
        joblib.dump(model, path, compress=c)
        print(f"    compress={c}: {os.path.getsize(path)} bytes")

    for f in ["model_no_compress.joblib", "model_compress.joblib"] + [f"model_c{c}.joblib" for c in [0, 3, 6, 9]]:
        if os.path.exists(f):
            os.remove(f)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
