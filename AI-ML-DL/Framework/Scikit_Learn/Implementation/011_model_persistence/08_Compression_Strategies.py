"""
Scikit-learn compression strategies: gzip, bz2, lzma, joblib compress levels
"""

import joblib
import gzip
import pickle
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def main():
    print("=" * 60)
    print("Compression Strategies: gzip, bz2, lzma, joblib")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    print("\n[1] joblib compress levels (0-9):")
    sizes = {}
    for c in [0, 3, 6, 9]:
        path = f"model_joblib_c{c}.joblib"
        joblib.dump(model, path, compress=c)
        sizes[c] = os.path.getsize(path)
        print(f"    compress={c}: {sizes[c]} bytes")

    print("\n[2] pickle + gzip:")
    with gzip.open("model_gzip.pkl.gz", "wb") as f:
        pickle.dump(model, f)
    size_gz = os.path.getsize("model_gzip.pkl.gz")
    print(f"    Size: {size_gz} bytes")

    print("\n[3] Load from gzip:")
    with gzip.open("model_gzip.pkl.gz", "rb") as f:
        loaded = pickle.load(f)
    acc = (loaded.predict(X_test) == y_test).mean()
    print(f"    Test accuracy: {acc:.4f}")

    print("\n[4] Compression comparison:")
    base = sizes[0]
    for c, s in sizes.items():
        ratio = base / s if s > 0 else 0
        print(f"    joblib compress={c}: {s} bytes, ratio {ratio:.2f}x")

    for c in sizes:
        p = f"model_joblib_c{c}.joblib"
        if os.path.exists(p):
            os.remove(p)
    if os.path.exists("model_gzip.pkl.gz"):
        os.remove("model_gzip.pkl.gz")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
