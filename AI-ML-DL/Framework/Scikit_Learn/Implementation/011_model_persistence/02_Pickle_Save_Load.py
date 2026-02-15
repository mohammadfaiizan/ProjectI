"""
Scikit-learn model persistence: pickle.dump, pickle.load, protocol parameter
"""

import pickle
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def main():
    print("=" * 60)
    print("Pickle Save/Load: dump, load, protocol")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    acc_before = (model.predict(X_test) == y_test).mean()
    print(f"\n[1] Model trained, test accuracy: {acc_before:.4f}")

    print("\n[2] pickle.dump with protocol=4 (Python 3.4+):")
    with open("model_p4.pkl", "wb") as f:
        pickle.dump(model, f, protocol=4)
    size_p4 = os.path.getsize("model_p4.pkl")
    print(f"    File size: {size_p4} bytes")

    print("\n[3] pickle.dump with protocol=5 (Python 3.8+):")
    with open("model_p5.pkl", "wb") as f:
        pickle.dump(model, f, protocol=5)
    size_p5 = os.path.getsize("model_p5.pkl")
    print(f"    File size: {size_p5} bytes")

    print("\n[4] pickle.load and verify:")
    with open("model_p4.pkl", "rb") as f:
        loaded = pickle.load(f)
    acc_after = (loaded.predict(X_test) == y_test).mean()
    print(f"    Loaded model type: {type(loaded).__name__}")
    print(f"    Test accuracy after load: {acc_after:.4f}")

    print("\n[5] protocol parameter summary:")
    print("    protocol 0-3: ASCII, larger files")
    print("    protocol 4: binary, Python 3.4+")
    print("    protocol 5: binary, Python 3.8+, out-of-band data")

    for f in ["model_p4.pkl", "model_p5.pkl"]:
        if os.path.exists(f):
            os.remove(f)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
