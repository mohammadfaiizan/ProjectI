"""
Scikit-learn model persistence: skops.io (save, load, get_untrusted_types) for secure serialization
"""

import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    import skops.io as sio
    SKOPS_AVAILABLE = True
except ImportError:
    SKOPS_AVAILABLE = False


def main():
    print("=" * 60)
    print("Skops Save/Load: secure serialization")
    print("=" * 60)

    if not SKOPS_AVAILABLE:
        print("\n[!] skops not installed. Run: pip install skops")
        return

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    print(f"\n[1] Model trained, test accuracy: {(model.predict(X_test) == y_test).mean():.4f}")

    print("\n[2] skops.io.dump (save):")
    sio.dump(model, "model_skops.skops")
    print(f"    File size: {os.path.getsize('model_skops.skops')} bytes")

    print("\n[3] skops.io.load (trusted types only):")
    loaded = sio.load("model_skops.skops", trusted=True)
    print(f"    Loaded type: {type(loaded).__name__}")
    print(f"    Test accuracy: {(loaded.predict(X_test) == y_test).mean():.4f}")

    print("\n[4] get_untrusted_types (security check):")
    untrusted = sio.get_untrusted_types("model_skops.skops")
    print(f"    Untrusted types: {untrusted}")

    print("\n[5] load with trusted=False (raises if untrusted):")
    try:
        loaded_strict = sio.load("model_skops.skops", trusted=False)
        print("    Load succeeded (all types trusted)")
    except Exception as e:
        print(f"    Load failed: {type(e).__name__}")

    if os.path.exists("model_skops.skops"):
        os.remove("model_skops.skops")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
