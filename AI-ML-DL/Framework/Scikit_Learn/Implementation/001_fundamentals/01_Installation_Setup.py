"""
Scikit-learn Installation and Setup
Demonstrates version check, dependencies, and show_versions()
"""

import sys

def main():
    print("=" * 60)
    print("Scikit-learn Installation and Setup")
    print("=" * 60)

    print("\n[1] Python version:")
    print(f"    {sys.version}")

    print("\n[2] Importing sklearn...")
    import sklearn
    print(f"    sklearn imported successfully")

    print("\n[3] Scikit-learn version:")
    print(f"    {sklearn.__version__}")

    print("\n[4] Key dependencies check:")
    try:
        import numpy as np
        print(f"    NumPy: {np.__version__}")
    except ImportError:
        print("    NumPy: NOT INSTALLED")

    try:
        import scipy
        print(f"    SciPy: {scipy.__version__}")
    except ImportError:
        print("    SciPy: NOT INSTALLED")

    try:
        import joblib
        print(f"    joblib: {joblib.__version__}")
    except ImportError:
        print("    joblib: NOT INSTALLED")

    try:
        import threadpoolctl
        print(f"    threadpoolctl: installed")
    except ImportError:
        print("    threadpoolctl: NOT INSTALLED")

    print("\n[5] show_versions() output:")
    print("-" * 40)
    sklearn.show_versions()
    print("-" * 40)

    print("\n[6] Basic functionality test:")
    from sklearn.linear_model import LinearRegression
    import numpy as np
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    model = LinearRegression().fit(X, y)
    pred = model.predict([[4]])
    print(f"    LinearRegression test: predict(4) = {pred[0]:.2f}")
    print("    Basic functionality: OK")

    print("\n" + "=" * 60)
    print("Setup verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
