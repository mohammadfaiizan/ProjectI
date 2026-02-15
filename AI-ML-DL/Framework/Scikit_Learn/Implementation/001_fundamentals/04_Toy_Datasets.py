"""
Scikit-learn Toy Datasets
load_iris, load_digits, load_wine, load_breast_cancer, load_diabetes
"""

from sklearn.datasets import (
    load_iris,
    load_digits,
    load_wine,
    load_breast_cancer,
    load_diabetes,
)


def main():
    print("=" * 60)
    print("Toy Datasets: Built-in Dataset Loaders")
    print("=" * 60)

    print("\n[1] load_iris - Classification:")
    iris = load_iris()
    print(f"    X shape: {iris.data.shape}")
    print(f"    y shape: {iris.target.shape}")
    print(f"    Feature names: {iris.feature_names}")
    print(f"    Target names: {list(iris.target_names)}")
    print(f"    Classes: {len(iris.target_names)}")

    print("\n[2] load_digits - Classification:")
    digits = load_digits()
    print(f"    X shape: {digits.data.shape}")
    print(f"    Images: 8x8 pixels, flattened to 64 features")
    print(f"    Classes: 0-9 (digits)")

    print("\n[3] load_wine - Classification:")
    wine = load_wine()
    print(f"    X shape: {wine.data.shape}")
    print(f"    Feature names (first 3): {wine.feature_names[:3]}")
    print(f"    Target names: {list(wine.target_names)}")

    print("\n[4] load_breast_cancer - Classification:")
    cancer = load_breast_cancer()
    print(f"    X shape: {cancer.data.shape}")
    print(f"    Target: malignant (0) vs benign (1)")
    print(f"    Class distribution: {dict(zip(*zip(*[(k, (cancer.target==k).sum()) for k in [0,1]])))}")

    print("\n[5] load_diabetes - Regression:")
    diabetes = load_diabetes()
    print(f"    X shape: {diabetes.data.shape}")
    print(f"    y shape: {diabetes.target.shape}")
    print(f"    y range: [{diabetes.target.min():.1f}, {diabetes.target.max():.1f}]")

    print("\n[6] Bunch object structure:")
    print(f"    Keys in iris: {list(iris.keys())}")
    print(f"    DESCR (first 80 chars): {iris.DESCR[:80]}...")

    print("\n[7] Return as tuple (return_X_y=True):")
    X, y = load_iris(return_X_y=True)
    print(f"    X, y = load_iris(return_X_y=True)")
    print(f"    Returns: (data, target) only")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
