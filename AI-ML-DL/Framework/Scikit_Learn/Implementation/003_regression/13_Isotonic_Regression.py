"""
Scikit-learn Isotonic Regression: IsotonicRegression (increasing)
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def main():
    print("=" * 60)
    print("Isotonic Regression: IsotonicRegression (increasing)")
    print("=" * 60)

    np.random.seed(42)
    X = np.sort(np.random.uniform(0, 10, 100))
    y = np.sin(X) + np.random.normal(0, 0.2, 100)

    X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y, test_size=0.2, random_state=42)

    print("\n[1] IsotonicRegression - monotonic fit:")
    iso = IsotonicRegression(increasing=True)
    iso.fit(X_train.ravel(), y_train)
    y_pred = iso.predict(X_test.ravel())
    print(f"    increasing=True (default)")
    print(f"    Test MSE: {mean_squared_error(y_test, y_pred):.4f}")

    print("\n[2] increasing parameter:")
    iso_auto = IsotonicRegression(increasing="auto")
    iso_auto.fit(X_train.ravel(), y_train)
    print(f"    increasing='auto': MSE = {mean_squared_error(y_test, iso_auto.predict(X_test.ravel())):.4f}")

    print("\n[3] out_of_bounds handling:")
    iso_ext = IsotonicRegression(out_of_bounds="clip")
    iso_ext.fit(X_train.ravel(), y_train)
    X_ext = np.array([-1, 0, 5, 11, 12])
    pred_ext = iso_ext.predict(X_ext)
    print(f"    out_of_bounds='clip': predict on [-1,0,5,11,12] = {pred_ext}")

    print("\n[4] X_thresholds_ and y_thresholds_:")
    print(f"    Number of thresholds: {len(iso.X_thresholds_)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
