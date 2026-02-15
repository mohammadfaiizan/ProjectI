"""
Scikit-learn Input Validation
check_array, check_X_y, check_is_fitted, validation utilities
"""

import numpy as np
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


def main():
    print("=" * 60)
    print("Input Validation Utilities")
    print("=" * 60)

    print("\n[1] check_array - validate feature matrix:")
    X_valid = np.array([[1, 2], [3, 4], [5, 6]])
    X_checked = check_array(X_valid)
    print(f"    Input shape: {X_valid.shape}")
    print(f"    Output: same data, validated")

    print("\n[2] check_array - accept_sparse:")
    from scipy import sparse
    X_sparse = sparse.csr_matrix([[1, 0], [0, 2]])
    X_s = check_array(X_sparse, accept_sparse=True)
    print(f"    Sparse input accepted when accept_sparse=True")

    print("\n[3] check_array - ensure_2d:")
    X_1d = np.array([1, 2, 3])
    X_2d = check_array(X_1d, ensure_2d=True)
    print(f"    Reshaped to 2D: {X_2d.shape}")

    print("\n[4] check_X_y - validate X and y together:")
    X, y = load_iris(return_X_y=True)
    X_ck, y_ck = check_X_y(X[:10], y[:10])
    print(f"    check_X_y ensures consistent n_samples")
    print(f"    X: {X_ck.shape}, y: {y_ck.shape}")

    print("\n[5] check_is_fitted - verify estimator is fitted:")
    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X[:50], y[:50])
    check_is_fitted(clf)
    print("    check_is_fitted(clf) passes when fitted")

    print("\n[6] check_is_fitted raises when unfitted:")
    clf_unfit = LogisticRegression()
    try:
        check_is_fitted(clf_unfit)
        print("    (Should not reach here)")
    except Exception as e:
        print(f"    NotFittedError: {type(e).__name__}")

    print("\n[7] attributes parameter for check_is_fitted:")
    check_is_fitted(clf, attributes=["coef_", "classes_"])
    print("    Can specify required attributes")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
