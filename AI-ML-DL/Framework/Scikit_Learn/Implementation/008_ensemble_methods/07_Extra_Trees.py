"""
Scikit-learn ExtraTreesClassifier and ExtraTreesRegressor
"""

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


def main():
    print("=" * 60)
    print("ExtraTreesClassifier and ExtraTreesRegressor")
    print("=" * 60)

    X_clf, y_clf = load_iris(return_X_y=True)
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )

    print("\n[1] ExtraTreesClassifier vs RandomForest:")
    rf = ExtraTreesClassifier(n_estimators=50, random_state=42)
    rf.fit(X_clf_train, y_clf_train)
    acc_rf = accuracy_score(y_clf_test, rf.predict(X_clf_test))
    print(f"    ExtraTreesClassifier: Accuracy = {acc_rf:.4f}")

    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=50, random_state=42)
    rfc.fit(X_clf_train, y_clf_train)
    acc_rfc = accuracy_score(y_clf_test, rfc.predict(X_clf_test))
    print(f"    RandomForestClassifier: Accuracy = {acc_rfc:.4f}")

    print("\n[2] ExtraTreesClassifier - n_estimators and max_depth:")
    for n, depth in [(25, 5), (50, 10), (100, None)]:
        et = ExtraTreesClassifier(n_estimators=n, max_depth=depth, random_state=42)
        et.fit(X_clf_train, y_clf_train)
        acc = accuracy_score(y_clf_test, et.predict(X_clf_test))
        print(f"    n={n}, max_depth={depth}: Accuracy = {acc:.4f}")

    X_reg, y_reg = make_regression(n_samples=300, n_features=10, noise=15, random_state=42)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    print("\n[3] ExtraTreesRegressor:")
    et_reg = ExtraTreesRegressor(n_estimators=50, max_depth=10, random_state=42)
    et_reg.fit(X_reg_train, y_reg_train)
    r2 = r2_score(y_reg_test, et_reg.predict(X_reg_test))
    print(f"    n_estimators=50: R2 = {r2:.4f}")

    print("\n[4] feature_importances_:")
    et = ExtraTreesClassifier(n_estimators=100, random_state=42)
    et.fit(X_clf_train, y_clf_train)
    for i, imp in enumerate(et.feature_importances_):
        print(f"    Feature {i}: {imp:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
