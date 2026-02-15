"""
Scikit-learn PartialDependenceDisplay for ensembles
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


def main():
    print("=" * 60)
    print("PartialDependenceDisplay for Ensembles")
    print("=" * 60)

    X_clf, y_clf = load_iris(return_X_y=True)
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )

    print("\n[1] RandomForestClassifier - PDP for features 0 and 2:")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_clf_train, y_clf_train)
    disp = PartialDependenceDisplay.from_estimator(
        rf, X_clf_train, features=[0, 2], kind="average"
    )
    print(f"    PDP computed for features [0, 2]")
    print(f"    Display object: {type(disp).__name__}")

    print("\n[2] Two-way PDP (feature interaction):")
    disp2 = PartialDependenceDisplay.from_estimator(
        rf, X_clf_train, features=[(0, 2)], kind="average"
    )
    print(f"    Two-way PDP for features (0, 2)")

    X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    print("\n[3] GradientBoostingRegressor - PDP:")
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_reg_train, y_reg_train)
    disp_gb = PartialDependenceDisplay.from_estimator(
        gb, X_reg_train, features=[0, 5], kind="average"
    )
    print(f"    PDP for features [0, 5]")

    print("\n[4] partial_dependence function - raw values:")
    from sklearn.inspection import partial_dependence
    pd_values, grid = partial_dependence(rf, X_clf_train, features=[0])
    print(f"    partial_dependence output shape: {pd_values.shape}")
    print(f"    Grid points for feature 0: {len(grid[0])}")

    print("\n[5] kind='individual' vs kind='average':")
    disp_avg = PartialDependenceDisplay.from_estimator(
        rf, X_clf_train, features=[0], kind="average"
    )
    disp_ind = PartialDependenceDisplay.from_estimator(
        rf, X_clf_train, features=[0], kind="individual"
    )
    print(f"    kind='average': single curve per target class")
    print(f"    kind='individual': ICE curves per sample")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
