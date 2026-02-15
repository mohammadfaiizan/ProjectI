"""
Scikit-learn StackingRegressor: estimators, final_estimator, cv
"""

import numpy as np
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    print("=" * 60)
    print("StackingRegressor: estimators, final_estimator, cv")
    print("=" * 60)

    X, y = make_regression(n_samples=300, n_features=10, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    estimators = [
        ("lr", LinearRegression()),
        ("ridge", Ridge(alpha=1.0)),
        ("dt", DecisionTreeRegressor(max_depth=10, random_state=42)),
        ("svr", SVR(kernel="rbf", C=1.0)),
    ]

    print("\n[1] StackingRegressor - default final_estimator (Ridge):")
    stk = StackingRegressor(estimators=estimators, cv=5)
    stk.fit(X_train, y_train)
    r2 = r2_score(y_test, stk.predict(X_test))
    mse = mean_squared_error(y_test, stk.predict(X_test))
    print(f"    cv=5: R2 = {r2:.4f}, MSE = {mse:.2f}")

    print("\n[2] Custom final_estimator:")
    stk_custom = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=10.0),
        cv=5
    )
    stk_custom.fit(X_train, y_train)
    r2 = r2_score(y_test, stk_custom.predict(X_test))
    print(f"    final_estimator=Ridge(alpha=10): R2 = {r2:.4f}")

    print("\n[3] cv - cross-validation folds:")
    for cv in [3, 5, 10]:
        stk = StackingRegressor(estimators=estimators, cv=cv)
        stk.fit(X_train, y_train)
        r2 = r2_score(y_test, stk.predict(X_test))
        print(f"    cv={cv}: R2 = {r2:.4f}")

    print("\n[4] named_estimators_ and final_estimator_:")
    stk.fit(X_train, y_train)
    print(f"    Base estimators: {list(stk.named_estimators_.keys())}")
    print(f"    Final estimator type: {type(stk.final_estimator_).__name__}")

    print("\n[5] transform - meta-features for test set:")
    meta = stk.transform(X_test[:5])
    print(f"    Meta-features shape (5 samples, {meta.shape[1]} base models): {meta.shape}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
