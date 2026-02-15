"""
Scikit-learn custom estimator advanced: input validation, get_params, set_params
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_X_y
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class AdvancedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.1, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def get_params(self, deep=True):
        return {"alpha": self.alpha, "fit_intercept": self.fit_intercept}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=False)
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        return X @ self.coef_


def main():
    print("=" * 60)
    print("Custom Estimator Advanced: Validation, get_params, set_params")
    print("=" * 60)

    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    print("\n[1] Fit and predict with input validation:")
    reg = AdvancedRegressor(alpha=0.1, fit_intercept=True)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    print(f"    RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.4f}")

    print("\n[2] get_params:")
    params = reg.get_params()
    print(f"    {params}")

    print("\n[3] set_params and refit:")
    reg.set_params(alpha=0.5)
    reg.fit(X_train, y_train)
    pred2 = reg.predict(X_test)
    print(f"    After set_params, RMSE: {np.sqrt(mean_squared_error(y_test, pred2)):.4f}")

    print("\n[4] Works with GridSearchCV (clone uses get_params/set_params):")
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(AdvancedRegressor(), {"alpha": [0.01, 0.1, 1.0]}, cv=3)
    gs.fit(X_train, y_train)
    print(f"    Best alpha: {gs.best_params_['alpha']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
