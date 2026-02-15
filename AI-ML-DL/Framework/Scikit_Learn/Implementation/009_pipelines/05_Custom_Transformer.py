"""
Scikit-learn custom transformer: BaseEstimator + TransformerMixin
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, offset=1.0):
        self.offset = offset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(np.abs(X) + self.offset)


class ThresholdFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.mask_ = None

    def fit(self, X, y=None):
        self.mask_ = np.var(X, axis=0) > self.threshold
        return self

    def transform(self, X):
        return X[:, self.mask_]


def main():
    print("=" * 60)
    print("Custom Transformer: BaseEstimator + TransformerMixin")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[1] LogTransformer:")
    log_t = LogTransformer(offset=0.1)
    X_log = log_t.fit_transform(X_train)
    print(f"    Input range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"    Output range: [{X_log.min():.2f}, {X_log.max():.2f}]")

    print("\n[2] ThresholdFilter:")
    thresh_t = ThresholdFilter(threshold=0.01)
    X_thresh = thresh_t.fit_transform(X_train)
    print(f"    Features kept: {thresh_t.mask_.sum()} / {X_train.shape[1]}")

    print("\n[3] Pipeline with custom transformers:")
    pipe = Pipeline([
        ("log", LogTransformer(offset=0.1)),
        ("filter", ThresholdFilter(threshold=0.01)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(f"    Pipeline accuracy: {accuracy_score(y_test, pred):.4f}")

    print("\n[4] get_params for custom transformer:")
    params = pipe.get_params()
    print(f"    log__offset: {params.get('log__offset')}")
    print(f"    filter__threshold: {params.get('filter__threshold')}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
