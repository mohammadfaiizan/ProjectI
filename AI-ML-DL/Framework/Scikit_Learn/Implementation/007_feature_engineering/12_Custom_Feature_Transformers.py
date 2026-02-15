"""
Scikit-learn Custom transformers using BaseEstimator + TransformerMixin
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(np.abs(X) + 1e-8)

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.columns]

class ThresholdBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (X > self.threshold).astype(float)

np.random.seed(42)
X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

print("--- LogTransformer ---")
log_t = LogTransformer()
print("Transformed:", log_t.fit_transform(X))

print("\n--- ColumnSelector ---")
col_sel = ColumnSelector(columns=[0])
print("Selected column:", col_sel.fit_transform(X))

print("\n--- ThresholdBinarizer ---")
bin_t = ThresholdBinarizer(threshold=3.0)
print("Binarized:", bin_t.fit_transform(X))

print("\n--- Pipeline compatibility ---")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pipe = Pipeline([
    ('log', LogTransformer()),
    ('scale', StandardScaler())
])
X_pipe = pipe.fit_transform(X)
print("Pipeline output shape:", X_pipe.shape)
print("Pipeline output:", np.round(X_pipe, 3))
