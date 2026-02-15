"""
Scikit-learn SequentialFeatureSelector (direction=forward/backward)
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=12, n_informative=5, random_state=42)

print("--- Forward selection ---")
estimator = LogisticRegression(max_iter=1000, random_state=42)
sfs_forward = SequentialFeatureSelector(
    estimator=estimator,
    n_features_to_select=5,
    direction='forward',
    cv=5
)
sfs_forward.fit(X, y)
print("Selected features (forward):", sfs_forward.get_support(indices=True))
X_forward = sfs_forward.transform(X)
print("Transformed shape:", X_forward.shape)

print("\n--- Backward selection ---")
sfs_backward = SequentialFeatureSelector(
    estimator=estimator,
    n_features_to_select=5,
    direction='backward',
    cv=5
)
sfs_backward.fit(X, y)
print("Selected features (backward):", sfs_backward.get_support(indices=True))

print("\n--- Forward with n_features_to_select='auto' (5-fold CV) ---")
sfs_auto = SequentialFeatureSelector(
    estimator=estimator,
    n_features_to_select='auto',
    direction='forward',
    cv=5,
    scoring='accuracy'
)
sfs_auto.fit(X, y)
print("Auto-selected count:", sfs_auto.n_features_to_select_)
print("Selected features:", sfs_auto.get_support(indices=True))
