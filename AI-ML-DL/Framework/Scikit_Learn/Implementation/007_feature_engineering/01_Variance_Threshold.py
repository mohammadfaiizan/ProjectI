"""
Scikit-learn VarianceThreshold: removing low-variance features
"""
import numpy as np
from sklearn.feature_selection import VarianceThreshold

np.random.seed(42)
X = np.array([
    [1.0, 2.0, 0.0, 0.0],
    [1.1, 2.1, 0.0, 0.0],
    [1.2, 2.2, 0.0, 0.0],
    [1.3, 2.3, 0.0, 0.0],
])
print("Original data (4 features):")
print(X)
print("Variances:", np.var(X, axis=0))

print("\n--- threshold=0.0 (remove constant features) ---")
selector = VarianceThreshold(threshold=0.0)
X_selected = selector.fit_transform(X)
print("Selected shape:", X_selected.shape)
print("Selected features:", selector.get_support())
print("Transformed data:", X_selected)

print("\n--- threshold=0.01 ---")
selector2 = VarianceThreshold(threshold=0.01)
X_selected2 = selector2.fit_transform(X)
print("Selected shape:", X_selected2.shape)
print("Selected features:", selector2.get_support())

print("\n--- threshold=0.1 ---")
selector3 = VarianceThreshold(threshold=0.1)
X_selected3 = selector3.fit_transform(X)
print("Selected shape:", X_selected3.shape)
print("Selected features:", selector3.get_support())
print("Feature names (if provided):", selector3.get_feature_names_out())
