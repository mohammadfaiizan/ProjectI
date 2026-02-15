"""
Scikit-learn Normalizer: per-sample scaling (l2, l1, max)
"""
import numpy as np
from sklearn.preprocessing import Normalizer, normalize

np.random.seed(42)
X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

print("Original data:")
print(X)

print("\n--- Normalizer (l2, default) ---")
norm_l2 = Normalizer(norm="l2")
X_l2 = norm_l2.fit_transform(X)
print("L2 normalized (unit norm per row):")
print(X_l2)
print("Row norms:", np.linalg.norm(X_l2, axis=1))

print("\n--- Normalizer (l1) ---")
norm_l1 = Normalizer(norm="l1")
X_l1 = norm_l1.fit_transform(X)
print("L1 normalized (sum to 1 per row):")
print(X_l1)
print("Row sums:", X_l1.sum(axis=1))

print("\n--- Normalizer (max) ---")
norm_max = Normalizer(norm="max")
X_max = norm_max.fit_transform(X)
print("Max normalized (divide by max per row):")
print(X_max)

print("\n--- normalize() function (no fit) ---")
X_fn = normalize(X, norm="l2")
print("normalize(X, norm='l2'):")
print(X_fn)