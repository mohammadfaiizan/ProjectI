"""
Scikit-learn StandardScaler: fit, transform, mean_, scale_, inverse_transform
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])

print("Original data:")
print(X)

scaler = StandardScaler()
scaler.fit(X)

print("\nFitted attributes:")
print("mean_:", scaler.mean_)
print("scale_ (std):", scaler.scale_)

X_scaled = scaler.transform(X)
print("\nTransformed data (z-scores):")
print(X_scaled)

print("\nMean of scaled data:", X_scaled.mean(axis=0))
print("Std of scaled data:", X_scaled.std(axis=0))

X_restored = scaler.inverse_transform(X_scaled)
print("\nInverse transform (restored):")
print(X_restored)

print("\nFit and transform in one step:")
scaler2 = StandardScaler()
X_scaled2 = scaler2.fit_transform(X)
print(X_scaled2)
