"""
Scikit-learn MinMaxScaler, RobustScaler, MaxAbsScaler
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler

np.random.seed(42)
X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])

print("Original data:")
print(X)

print("\n--- MinMaxScaler (scale to [0, 1]) ---")
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X)
print("Transformed:", X_minmax)
print("data_min_:", minmax.data_min_)
print("data_max_:", minmax.data_max_)

print("\n--- RobustScaler (median and IQR, robust to outliers) ---")
X_outlier = np.vstack([X, [[100.0, 1000.0]]])
robust = RobustScaler()
X_robust = robust.fit_transform(X_outlier)
print("Data with outlier:", X_outlier)
print("Robust scaled:", X_robust)

print("\n--- MaxAbsScaler (scale by max absolute value) ---")
maxabs = MaxAbsScaler()
X_maxabs = maxabs.fit_transform(X)
print("Transformed:", X_maxabs)
print("max_abs_:", maxabs.max_abs_)
