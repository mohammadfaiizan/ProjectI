"""
Scikit-learn PowerTransformer (yeo-johnson, box-cox) and QuantileTransformer
"""
import numpy as np
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

np.random.seed(42)
X = np.random.exponential(scale=2.0, size=(100, 2))

print("Original data (exponential distribution):")
print("Sample:", X[:5])
print("Skewness:", np.mean((X - X.mean())**3) / (X.std()**3))

print("\n--- PowerTransformer (Yeo-Johnson) ---")
pt_yj = PowerTransformer(method='yeo-johnson')
X_yj = pt_yj.fit_transform(X)
print("Transformed sample:", X_yj[:5])
print(" lambdas_:", pt_yj.lambdas_)

print("\n--- PowerTransformer (Box-Cox, positive data only) ---")
X_pos = X + 1
pt_bc = PowerTransformer(method='box-cox')
X_bc = pt_bc.fit_transform(X_pos)
print("Box-Cox transformed sample:", X_bc[:5])

print("\n--- QuantileTransformer (uniform output) ---")
qt = QuantileTransformer(output_distribution='uniform')
X_qt = qt.fit_transform(X)
print("Quantile (uniform) sample:", X_qt[:5])

print("\n--- QuantileTransformer (normal output) ---")
qt_norm = QuantileTransformer(output_distribution='normal')
X_qt_norm = qt_norm.fit_transform(X)
print("Quantile (normal) sample:", X_qt_norm[:5])
