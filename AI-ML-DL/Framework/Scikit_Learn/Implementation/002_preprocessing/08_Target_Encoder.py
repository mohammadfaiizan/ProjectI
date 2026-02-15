"""
Scikit-learn TargetEncoder for supervised categorical encoding
"""
import numpy as np
from sklearn.preprocessing import TargetEncoder

np.random.seed(42)
X = np.array([['A', 'X'], ['B', 'Y'], ['A', 'Y'], ['B', 'X'], ['A', 'X'],
              ['B', 'Y'], ['A', 'Y'], ['B', 'X']])
y = np.array([1, 0, 1, 0, 1, 0, 1, 0])

print("Features:", X)
print("Target:", y)

enc = TargetEncoder()
enc.fit(X, y)

X_encoded = enc.transform(X)
print("\nTarget-encoded features:")
print(X_encoded)

print("\n--- With smoothing ---")
enc_smooth = TargetEncoder(smooth='auto')
enc_smooth.fit(X, y)
X_smooth = enc_smooth.transform(X)
print("Smoothed encoding:", X_smooth)

X_new = np.array([['A', 'X'], ['C', 'Z']])
X_new_enc = enc.transform(X_new)
print("\nNew data encoded:", X_new_enc)
