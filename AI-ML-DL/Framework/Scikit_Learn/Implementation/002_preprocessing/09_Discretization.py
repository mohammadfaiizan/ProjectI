"""
Scikit-learn KBinsDiscretizer (n_bins, encode, strategy) and Binarizer
"""
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, Binarizer

np.random.seed(42)
X = np.array([[1.5, 2.3], [3.1, 4.2], [5.0, 1.8], [2.2, 3.9], [4.5, 5.1]])

print("Original data:")
print(X)

print("\n--- KBinsDiscretizer (uniform strategy) ---")
kbd = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
X_binned = kbd.fit_transform(X)
print("Binned (ordinal):", X_binned)
print("bin_edges_:", kbd.bin_edges_)

print("\n--- KBinsDiscretizer (quantile strategy) ---")
kbd_q = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
X_binned_q = kbd_q.fit_transform(X)
print("Binned (quantile):", X_binned_q)

print("\n--- KBinsDiscretizer (onehot encode) ---")
kbd_oh = KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='uniform')
X_oh = kbd_oh.fit_transform(X)
print("Onehot-dense:", X_oh)

print("\n--- Binarizer ---")
binarizer = Binarizer(threshold=3.0)
X_bin = binarizer.fit_transform(X)
print("Binarized (threshold=3):", X_bin)
