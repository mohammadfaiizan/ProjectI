"""
Scikit-learn OneHotEncoder: sparse_output, handle_unknown, drop, categories
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder

X = np.array([['red', 'small'], ['blue', 'large'], ['green', 'medium'],
              ['red', 'large'], ['blue', 'small']])
print("Original data:")
print(X)

enc = OneHotEncoder(sparse_output=False)
enc.fit(X)

print("\nEncoded (dense):")
X_encoded = enc.transform(X)
print(X_encoded)
print("categories_:", enc.get_feature_names_out())

print("\n--- sparse_output=True ---")
enc_sparse = OneHotEncoder(sparse_output=True)
X_sparse = enc_sparse.fit_transform(X)
print("Sparse matrix shape:", X_sparse.shape)
print("Dense view:", X_sparse.toarray())

print("\n--- drop='first' (avoid multicollinearity) ---")
enc_drop = OneHotEncoder(drop='first', sparse_output=False)
X_drop = enc_drop.fit_transform(X)
print("Dropped first:", X_drop)
print("Feature names:", enc_drop.get_feature_names_out())

print("\n--- handle_unknown='ignore' ---")
enc_unk = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
enc_unk.fit(X)
X_new = np.array([['red', 'small'], ['purple', 'tiny']])
X_new_enc = enc_unk.transform(X_new)
print("New with unknown:", X_new_enc)
