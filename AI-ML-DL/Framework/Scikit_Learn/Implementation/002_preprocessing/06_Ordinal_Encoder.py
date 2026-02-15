"""
Scikit-learn OrdinalEncoder: categories, handle_unknown
"""
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

X = np.array([['small', 'low'], ['medium', 'high'], ['large', 'medium'],
              ['small', 'high'], ['medium', 'low']])
print("Original data:")
print(X)

enc = OrdinalEncoder()
enc.fit(X)

print("\ncategories_:", enc.categories_)

X_encoded = enc.transform(X)
print("Encoded:", X_encoded)

print("\n--- With specified categories ---")
enc_cat = OrdinalEncoder(categories=[['small', 'medium', 'large'],
                                     ['low', 'medium', 'high']])
X_enc_cat = enc_cat.fit_transform(X)
print("Encoded with order:", X_enc_cat)

print("\n--- handle_unknown='use_encoded_value' ---")
enc_unk = OrdinalEncoder(handle_unknown='use_encoded_value',
                         unknown_value=-1)
enc_unk.fit(X)
X_new = np.array([['small', 'low'], ['extra', 'unknown']])
X_new_enc = enc_unk.transform(X_new)
print("New data with unknown:", X_new)
print("Encoded:", X_new_enc)
