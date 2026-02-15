"""
Scikit-learn LabelEncoder: fit, transform, inverse_transform, classes_
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder

y = np.array(['red', 'blue', 'green', 'blue', 'red', 'green'])
print("Original labels:", y)

le = LabelEncoder()
le.fit(y)

print("\nclasses_:", le.classes_)

y_encoded = le.transform(y)
print("Encoded:", y_encoded)

y_new = np.array(['green', 'red', 'blue'])
y_new_encoded = le.transform(y_new)
print("New labels encoded:", y_new_encoded)

y_decoded = le.inverse_transform(y_encoded)
print("Inverse transform:", y_decoded)

print("\nFit and transform in one step:")
le2 = LabelEncoder()
y_enc = le2.fit_transform(y)
print(y_enc)
