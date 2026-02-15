"""
Scikit-learn PolynomialFeatures: degree, interaction_only, include_bias
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print("Original data (2 features):")
print(X)

print("\n--- degree=2, include_bias=True ---")
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)
print("Polynomial features:", X_poly)
print("Feature names:", poly.get_feature_names_out())

print("\n--- degree=2, interaction_only=True ---")
poly_int = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_int = poly_int.fit_transform(X)
print("Interaction only:", X_int)
print("Feature names:", poly_int.get_feature_names_out())

print("\n--- degree=3 ---")
poly3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly3 = poly3.fit_transform(X)
print("Degree 3 shape:", X_poly3.shape)
print("Feature names:", poly3.get_feature_names_out())
