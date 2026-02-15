"""
Scikit-learn PolynomialFeatures for interaction features
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)
X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
print("Original data (3 features):")
print(X)

print("\n--- degree=2, full polynomial ---")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print("Shape:", X_poly.shape)
print("Feature names:", poly.get_feature_names_out())

print("\n--- degree=2, interaction_only=True ---")
poly_int = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_int = poly_int.fit_transform(X)
print("Shape:", X_int.shape)
print("Feature names:", poly_int.get_feature_names_out())

print("\n--- degree=3, interaction_only ---")
poly3 = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)
X_int3 = poly3.fit_transform(X)
print("Shape:", X_int3.shape)
print("Feature names:", poly3.get_feature_names_out())

print("\n--- include_bias=True ---")
poly_bias = PolynomialFeatures(degree=2, include_bias=True)
X_bias = poly_bias.fit_transform(X)
print("First row:", X_bias[0])
print("Feature names:", poly_bias.get_feature_names_out())
