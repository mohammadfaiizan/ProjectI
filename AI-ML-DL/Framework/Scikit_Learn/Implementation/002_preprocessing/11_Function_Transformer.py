"""
Scikit-learn FunctionTransformer: func, inverse_func, validate
"""
import numpy as np
from sklearn.preprocessing import FunctionTransformer

np.random.seed(42)
X = np.array([[1.0, 4.0], [2.0, 9.0], [3.0, 16.0], [4.0, 25.0]])
print("Original data:")
print(X)

print("\n--- Log transform ---")
log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
X_log = log_transformer.fit_transform(X)
print("Log1p transformed:", X_log)
X_inv = log_transformer.inverse_transform(X_log)
print("Inverse (expm1):", X_inv)

print("\n--- Square transform ---")
square_transformer = FunctionTransformer(func=np.square)
X_sq = square_transformer.fit_transform(X)
print("Squared:", X_sq)

print("\n--- Custom function ---")
def add_constant(X, c=1.0):
    return X + c

def subtract_constant(X, c=1.0):
    return X - c

custom_transformer = FunctionTransformer(
    func=add_constant,
    inverse_func=subtract_constant,
    kw_args={'c': 5.0},
    inv_kw_args={'c': 5.0}
)
X_custom = custom_transformer.fit_transform(X)
print("Add 5:", X_custom)
print("Inverse:", custom_transformer.inverse_transform(X_custom))

print("\n--- validate=False (skip input validation) ---")
ft = FunctionTransformer(func=np.log1p, validate=False)
X_out = ft.fit_transform(X)
print("No validation:", X_out[:2])
