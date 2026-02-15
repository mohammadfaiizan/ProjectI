"""
Scikit-learn mutual_info_classif, mutual_info_regression, discrete_features
"""
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

np.random.seed(42)
X_clf, y_clf = make_classification(n_samples=200, n_features=10, n_informative=4, random_state=42)
X_reg, y_reg = make_regression(n_samples=200, n_features=10, n_informative=4, random_state=42)

print("--- mutual_info_classif ---")
mi_clf = mutual_info_classif(X_clf, y_clf, random_state=42)
print("Mutual information scores:", np.round(mi_clf, 4))
print("Top 3 features:", np.argsort(mi_clf)[-3:][::-1])

print("\n--- mutual_info_regression ---")
mi_reg = mutual_info_regression(X_reg, y_reg, random_state=42)
print("Mutual information scores:", np.round(mi_reg, 4))
print("Top 3 features:", np.argsort(mi_reg)[-3:][::-1])

print("\n--- discrete_features (mixed data) ---")
X_mixed = np.column_stack([
    np.random.randint(0, 5, 200),
    np.random.randn(200),
    np.random.randint(0, 3, 200),
])
discrete_mask = [True, False, True]
mi_mixed = mutual_info_classif(X_mixed, y_clf, discrete_features=discrete_mask, random_state=42)
print("Discrete features mask:", discrete_mask)
print("MI scores (mixed):", np.round(mi_mixed, 4))

print("\n--- n_neighbors parameter ---")
mi_k5 = mutual_info_classif(X_clf, y_clf, n_neighbors=5, random_state=42)
mi_k10 = mutual_info_classif(X_clf, y_clf, n_neighbors=10, random_state=42)
print("n_neighbors=5 (first 5):", np.round(mi_k5[:5], 4))
print("n_neighbors=10 (first 5):", np.round(mi_k10[:5], 4))
