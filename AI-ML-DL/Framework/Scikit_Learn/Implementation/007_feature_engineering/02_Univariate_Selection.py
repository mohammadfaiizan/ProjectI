"""
Scikit-learn SelectKBest, SelectPercentile, f_classif, f_regression
"""
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, f_regression

np.random.seed(42)
X_clf, y_clf = make_classification(n_samples=100, n_features=20, n_informative=5, random_state=42)
X_reg, y_reg = make_regression(n_samples=100, n_features=20, n_informative=5, random_state=42)

print("--- SelectKBest with f_classif (classification) ---")
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_clf, y_clf)
print("Original shape:", X_clf.shape)
print("Selected shape:", X_selected.shape)
print("Scores:", selector.scores_[:5], "...")
print("Selected indices:", selector.get_support(indices=True))

print("\n--- SelectKBest with f_regression (regression) ---")
selector_reg = SelectKBest(score_func=f_regression, k=5)
X_reg_selected = selector_reg.fit_transform(X_reg, y_reg)
print("Original shape:", X_reg.shape)
print("Selected shape:", X_reg_selected.shape)
print("Selected indices:", selector_reg.get_support(indices=True))

print("\n--- SelectPercentile (top 25%) ---")
selector_pct = SelectPercentile(score_func=f_classif, percentile=25)
X_pct = selector_pct.fit_transform(X_clf, y_clf)
print("Selected shape:", X_pct.shape)
print("Number of features selected:", X_pct.shape[1])

print("\n--- Scores and p-values ---")
scores, pvalues = f_classif(X_clf, y_clf)
print("F-scores (first 5):", scores[:5])
print("P-values (first 5):", pvalues[:5])
