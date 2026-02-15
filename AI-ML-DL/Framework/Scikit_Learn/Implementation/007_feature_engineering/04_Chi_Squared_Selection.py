"""
Scikit-learn chi2, SelectKBest with chi2 for categorical features
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
X_nonneg = np.abs(X)
X_scaled = MinMaxScaler().fit_transform(X_nonneg)
X_binned = (X_scaled * 4).astype(int)

print("--- chi2 requires non-negative features ---")
print("Sample binned data (first 3 rows):", X_binned[:3])

print("\n--- chi2 scores and p-values ---")
scores, pvalues = chi2(X_binned, y)
print("Chi2 scores:", np.round(scores, 2))
print("P-values:", np.round(pvalues, 6))

print("\n--- SelectKBest with chi2 ---")
selector = SelectKBest(score_func=chi2, k=5)
X_selected = selector.fit_transform(X_binned, y)
print("Original shape:", X_binned.shape)
print("Selected shape:", X_selected.shape)
print("Selected feature indices:", selector.get_support(indices=True))
print("Selected scores:", np.round(selector.scores_[selector.get_support()], 2))

print("\n--- Feature names with SelectKBest ---")
selector_names = SelectKBest(score_func=chi2, k=3)
X_sel = selector_names.fit_transform(X_binned, y)
feature_names = [f"feat_{i}" for i in range(X_binned.shape[1])]
selected_names = selector_names.get_feature_names_out(feature_names)
print("Selected feature names:", selected_names)
