"""
Scikit-learn SelectFromModel (estimator, threshold, max_features)
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=15, n_informative=5, random_state=42)

print("--- SelectFromModel with Lasso (linear coef_) ---")
lasso = LassoCV(cv=5, random_state=42)
sfm_lasso = SelectFromModel(lasso, threshold='median')
sfm_lasso.fit(X, y)
print("Selected features:", sfm_lasso.get_support(indices=True))
print("Threshold used:", sfm_lasso.threshold_)
X_lasso = sfm_lasso.transform(X)
print("Transformed shape:", X_lasso.shape)

print("\n--- SelectFromModel with RandomForest (feature_importances_) ---")
rf = RandomForestClassifier(n_estimators=50, random_state=42)
sfm_rf = SelectFromModel(rf, threshold=0.1)
sfm_rf.fit(X, y)
print("Selected features:", sfm_rf.get_support(indices=True))
print("Importances:", np.round(sfm_rf.estimator_.feature_importances_, 3))

print("\n--- max_features parameter ---")
sfm_max = SelectFromModel(rf, max_features=5)
sfm_max.fit(X, y)
print("Selected (max_features=5):", sfm_max.get_support(indices=True))

print("\n--- Custom threshold (absolute) ---")
sfm_abs = SelectFromModel(rf, threshold=0.05)
sfm_abs.fit(X, y)
print("Selected (threshold=0.05):", sfm_abs.get_support(indices=True))
