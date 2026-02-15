"""
Scikit-learn RFE (estimator, n_features_to_select), RFECV
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=15, n_informative=5, random_state=42)

print("--- RFE with LogisticRegression ---")
estimator = LogisticRegression(max_iter=1000, random_state=42)
rfe = RFE(estimator=estimator, n_features_to_select=5, step=1)
rfe.fit(X, y)
print("Selected features:", rfe.get_support(indices=True))
print("Feature rankings (1=selected):", rfe.ranking_)
print("Selected mask:", rfe.get_support())
X_rfe = rfe.transform(X)
print("Transformed shape:", X_rfe.shape)

print("\n--- RFE with step=2 ---")
rfe_step = RFE(estimator=estimator, n_features_to_select=5, step=2)
rfe_step.fit(X, y)
print("Selected features:", rfe_step.get_support(indices=True))

print("\n--- RFECV (cross-validated selection) ---")
rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv.fit(X, y)
print("Optimal number of features:", rfecv.n_features_)
print("Selected features:", rfecv.get_support(indices=True))
print("Grid scores (by n_features):", np.round(rfecv.cv_results_['mean_test_score'], 3))
