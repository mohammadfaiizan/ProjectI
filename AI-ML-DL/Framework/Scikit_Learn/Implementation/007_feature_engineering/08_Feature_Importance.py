"""
Scikit-learn Tree-based feature_importances_, linear model coef_
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=10, n_informative=4, random_state=42)

print("--- RandomForest feature_importances_ ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
print("Importances:", np.round(rf.feature_importances_, 4))
print("Top 3 features:", np.argsort(rf.feature_importances_)[-3:][::-1])

print("\n--- DecisionTree feature_importances_ ---")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X, y)
print("Importances:", np.round(dt.feature_importances_, 4))

print("\n--- LogisticRegression coef_ (absolute for importance) ---")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X, y)
coef_abs = np.abs(lr.coef_[0])
print("Coefficients:", np.round(lr.coef_[0], 4))
print("Absolute coef (importance proxy):", np.round(coef_abs, 4))
print("Top 3 by |coef|:", np.argsort(coef_abs)[-3:][::-1])

print("\n--- Feature importance comparison ---")
print("RF top:", np.argsort(rf.feature_importances_)[-3:][::-1])
print("DT top:", np.argsort(dt.feature_importances_)[-3:][::-1])
print("LR top:", np.argsort(coef_abs)[-3:][::-1])
