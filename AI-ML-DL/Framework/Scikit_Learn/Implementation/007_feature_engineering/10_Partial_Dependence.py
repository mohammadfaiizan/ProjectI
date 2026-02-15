"""
Scikit-learn partial_dependence, PartialDependenceDisplay
"""
import os
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
X, y = make_regression(n_samples=200, n_features=5, n_informative=3, random_state=42)
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)

print("--- partial_dependence (1D) ---")
pd_result = partial_dependence(model, X, features=[0])
print("Feature 0 - grid shape:", pd_result['grid_values'][0].shape)
print("Feature 0 - average partial dependence:", np.round(pd_result['average'][0][:5], 3), "...")

print("\n--- partial_dependence (2D interaction) ---")
pd_2d = partial_dependence(model, X, features=[0, 1])
print("2D grid - feature 0:", pd_2d['grid_values'][0].shape)
print("2D grid - feature 1:", pd_2d['grid_values'][1].shape)
print("2D average shape:", pd_2d['average'][0].shape)

print("\n--- PartialDependenceDisplay (saves plot) ---")
fig, ax = plt.subplots(figsize=(6, 4))
PartialDependenceDisplay.from_estimator(model, X, features=[0, 1], ax=ax)
plt.savefig(os.path.join(os.path.dirname(__file__), 'pd_plot.png'), dpi=80)
plt.close()
print("Plot saved to pd_plot.png")
