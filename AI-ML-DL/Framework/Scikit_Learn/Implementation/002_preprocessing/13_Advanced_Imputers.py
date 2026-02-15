"""
Scikit-learn KNNImputer (n_neighbors) and IterativeImputer (estimator)
"""
import numpy as np
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)
X = np.array([[1.0, 2.0, 3.0],
              [4.0, np.nan, 6.0],
              [7.0, 8.0, np.nan],
              [np.nan, 11.0, 12.0],
              [13.0, 14.0, 15.0]])
print("Data with missing values:")
print(X)

print("\n--- KNNImputer (n_neighbors=2) ---")
knn_imputer = KNNImputer(n_neighbors=2)
X_knn = knn_imputer.fit_transform(X)
print("KNN imputed:", X_knn)

print("\n--- KNNImputer (n_neighbors=3, weights='distance') ---")
knn_dist = KNNImputer(n_neighbors=3, weights='distance')
X_knn_dist = knn_dist.fit_transform(X)
print("Distance-weighted KNN:", X_knn_dist)

print("\n--- IterativeImputer (default BayesianRidge) ---")
iter_imputer = IterativeImputer(max_iter=10, random_state=42)
X_iter = iter_imputer.fit_transform(X)
print("Iterative imputed:", X_iter)

print("\n--- IterativeImputer (RandomForest estimator) ---")
iter_rf = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10),
                           max_iter=5, random_state=42)
X_iter_rf = iter_rf.fit_transform(X)
print("RF-based iterative:", X_iter_rf)
