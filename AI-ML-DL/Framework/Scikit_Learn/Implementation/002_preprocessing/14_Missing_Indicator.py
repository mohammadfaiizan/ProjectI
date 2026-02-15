"""
Scikit-learn MissingIndicator (features, sparse_output) and add_indicator
"""
import numpy as np
from sklearn.impute import MissingIndicator, SimpleImputer

np.random.seed(42)
X = np.array([[1.0, 2.0, np.nan],
              [3.0, np.nan, 6.0],
              [np.nan, 8.0, 9.0],
              [4.0, 5.0, 6.0]])
print("Data with missing values:")
print(X)

print("\n--- MissingIndicator ---")
indicator = MissingIndicator(features='missing-only', sparse_output=False)
X_ind = indicator.fit_transform(X)
print("Missing indicators:", X_ind)
print("features_:", indicator.features_)

print("\n--- MissingIndicator (all features) ---")
ind_all = MissingIndicator(features='all', sparse_output=False)
X_ind_all = ind_all.fit_transform(X)
print("All-feature indicators:", X_ind_all)

print("\n--- SimpleImputer with add_indicator=True ---")
imputer = SimpleImputer(strategy='mean', add_indicator=True)
X_imp = imputer.fit_transform(X)
print("Imputed + indicators:", X_imp)
print("Indicator features appended to imputed data")
