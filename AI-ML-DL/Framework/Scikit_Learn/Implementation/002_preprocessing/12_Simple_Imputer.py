"""
Scikit-learn SimpleImputer: strategy (mean, median, most_frequent, constant)
"""
import numpy as np
from sklearn.impute import SimpleImputer

X = np.array([[1.0, 2.0, np.nan],
              [3.0, np.nan, 6.0],
              [np.nan, 8.0, 9.0],
              [4.0, 5.0, 6.0]])
print("Data with missing values:")
print(X)

print("\n--- strategy='mean' ---")
imputer_mean = SimpleImputer(strategy='mean')
X_mean = imputer_mean.fit_transform(X)
print("Mean imputed:", X_mean)
print("Statistics_:", imputer_mean.statistics_)

print("\n--- strategy='median' ---")
imputer_median = SimpleImputer(strategy='median')
X_median = imputer_median.fit_transform(X)
print("Median imputed:", X_median)

print("\n--- strategy='most_frequent' ---")
X_cat = np.array([['a', 'x'], [np.nan, 'y'], ['a', np.nan], ['b', 'y']])
imputer_freq = SimpleImputer(strategy='most_frequent')
X_freq = imputer_freq.fit_transform(X_cat)
print("Categorical data:", X_cat)
print("Most frequent imputed:", X_freq)

print("\n--- strategy='constant' ---")
imputer_const = SimpleImputer(strategy='constant', fill_value=-1)
X_const = imputer_const.fit_transform(X)
print("Constant (-1) imputed:", X_const)
