# Imputation and Preprocessing Pipelines

---

## Table of Contents

- [Overview](#overview)
- [SimpleImputer](#simpleimputer)
- [KNNImputer](#knnimputer)
- [IterativeImputer](#iterativeimputer)
- [MissingIndicator](#missingindicator)
- [Preprocessing Pipelines](#preprocessing-pipelines)
- [ColumnTransformer for Mixed Data](#columntransformer-for-mixed-data)
- [Order of Operations](#order-of-operations)
- [Best Practices](#best-practices)

---

## Overview

**Imputation** fills missing values (NaN) with estimated values. Different strategies suit different data types and missingness patterns. **Preprocessing pipelines** chain imputation, scaling, and encoding to ensure consistent transformation and avoid data leakage in cross-validation.

| Imputer | Strategy | Use Case |
|---------|----------|----------|
| **SimpleImputer** | mean, median, most_frequent, constant | Simple univariate imputation |
| **KNNImputer** | k-nearest neighbors | Multivariate, preserves structure |
| **IterativeImputer** | Model-based (e.g., Bayesian ridge) | Complex dependencies |
| **MissingIndicator** | Add binary missingness flags | Inform model about missingness |

---

## SimpleImputer

**SimpleImputer** replaces missing values with a statistic (mean, median, most_frequent) or a constant. Computed per feature during **fit**.

### Parameters

| Parameter | Description |
|-----------|-------------|
| **strategy** | 'mean', 'median', 'most_frequent', 'constant' |
| **fill_value** | Used when strategy='constant' |
| **add_indicator** | Add MissingIndicator as extra features |
| **copy** | Copy input or modify in place |

### Key Attribute

- **statistics_**: Per-feature value used for imputation (mean, median, etc.)

### Usage

```python
from sklearn.impute import SimpleImputer
import numpy as np

X = np.array([[1, 2, np.nan], [3, np.nan, 6], [np.nan, 8, 9]])

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
print(imputer.statistics_)
```

### Strategy Selection

| Strategy | Data Type | Notes |
|----------|-----------|-------|
| **mean** | Numeric | Sensitive to outliers |
| **median** | Numeric | Robust to outliers |
| **most_frequent** | Numeric or categorical | Mode |
| **constant** | Any | Use fill_value |

---

## KNNImputer

**KNNImputer** imputes using the mean of k nearest neighbors. Preserves multivariate structure better than univariate imputation.

### Parameters

| Parameter | Description |
|-----------|-------------|
| **n_neighbors** | Number of neighbors (default 5) |
| **weights** | 'uniform', 'distance' |
| **metric** | Distance metric for neighbors |

### Usage

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
```

### When to Use

- Missing values have structure (e.g., correlated features)
- Dataset is not too large (computationally expensive)

---

## IterativeImputer

**IterativeImputer** models each feature with missing values as a function of other features. Iteratively refines imputations. Based on MICE (Multivariate Imputation by Chained Equations).

### Parameters

| Parameter | Description |
|-----------|-------------|
| **estimator** | Estimator for modeling (default BayesianRidge) |
| **max_iter** | Number of imputation rounds |
| **tol** | Convergence tolerance |
| **initial_strategy** | 'mean', 'median', 'most_frequent', 'constant' |

### Usage

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=42)
X_imputed = imputer.fit_transform(X)
```

### Important Notes

- Requires **enable_iterative_imputer** (experimental API)
- Slower than SimpleImputer; use for complex dependencies

---

## MissingIndicator

**MissingIndicator** adds binary indicators (0/1) for missing values. Helps models learn that missingness can be informative (e.g., "not reported" vs "reported as zero").

### Parameters

| Parameter | Description |
|-----------|-------------|
| **features** | 'missing-only', 'all' |
| **error_on_new** | Raise if new columns have missing |

### Usage

```python
from sklearn.impute import MissingIndicator

indicator = MissingIndicator(features="missing-only")
X_missing = indicator.fit_transform(X)
# Combine with imputed data
X_full = np.hstack([imputer.fit_transform(X), X_missing])
```

### With SimpleImputer

```python
imputer = SimpleImputer(strategy="mean", add_indicator=True)
X_imputed = imputer.fit_transform(X)
# Extra columns indicate which values were missing
```

---

## Preprocessing Pipelines

**Pipeline** chains transformers so that **fit** is called only on training data and **transform** is applied consistently. Critical for avoiding data leakage.

### Basic Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
X_train_transformed = pipe.fit_transform(X_train)
X_test_transformed = pipe.transform(X_test)
```

### Order of Steps

1. **Imputation** first (handles NaN before scaling/encoding)
2. **Encoding** for categoricals
3. **Scaling** for numeric features
4. **Feature selection** (optional)
5. **Model** as final step

---

## ColumnTransformer for Mixed Data

**ColumnTransformer** applies different pipelines to different column subsets. Essential for mixed numeric and categorical data.

```python
from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, [0, 1, 2]),
    ("cat", cat_pipeline, [3, 4]),
])
```

### With make_column_selector

```python
from sklearn.compose import make_column_selector

preprocessor = ColumnTransformer([
    ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
])
```

---

## Order of Operations

| Step | Purpose |
|------|---------|
| 1. Imputation | Remove NaN so downstream steps work |
| 2. Encoding | Convert categoricals to numeric |
| 3. Scaling | Normalize numeric features |
| 4. Optional: Feature selection | Reduce dimensionality |
| 5. Model | Final estimator |

### Full Example

```python
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression()),
])
full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)
```

---

## Best Practices

| Practice | Reason |
|----------|--------|
| Fit imputer on **training data only** | Avoid leakage |
| Use **Pipeline** with CV | Correct fold-wise fitting |
| Consider **add_indicator** for informative missingness | Improve model |
| Use **median** for skewed numeric data | Robust to outliers |
| Use **most_frequent** for categorical | Simple and effective |
| Test **transform** on data with different missing patterns | Robustness |
