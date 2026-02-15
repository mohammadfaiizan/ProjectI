# Inspection, Validation, and Patterns

---

## Table of Contents

- [Overview](#overview)
- [Estimator Inspection](#estimator-inspection)
- [Parameter Access](#parameter-access)
- [Model Validation](#model-validation)
- [Cloning Estimators](#cloning-estimators)
- [Set and Get Params](#set-and-get-params)
- [BaseEstimator and Mixins](#baseestimator-and-mixins)
- [Common Patterns](#common-patterns)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

---

## Overview

**Inspection** refers to accessing fitted attributes (e.g., **coef_**, **feature_importances_**) and understanding model state. **Validation** ensures inputs meet estimator requirements. Scikit-learn follows consistent **patterns** (fit/transform/predict, get_params/set_params) that enable composition and meta-estimators.

| Concept | Purpose |
|---------|---------|
| **Fitted attributes** | Learned parameters (suffix `_`) |
| **get_params / set_params** | Hyperparameter access for tuning |
| **clone** | Copy estimator with same parameters |
| **check_*** | Internal validation functions |

---

## Estimator Inspection

After **fit**, estimators store learned state in attributes ending with `_`. These are read-only and should not be modified.

### Common Fitted Attributes

| Estimator Type | Example Attributes |
|----------------|-------------------|
| **LinearRegression** | coef_, intercept_ |
| **StandardScaler** | mean_, scale_ |
| **PCA** | components_, explained_variance_ratio_ |
| **RandomForest** | feature_importances_, estimators_ |
| **KMeans** | cluster_centers_, labels_ |

### Usage

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X_train, y_train)
print(model.coef_)
print(model.intercept_)
```

---

## Parameter Access

**get_params()** returns a dict of constructor parameters. **set_params(**kwargs)** updates parameters and returns self (for chaining).

```python
params = model.get_params()
model.set_params(C=0.5, max_iter=1000)
```

### Nested Parameters (Pipeline)

For pipelines, use **step__param** syntax:

```python
pipe.get_params()  # Includes "clf__C", "scaler__with_mean", etc.
pipe.set_params(clf__C=0.1)
```

---

## Model Validation

Scikit-learn performs internal checks via **check_X_y**, **check_array**, and similar. Inputs must be:

- **X**: 2D array-like, finite, no NaN (unless estimator allows)
- **y**: 1D for supervised, aligned with X

### Common Issues

- **NaN** in X or y: Many estimators raise; use **SimpleImputer** for X
- **Wrong shape**: Ensure X is (n_samples, n_features)
- **Wrong dtypes**: Most expect float; encode categoricals first

---

## Cloning Estimators

**clone** creates a deep copy of an estimator with the same parameters but no fitted state. Used internally by **GridSearchCV** and **cross_val_score**.

```python
from sklearn.base import clone

model_fitted = LogisticRegression().fit(X_train, y_train)
model_copy = clone(model_fitted)
# model_copy is unfitted, same hyperparameters
```

---

## Set and Get Params

**get_params(deep=True)** returns all parameters. **deep=True** includes parameters of nested estimators.

```python
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression()),
])
params = pipe.get_params(deep=True)
# params["clf__C"], params["clf__max_iter"], etc.
```

**set_params** supports partial updates:

```python
pipe.set_params(clf__C=0.5)
```

---

## BaseEstimator and Mixins

Custom estimators should inherit from **BaseEstimator** and appropriate mixins:

- **BaseEstimator**: get_params, set_params
- **ClassifierMixin**: score uses accuracy
- **RegressorMixin**: score uses R-squared
- **TransformerMixin**: fit_transform
- **ClusterMixin**: fit_predict

```python
from sklearn.base import BaseEstimator, ClassifierMixin

class MyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, param=1):
        self.param = param

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))
```

---

## Common Patterns

### Fit-Transform-Predict

```python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Fit-Transform (Transformers)

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Pipeline Pattern

```python
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
# Preprocessing applied automatically
```

### Cross-Validation Pattern

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X, y, cv=5)
```

---

## Error Handling

Common exceptions:

- **ValueError**: Invalid input (wrong shape, NaN, etc.)
- **NotFittedError**: predict/transform called before fit
- **ConvergenceWarning**: Optimizer did not converge

### Checking Fitted State

```python
from sklearn.utils.validation import check_is_fitted

check_is_fitted(model)  # Raises NotFittedError if not fitted
```

---

## Best Practices

| Practice | Reason |
|----------|--------|
| Inspect **fitted attributes** for debugging | Understand model state |
| Use **clone** before modifying | Avoid mutating shared objects |
| Use **get_params** for logging | Reproducibility |
| Inherit **BaseEstimator** for custom estimators | API compatibility |
| Validate inputs before fit | Clear error messages |
