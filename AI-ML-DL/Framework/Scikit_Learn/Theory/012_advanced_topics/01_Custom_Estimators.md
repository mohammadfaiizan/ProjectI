# Custom Estimators, Metrics, and CV Splitters

---

## Table of Contents

- [Overview](#overview)
- [Custom Estimator Basics](#custom-estimator-basics)
- [BaseEstimator and Mixins](#baseestimator-and-mixins)
- [check_is_fitted](#check_is_fitted)
- [Advanced Custom Estimators](#advanced-custom-estimators)
- [Input Validation](#input-validation)
- [get_params and set_params](#get_params-and-set_params)
- [Custom Metrics](#custom-metrics)
- [make_scorer](#make_scorer)
- [Custom Cross-Validation Splitters](#custom-cross-validation-splitters)
- [BaseCrossValidator](#basecrossvalidator)
- [When to Use Custom Components](#when-to-use-custom-components)

---

## Overview

Scikit-learn's **estimator API** is designed for extensibility. You can create **custom estimators**, **custom metrics**, and **custom cross-validation splitters** that integrate seamlessly with pipelines, GridSearchCV, and other tools. This document covers the implementation patterns for each.

---

## Custom Estimator Basics

### Why Custom Estimators

- Implement domain-specific algorithms not in sklearn
- Wrap external libraries with the sklearn interface
- Create composite or meta-algorithms
- Ensure compatibility with **Pipeline**, **GridSearchCV**, and **cross_val_score**

### Core Requirements

| Requirement | Purpose |
|-------------|---------|
| **fit(X, y)** | Learn from data, store fitted attributes |
| **predict(X)** or **transform(X)** | Apply learned model |
| **get_params(deep=True)** | Return constructor parameters for clone/set_params |
| **set_params(\\*\\*params)** | Set parameters, return self |

---

## BaseEstimator and Mixins

**BaseEstimator** provides default `get_params` and `set_params` based on `__init__` parameters. **ClassifierMixin** adds `score(X, y)` using accuracy. **RegressorMixin** adds `score(X, y)` using R-squared.

```python
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class SimpleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.mean_ = np.mean(X[y == self.classes_[0]], axis=0)
        return self

    def predict(self, X):
        dist = np.linalg.norm(X - self.mean_, axis=1)
        return np.where(dist < np.median(dist), self.classes_[0], self.classes_[1])
```

### Mixin Summary

| Mixin | Adds |
|-------|------|
| **ClassifierMixin** | score() using accuracy |
| **RegressorMixin** | score() using R2 |
| **TransformerMixin** | fit_transform() = fit().transform() |
| **ClusterMixin** | fit_predict() for clustering |

---

## check_is_fitted

**check_is_fitted** raises `NotFittedError` if the estimator has not been fitted. Use it at the start of `predict` or `transform` to avoid cryptic errors.

```python
from sklearn.utils.validation import check_is_fitted

def predict(self, X):
    check_is_fitted(self, ["mean_"])
    return ...
```

Fitted attributes should end with a trailing underscore (e.g., `coef_`, `mean_`) by convention.

---

## Advanced Custom Estimators

For full compatibility with **GridSearchCV** and **clone**, implement:

1. **get_params(deep=True)** – return all constructor parameters
2. **set_params(**params)** – set parameters and return self
3. **Input validation** – use `check_X_y` and `check_array` for robust behavior

---

## Input Validation

**check_X_y(X, y)** validates X and y for supervised learning. **check_array(X)** validates X for transform/predict. Both handle sparse matrices and optional constraints.

```python
from sklearn.utils.validation import check_X_y, check_array

def fit(self, X, y):
    X, y = check_X_y(X, y, accept_sparse=False)
    ...

def predict(self, X):
    X = check_array(X, accept_sparse=False)
    ...
```

---

## get_params and set_params

**get_params** must return a dict of all init parameters. **set_params** updates attributes and returns self for method chaining. BaseEstimator provides defaults if all init args are stored as attributes.

```python
def get_params(self, deep=True):
    return {"alpha": self.alpha, "fit_intercept": self.fit_intercept}

def set_params(self, **params):
    for key, value in params.items():
        setattr(self, key, value)
    return self
```

---

## Custom Metrics

Scikit-learn uses **scoring** callables for cross-validation and model selection. A scorer is a function `(estimator, X, y) -> float`. Higher is better for most metrics; use `greater_is_better=False` for losses like MSE.

---

## make_scorer

**make_scorer** wraps a metric function into a scorer compatible with `cross_val_score` and `GridSearchCV`.

```python
from sklearn.metrics import make_scorer, f1_score

f1_scorer = make_scorer(f1_score, average="macro")
scores = cross_val_score(clf, X, y, cv=5, scoring=f1_scorer)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **greater_is_better** | True if higher score is better (default: True) |
| **needs_threshold** | True for metrics needing predict_proba (e.g., roc_auc) |
| **needs_proba** | True for metrics using probability estimates |
| **kwargs** | Passed to the underlying metric function |

```python
auc_scorer = make_scorer(roc_auc_score, needs_threshold=True, multi_class="ovr")
```

---

## Custom Cross-Validation Splitters

Standard splitters (KFold, StratifiedKFold, etc.) may not suit all use cases. Examples:

- **Time series**: avoid future leakage; use expanding or sliding window
- **Grouped data**: ensure same group does not appear in both train and test
- **Spatial data**: geographic blocks

---

## BaseCrossValidator

Implement **BaseCrossValidator** to create a custom CV strategy. Required methods:

| Method | Purpose |
|--------|---------|
| **split(X, y, groups)** | Generator yielding (train_idx, test_idx) |
| **get_n_splits(X, y, groups)** | Return number of splits |

```python
from sklearn.model_selection import BaseCrossValidator

class BlockCV(BaseCrossValidator):
    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        indices = np.arange(n)
        for i in range(self.n_splits):
            test_start = i * n // self.n_splits
            test_stop = (i + 1) * n // self.n_splits
            test_idx = indices[test_start:test_stop]
            train_idx = np.concatenate([indices[:test_start], indices[test_stop:]])
            yield train_idx, test_idx
```

---

## When to Use Custom Components

| Component | Use When |
|-----------|----------|
| **Custom estimator** | Algorithm not in sklearn, domain-specific logic |
| **Custom metric** | Business-specific evaluation (e.g., cost-weighted) |
| **Custom CV** | Time series, grouped data, spatial blocks |

Custom components enable full integration with sklearn's ecosystem while preserving your domain requirements.
