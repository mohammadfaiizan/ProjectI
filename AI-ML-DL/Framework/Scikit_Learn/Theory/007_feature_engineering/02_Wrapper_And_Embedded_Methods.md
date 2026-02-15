# Wrapper and Embedded Methods

---

## Table of Contents

- [Overview](#overview)
- [Recursive Feature Elimination](#recursive-feature-elimination)
- [RFE with Cross-Validation](#rfe-with-cross-validation)
- [SelectFromModel](#selectfrommodel)
- [Sequential Feature Selection](#sequential-feature-selection)
- [Embedded Methods in Tree Models](#embedded-methods-in-tree-models)
- [L1-Based Selection](#l1-based-selection)
- [Comparison of Methods](#comparison-of-methods)
- [Best Practices](#best-practices)

---

## Overview

**Wrapper methods** use a learning algorithm to evaluate feature subsets (e.g., RFE, sequential selection). **Embedded methods** perform feature selection as part of model training (e.g., L1 regularization, tree-based importance). They are often more accurate than filter methods but computationally heavier.

| Method | Type | Use Case |
|--------|------|----------|
| **RFE** | Wrapper | Backward elimination |
| **RFECV** | Wrapper | RFE with CV for n_features |
| **SelectFromModel** | Embedded | Post-hoc threshold on importance |
| **SequentialFeatureSelector** | Wrapper | Forward/backward stepwise |
| **L1 (Lasso)** | Embedded | Sparse linear models |

---

## Recursive Feature Elimination

**RFE** (Recursive Feature Elimination) trains a model, ranks features by importance (e.g., coef_, feature_importances_), removes the least important, and repeats until n_features_to_select remain.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **estimator** | Model with coef_ or feature_importances_ |
| **n_features_to_select** | Number of features to keep |
| **step** | Features to remove per iteration (int or float) |

### Key Attributes

| Attribute | Description |
|-----------|-------------|
| **ranking_** | Feature ranking (1 = selected) |
| **support_** | Boolean mask of selected features |
| **n_features_in_** | Number of input features |

### Usage

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression(max_iter=1000)
rfe = RFE(estimator, n_features_to_select=5, step=1)
X_selected = rfe.fit_transform(X, y)
print(rfe.ranking_)
print(rfe.support_)
```

---

## RFE with Cross-Validation

**RFECV** performs RFE and uses cross-validation to select the optimal number of features. Automatically finds n_features_to_select.

### Usage

```python
from sklearn.feature_selection import RFECV

rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=5, scoring="accuracy")
rfecv.fit(X, y)
print(rfecv.n_features_to_select_)
print(rfecv.cv_results_)
```

### Key Attributes

- **n_features_to_select_**: Optimal number of features
- **cv_results_**: Dict with mean_test_score, etc.
- **grid_scores_**: Deprecated; use cv_results_

---

## SelectFromModel

**SelectFromModel** selects features based on a threshold applied to model-derived importance (coef_, feature_importances_). Fits the model once, then filters by importance.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **estimator** | Model with coef_ or feature_importances_ |
| **threshold** | Min importance (float, "median", "mean") |
| **prefit** | If True, estimator is already fitted |

### Usage

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100),
    threshold="median"
)
X_selected = selector.fit_transform(X, y)
```

### With L1 (Lasso)

```python
from sklearn.linear_model import Lasso

selector = SelectFromModel(Lasso(alpha=0.1), prefit=False)
X_selected = selector.fit_transform(X, y)
```

---

## Sequential Feature Selection

**SequentialFeatureSelector** performs forward or backward stepwise selection. Adds or removes one feature at a time based on cross-validation score.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **estimator** | Model to evaluate |
| **n_features_to_select** | Target number of features |
| **direction** | 'forward' or 'backward' |
| **scoring** | Metric for evaluation |
| **cv** | Cross-validation strategy |

### Usage

```python
from sklearn.feature_selection import SequentialFeatureSelector

sfs = SequentialFeatureSelector(
    LogisticRegression(),
    n_features_to_select=5,
    direction="forward",
    cv=5
)
X_selected = sfs.fit_transform(X, y)
print(sfs.get_support())
```

---

## Embedded Methods in Tree Models

Tree-based models (RandomForest, GradientBoosting, etc.) provide **feature_importances_**. Use **SelectFromModel** to threshold them.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

rf = RandomForestClassifier(n_estimators=100)
selector = SelectFromModel(rf, threshold=0.05)
X_selected = selector.fit_transform(X, y)
```

---

## L1-Based Selection

**L1 (Lasso)** regularization drives some coefficients to zero. Non-zero coefficients indicate selected features. Use **SelectFromModel** with Lasso or LogisticRegression(penalty="l1").

```python
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

lasso = Lasso(alpha=0.1)
selector = SelectFromModel(lasso)
X_selected = selector.fit_transform(X, y)
```

---

## Comparison of Methods

| Method | Pros | Cons |
|--------|------|------|
| **RFE** | Model-aware, flexible | Slow for many features |
| **RFECV** | Auto n_features | Slower |
| **SelectFromModel** | Fast, one fit | Depends on model quality |
| **SequentialFeatureSelector** | Exhaustive search | Very slow |
| **L1** | Built-in, interpretable | Linear models only |

---

## Best Practices

| Practice | Reason |
|----------|--------|
| Use **RFECV** when n_features unknown | Automatic selection |
| Use **SelectFromModel** with tree models | Fast, robust |
| Use **L1** for linear models | Sparse, interpretable |
| Combine with **Pipeline** | Avoid leakage in CV |
| Scale features before L1/RFE with linear models | Coef magnitude matters |
