# Multi-Output and Chain Methods

---

## Table of Contents

- [Overview](#overview)
- [Multi-Output Problems](#multi-output-problems)
- [MultiOutputClassifier](#multioutputclassifier)
- [MultiOutputRegressor](#multioutputregressor)
- [RegressorChain](#regressorchain)
- [ClassifierChain](#classifierchain)
- [Order Parameter](#order-parameter)
- [When to Use Each Approach](#when-to-use-each-approach)

---

## Overview

**Multi-output** problems have multiple target variables per sample. Scikit-learn provides **MultiOutputClassifier**, **MultiOutputRegressor**, **RegressorChain**, and **ClassifierChain** to handle them. Chains exploit dependencies between targets; multi-output treats them independently.

---

## Multi-Output Problems

### Classification

**Multi-label classification**: each sample can have multiple labels (e.g., tags for an article). The target is a binary matrix of shape (n_samples, n_labels).

### Regression

**Multi-target regression**: each sample has multiple continuous targets (e.g., predicting temperature and humidity). The target is a matrix of shape (n_samples, n_targets).

---

## MultiOutputClassifier

**MultiOutputClassifier** wraps any classifier and trains one estimator per target. Each target is treated independently. Predictions are stacked into shape (n_samples, n_outputs).

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

clf = MultiOutputClassifier(LogisticRegression(), n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **estimator** | Base classifier (must support multi-output or be wrapped per target) |
| **n_jobs** | Number of parallel jobs for fitting (default: None) |

### Fitted Attributes

- **estimators_**: list of fitted classifiers, one per output

---

## MultiOutputRegressor

**MultiOutputRegressor** wraps any regressor and trains one estimator per target. Targets are independent.

```python
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge

reg = MultiOutputRegressor(Ridge(alpha=1.0))
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

### When to Use

Use when targets are largely independent. Simple, parallelizable, and works with any base regressor.

---

## RegressorChain

**RegressorChain** trains regressors in sequence. Each regressor uses the original features plus the predictions of previous targets. This models **target dependencies**.

```python
from sklearn.multioutput import RegressorChain
from sklearn.linear_model import Ridge

chain = RegressorChain(Ridge(), order=[0, 1, 2])
chain.fit(X_train, y_train)
y_pred = chain.predict(X_test)
```

### Order Parameter

| Value | Behavior |
|-------|----------|
| **list** | Explicit order (e.g., [2, 0, 1]) |
| **None** | Random order (different each fit) |

Order affects performance when targets are correlated. Earlier targets in the chain influence later ones.

---

## ClassifierChain

**ClassifierChain** extends the chain idea to multi-label classification. Each classifier receives original features plus binary predictions from previous labels. Useful when labels are dependent (e.g., hierarchical tags).

```python
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression

chain = ClassifierChain(LogisticRegression(), order=[0, 1, 2, 3])
chain.fit(X_train, y_train)
y_pred = chain.predict(X_test)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **estimator** | Base classifier |
| **order** | Label order (list or None for random) |
| **cv** | If int, fit multiple chains with different random orders and average (ensemble) |

### cv for Ensemble

Setting **cv=5** fits 5 chains with different random orders and averages predictions. Can improve robustness when optimal order is unknown.

---

## Order Parameter

For **ClassifierChain** and **RegressorChain**:

- **order=[0, 1, 2]**: first target, then second (using first as feature), then third (using first and second)
- **order=None**: random order each fit; use **order_** attribute to inspect

```python
chain = ClassifierChain(LogisticRegression(), order=None)
chain.fit(X_train, y_train)
print(chain.order_)
```

---

## When to Use Each Approach

| Method | Use When |
|--------|----------|
| **MultiOutputClassifier** | Labels are independent; simple baseline |
| **MultiOutputRegressor** | Targets are independent |
| **RegressorChain** | Targets are correlated; order may matter |
| **ClassifierChain** | Labels are dependent; chain captures dependencies |
| **ClassifierChain(cv=k)** | Unknown label order; ensemble over random orders |

Chains add complexity but can improve performance when target dependencies exist. Start with multi-output for a baseline, then try chains if dependencies are suspected.
