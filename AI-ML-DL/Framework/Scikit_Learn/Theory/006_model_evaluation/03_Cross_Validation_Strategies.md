# Cross-Validation Strategies

---

## Table of Contents

- [Overview](#overview)
- [KFold and StratifiedKFold](#kfold-and-stratifiedkfold)
- [Repeated KFold Variants](#repeated-kfold-variants)
- [GroupKFold and TimeSeriesSplit](#groupkfold-and-timeseriessplit)
- [LeaveOneOut and LeavePOut](#leaveoneout-and-leavepout)
- [ShuffleSplit and StratifiedShuffleSplit](#shufflesplit-and-stratifiedshufflesplit)
- [cross_val_score and cross_validate](#cross_val_score-and-cross_validate)

---

## Overview

**Cross-validation** splits data into folds, trains on some folds and evaluates on others, providing robust performance estimates. Different splitters suit different scenarios: **KFold** for standard regression, **StratifiedKFold** for classification with class balance, **TimeSeriesSplit** for temporal data.

---

## KFold and StratifiedKFold

### KFold

**KFold** divides data into K consecutive (or shuffled) folds. Each fold serves as the test set once.

```python
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf)
```

| Parameter | Description |
|-----------|-------------|
| **n_splits** | Number of folds |
| **shuffle** | Shuffle before splitting |
| **random_state** | Reproducibility |

### StratifiedKFold

**StratifiedKFold** preserves class proportions in each fold. Use for classification, especially with imbalanced classes.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=skf)
```

---

## Repeated KFold Variants

### RepeatedKFold

**RepeatedKFold** repeats KFold with different random shuffles. Increases stability of estimates.

```python
from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
# 5 x 3 = 15 total splits
```

### RepeatedStratifiedKFold

**RepeatedStratifiedKFold** repeats StratifiedKFold with different shuffles.

```python
from sklearn.model_selection import RepeatedStratifiedKFold

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
```

---

## GroupKFold and TimeSeriesSplit

### GroupKFold

**GroupKFold** ensures samples from the same group are not split across train and test. Use when you have grouped data (e.g., multiple samples per patient).

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, groups=groups, cv=gkf)
```

### TimeSeriesSplit

**TimeSeriesSplit** uses expanding window: train on past, test on future. Use for time series.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    # train_idx always before test_idx
    pass
```

---

## LeaveOneOut and LeavePOut

### LeaveOneOut

**LeaveOneOut** uses one sample as test, rest as train. N splits for N samples. High variance, expensive.

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
```

### LeavePOut

**LeavePOut** leaves P samples out as test. \(\binom{n}{p}\) splits. Very expensive for large P.

```python
from sklearn.model_selection import LeavePOut

lpo = LeavePOut(p=2)
```

---

## ShuffleSplit and StratifiedShuffleSplit

### ShuffleSplit

**ShuffleSplit** randomly samples train and test sets. Test sets can overlap across splits. Useful when you want many splits with limited data.

```python
from sklearn.model_selection import ShuffleSplit

ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores = cross_val_score(model, X, y, cv=ss)
```

| Parameter | Description |
|-----------|-------------|
| **n_splits** | Number of splits |
| **test_size** | Fraction or count for test |
| **train_size** | Fraction or count for train |

### StratifiedShuffleSplit

**StratifiedShuffleSplit** preserves class proportions in random splits. Use for classification.

```python
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
```

---

## cross_val_score and cross_validate

### cross_val_score

**cross_val_score** returns an array of scores, one per fold.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(scores.mean(), scores.std() * 2)
```

| Parameter | Description |
|-----------|-------------|
| **cv** | Int (folds), splitter, or iterable |
| **scoring** | Metric name or callable |

### cross_validate

**cross_validate** supports multiple metrics and returns a dict with fit times, score times, and scores.

```python
from sklearn.model_selection import cross_validate

results = cross_validate(
    model, X, y, cv=5,
    scoring=['accuracy', 'precision_macro', 'recall_macro'],
    return_train_score=True,
    return_estimator=True,
)
# results['test_accuracy'], results['train_accuracy']
# results['fit_time'], results['score_time']
# results['estimator']  # Fitted models per fold
```

### return_train_score

When `return_train_score=True`, compare train vs test to detect overfitting (high train, low test).

### return_estimator

When `return_estimator=True`, access the fitted estimator for each fold (e.g., for ensemble or inspection).

---

## Splitter Summary

| Splitter | Use Case |
|----------|----------|
| **KFold** | Standard regression |
| **StratifiedKFold** | Classification, preserve class balance |
| **RepeatedKFold** | More stable estimates |
| **GroupKFold** | Grouped data |
| **TimeSeriesSplit** | Time series |
| **LeaveOneOut** | Small data, exhaustive |
| **ShuffleSplit** | Many random splits |
| **StratifiedShuffleSplit** | Classification with random splits |

---
