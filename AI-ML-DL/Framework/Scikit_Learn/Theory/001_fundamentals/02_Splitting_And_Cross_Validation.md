# Splitting and Cross-Validation

---

## Table of Contents

- [Overview](#overview)
- [Train-Test Split](#train-test-split)
- [Stratified Splitting](#stratified-splitting)
- [K-Fold Cross-Validation](#k-fold-cross-validation)
- [Stratified K-Fold](#stratified-k-fold)
- [Leave-One-Out and Leave-P-Out](#leave-one-out-and-leave-p-out)
- [Time Series Splits](#time-series-splits)
- [Repeated and Shuffle Splits](#repeated-and-shuffle-splits)
- [cross_val_score and cross_validate](#cross_val_score-and-cross_validate)
- [Best Practices](#best-practices)

---

## Overview

**Data splitting** separates data into training and evaluation sets to estimate generalization performance. **Cross-validation** repeatedly splits data into folds, training on some and evaluating on others, to obtain more robust performance estimates. Proper splitting avoids **data leakage** and overfitting to a single holdout set.

| Method | Use Case |
|--------|----------|
| **train_test_split** | Simple holdout evaluation |
| **StratifiedKFold** | Classification with imbalanced classes |
| **KFold** | Regression, balanced classification |
| **TimeSeriesSplit** | Time-ordered data |
| **LeaveOneOut** | Small datasets, exhaustive evaluation |

---

## Train-Test Split

**train_test_split** randomly divides data into training and test sets. The test set is held out for final evaluation and must not influence training.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **test_size** | Fraction (0.0-1.0) or absolute number of test samples |
| **train_size** | Fraction or count for training (alternative to test_size) |
| **random_state** | Seed for reproducibility |
| **shuffle** | Whether to shuffle before splitting (default True) |
| **stratify** | Preserve class distribution (for classification) |

### Usage

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# With stratification for classification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### Important Notes

- Typical **test_size**: 0.2 to 0.3
- Always use **random_state** for reproducibility
- Use **stratify** when classes are imbalanced

---

## Stratified Splitting

**Stratified splitting** ensures that train and test sets have similar class proportions. Critical for imbalanced datasets where random splitting might omit minority classes from the test set.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# Each split maintains the same class ratio as the full dataset
```

---

## K-Fold Cross-Validation

**KFold** divides data into k consecutive folds. Each fold serves once as the validation set while the rest are used for training. Produces k performance estimates.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **n_splits** | Number of folds (default 5) |
| **shuffle** | Shuffle before splitting |
| **random_state** | Seed when shuffle=True |

### Usage

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate
```

### Important Notes

- **shuffle=True** recommended for non-time-series data
- More folds: lower bias, higher variance in estimates
- Common choices: 5 or 10 folds

---

## Stratified K-Fold

**StratifiedKFold** preserves class proportions in each fold. Each fold has approximately the same class distribution as the full dataset.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

### When to Use

- **Classification** with imbalanced or multi-class targets
- Prefer over **KFold** when class distribution matters

---

## Leave-One-Out and Leave-P-Out

**LeaveOneOut** uses each sample as a test set once (n folds for n samples). **LeavePOut** leaves p samples out. Exhaustive but computationally expensive.

```python
from sklearn.model_selection import LeaveOneOut, LeavePOut

loo = LeaveOneOut()
for train_idx, test_idx in loo.split(X):
    pass  # test_idx has exactly 1 element

lpo = LeavePOut(p=2)
for train_idx, test_idx in lpo.split(X):
    pass  # test_idx has 2 elements
```

### When to Use

- **LeaveOneOut**: Very small datasets
- **LeavePOut**: Rarely; KFold is usually sufficient

---

## Time Series Splits

**TimeSeriesSplit** creates expanding training windows. Training set grows over time; test set is always in the future. Prevents future information from leaking into training.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    # train_idx always before test_idx in time
    X_train, X_test = X[train_idx], X[test_idx]
```

### When to Use

- Time-ordered data (stock prices, sensor readings)
- Prevents look-ahead bias

---

## Repeated and Shuffle Splits

**RepeatedKFold** repeats KFold with different random seeds. **RepeatedStratifiedKFold** does the same for StratifiedKFold. **ShuffleSplit** randomly samples train/test indices without fold structure.

```python
from sklearn.model_selection import RepeatedKFold, ShuffleSplit

rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
# 15 total train/test splits

ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
for train_idx, test_idx in ss.split(X):
    pass
```

---

## cross_val_score and cross_validate

**cross_val_score** runs cross-validation and returns an array of scores. **cross_validate** runs CV and returns a dict with scores, fit times, and optionally train scores.

### cross_val_score

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator, X, y, cv=5, scoring="accuracy")
print(scores.mean(), scores.std())
```

### cross_validate

```python
from sklearn.model_selection import cross_validate

results = cross_validate(
    estimator, X, y, cv=5,
    scoring=["accuracy", "f1_macro"],
    return_train_score=True
)
# results: fit_time, score_time, test_accuracy, test_f1_macro, train_accuracy, train_f1_macro
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **cv** | Int (folds), CV splitter, or "prefit" |
| **scoring** | Metric name or dict of metrics |
| **return_train_score** | Include training scores |
| **n_jobs** | Parallel jobs (-1 = all cores) |

---

## Best Practices

| Practice | Reason |
|----------|--------|
| Use **stratify** for classification | Preserve class distribution |
| Use **Pipeline** with CV | Avoid leakage from preprocessing |
| Report **mean and std** of CV scores | Quantify uncertainty |
| Use **nested CV** for model selection | Unbiased performance estimate |
| Fix **random_state** | Reproducibility |

### Data Leakage Warning

Fit **transformers** (e.g., StandardScaler) only on training data. Use **Pipeline** so that cross_val_score fits the entire pipeline on each fold's training data and transforms test data correctly.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
scores = cross_val_score(pipe, X, y, cv=5)  # Correct: no leakage
```
