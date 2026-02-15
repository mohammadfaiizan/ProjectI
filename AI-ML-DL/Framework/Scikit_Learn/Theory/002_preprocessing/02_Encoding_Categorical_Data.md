# Encoding Categorical Data

---

## Table of Contents

- [Overview](#overview)
- [LabelEncoder](#labelencoder)
- [OrdinalEncoder](#ordinalencoder)
- [OneHotEncoder](#onehotencoder)
- [TargetEncoder](#targetencoder)
- [Comparison and Selection](#comparison-and-selection)

---

## Overview

Machine learning models require numeric input. **Categorical encoding** converts categorical variables into numeric representations. The choice of encoder affects model performance and interpretability.

| Encoder | Output | Use Case |
|---------|--------|----------|
| LabelEncoder | Single integer per category | Target variable (classification) |
| OrdinalEncoder | Integer per category per column | Ordinal features with known order |
| OneHotEncoder | Binary columns (one per category) | Nominal features, no order |
| TargetEncoder | Mean target per category | Supervised encoding, high cardinality |

---

## LabelEncoder

**LabelEncoder** encodes a single target vector (1D array) into integers. Designed for **target variables** in classification, not for feature encoding.

### Key Attributes

- **classes_**: Array of unique classes in the order they were encoded

### Methods

- **fit(y)**: Learn the mapping from labels to integers
- **transform(y)**: Encode labels
- **inverse_transform(y_encoded)**: Decode back to labels

### Usage

```python
from sklearn.preprocessing import LabelEncoder

y = ['red', 'blue', 'green', 'blue', 'red']
le = LabelEncoder()
le.fit(y)

print(le.classes_)  # ['blue', 'green', 'red']
y_encoded = le.transform(y)  # [2, 0, 1, 0, 2]
y_decoded = le.inverse_transform(y_encoded)
```

### Important Notes

- **Do not use for features**: Use **OrdinalEncoder** or **OneHotEncoder** instead
- Encodes alphabetically by default
- Unknown labels at transform time raise an error

---

## OrdinalEncoder

**OrdinalEncoder** encodes multiple categorical columns into integers. Supports explicit category ordering and handling of unknown categories.

### Key Attributes

- **categories_**: List of arrays, one per column, defining category order

### Parameters

- **categories**: List of category arrays to enforce order
- **handle_unknown**: 'error' (default), 'use_encoded_value' (with **unknown_value**)

### Usage

```python
from sklearn.preprocessing import OrdinalEncoder

X = [['small', 'low'], ['medium', 'high'], ['large', 'medium']]
enc = OrdinalEncoder()

# Auto-inferred order
enc.fit(X)
X_encoded = enc.transform(X)

# Explicit order
enc = OrdinalEncoder(categories=[['small', 'medium', 'large'],
                                 ['low', 'medium', 'high']])
X_encoded = enc.fit_transform(X)

# Handle unknown at transform
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
enc.fit(X)
X_new = [['extra', 'unknown']]
X_new_encoded = enc.transform(X_new)  # Uses -1 for unknown
```

### When to Use

- **Ordinal** features with meaningful order (e.g., size: S, M, L)
- Tree-based models that can handle ordinal encoding
- When you need compact representation

---

## OneHotEncoder

**OneHotEncoder** creates binary (dummy) columns for each category. Each category becomes a separate binary feature. Standard choice for **nominal** categorical variables.

### Parameters

- **sparse_output**: True (sparse matrix) or False (dense array)
- **handle_unknown**: 'error', 'ignore' (encode as all zeros)
- **drop**: None, 'first', or list of categories to drop (reduces multicollinearity)
- **categories**: Explicit category lists per column

### Usage

```python
from sklearn.preprocessing import OneHotEncoder

X = [['red', 'small'], ['blue', 'large'], ['green', 'medium']]
enc = OneHotEncoder(sparse_output=False)
enc.fit(X)

X_encoded = enc.transform(X)
print(enc.get_feature_names_out())

# Drop first category to avoid multicollinearity
enc_drop = OneHotEncoder(drop='first', sparse_output=False)
X_dropped = enc_drop.fit_transform(X)

# Handle unknown at transform
enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
enc.fit(X)
X_new = [['purple', 'tiny']]
X_new_encoded = enc.transform(X_new)  # Unknown -> zeros
```

### Important Notes

- **Multicollinearity**: With intercept, use `drop='first'` to avoid redundant columns
- **High cardinality**: Many categories lead to many columns; consider **TargetEncoder**
- **Sparse output**: Use `sparse_output=True` for memory efficiency with many categories

---

## TargetEncoder

**TargetEncoder** (supervised encoding) replaces each category with the mean of the target variable for that category. Useful for high-cardinality categorical features.

### Parameters

- **smooth**: 'auto' or float; smooths encoding with global mean to reduce overfitting
- **cv**: Cross-validation folds for fitting (reduces target leakage)

### Usage

```python
from sklearn.preprocessing import TargetEncoder

X = [['A', 'X'], ['B', 'Y'], ['A', 'Y'], ['B', 'X']]
y = [1, 0, 1, 0, 1, 0]

enc = TargetEncoder()
enc.fit(X, y)
X_encoded = enc.transform(X)

# With smoothing
enc_smooth = TargetEncoder(smooth='auto')
enc_smooth.fit(X, y)
X_smooth = enc_smooth.transform(X)
```

### Important Notes

- **Target leakage**: Encoding uses target information; use **cv** or fit only on train split
- **Overfitting**: Small category counts can overfit; use **smooth**
- Best for **high-cardinality** categorical features in supervised learning

---

## Comparison and Selection

| Encoder | Pros | Cons |
|---------|------|------|
| LabelEncoder | Simple, compact | For targets only; arbitrary order for features |
| OrdinalEncoder | Compact, preserves order | Implies distance between categories |
| OneHotEncoder | No implied order, interpretable | Many columns for high cardinality |
| TargetEncoder | Captures target relationship, compact | Risk of leakage; needs careful validation |

### Selection Guide

- **Target (y)**: Use **LabelEncoder**
- **Ordinal features** (size, rating): Use **OrdinalEncoder** with explicit order
- **Nominal, low cardinality**: Use **OneHotEncoder** (with `drop='first'` if using linear models)
- **Nominal, high cardinality**: Use **TargetEncoder** with smoothing and proper CV
