# Discretization and Feature Generation

---

## Table of Contents

- [Overview](#overview)
- [KBinsDiscretizer](#kbinsdiscretizer)
- [Binarizer](#binarizer)
- [PolynomialFeatures](#polynomialfeatures)
- [FunctionTransformer](#functiontransformer)
- [Summary](#summary)

---

## Overview

**Discretization** converts continuous features into discrete bins. **Feature generation** creates new features from existing ones (e.g., polynomial terms, custom transforms). Both can improve model expressiveness or handle non-linear relationships.

| Transformer | Purpose |
|-------------|---------|
| KBinsDiscretizer | Bin continuous data into intervals |
| Binarizer | Threshold-based binary encoding |
| PolynomialFeatures | Polynomial and interaction terms |
| FunctionTransformer | Apply arbitrary functions |

---

## KBinsDiscretizer

**KBinsDiscretizer** bins continuous features into k discrete intervals. Useful for non-linear relationships, reducing noise, or creating ordinal features.

### Parameters

- **n_bins**: Number of bins (default 5)
- **encode**: 'ordinal' (bin index), 'onehot', 'onehot-dense'
- **strategy**: 'uniform' (equal width), 'quantile' (equal frequency), 'kmeans'

### Key Attribute

- **bin_edges_**: Array of bin edges per feature

### Strategy Comparison

| strategy | Description | Use Case |
|----------|-------------|----------|
| uniform | Equal-width bins | Uniformly distributed data |
| quantile | Equal-frequency bins | Skewed data, balanced bin sizes |
| kmeans | K-means clustering | Data-driven bin boundaries |

### Usage

```python
from sklearn.preprocessing import KBinsDiscretizer

X = [[1.5], [3.2], [5.0], [2.1], [4.8]]

# Ordinal encoding (bin indices 0, 1, 2, ...)
kbd = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
X_binned = kbd.fit_transform(X)
print(kbd.bin_edges_)

# One-hot encoding of bins
kbd_oh = KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='quantile')
X_oh = kbd_oh.fit_transform(X)
```

### Important Notes

- **uniform** can create empty bins for skewed data
- **quantile** ensures similar sample counts per bin
- **encode='ordinal'** implies order; use **onehot** if bins are nominal

---

## Binarizer

**Binarizer** converts values to 0 or 1 based on a threshold. Simple binary encoding for count-like or continuous features.

### Parameters

- **threshold**: Values > threshold become 1, else 0 (default 0.0)

### Usage

```python
from sklearn.preprocessing import Binarizer

X = [[1.0, 2.0], [3.0, 0.5], [2.5, 4.0]]
binarizer = Binarizer(threshold=2.0)
X_bin = binarizer.fit_transform(X)
# [[0, 0], [1, 0], [1, 1]]
```

### When to Use

- Converting counts to binary (e.g., "has feature" vs "does not")
- Simple threshold-based features
- **Note**: `Binarizer` is stateless; `fit` does nothing. Prefer **KBinsDiscretizer** with n_bins=2 for fit-based thresholds.

---

## PolynomialFeatures

**PolynomialFeatures** generates polynomial and interaction terms. Expands feature space to capture non-linear relationships.

### Parameters

- **degree**: Degree of polynomial (default 2)
- **interaction_only**: If True, only interaction terms (no x^2, x^3, etc.)
- **include_bias**: If True, add a constant 1 column (intercept)

### Generated Features (degree=2, 2 features x1, x2)

| include_bias | Features |
|--------------|----------|
| True | 1, x1, x2, x1^2, x1*x2, x2^2 |
| False | x1, x2, x1^2, x1*x2, x2^2 |

### Usage

```python
from sklearn.preprocessing import PolynomialFeatures

X = [[1, 2], [3, 4], [5, 6]]

# Full polynomial (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)
print(poly.get_feature_names_out())
# ['1', 'x0', 'x1', 'x0^2', 'x0 x1', 'x1^2']

# Interaction only (no squared terms)
poly_int = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_int = poly_int.fit_transform(X)
print(poly_int.get_feature_names_out())
# ['x0', 'x1', 'x0 x1']
```

### Important Notes

- **Curse of dimensionality**: High degree leads to many features; use regularization
- **interaction_only**: Useful when you want interactions but not higher powers
- **include_bias**: Set False if your model (e.g., LinearRegression) already has intercept

---

## FunctionTransformer

**FunctionTransformer** wraps a function to create a stateless or stateful transformer. Enables custom transformations within **Pipeline**.

### Parameters

- **func**: Callable to apply (e.g., np.log1p)
- **inverse_func**: Callable for inverse transform (optional)
- **validate**: If True, check input (default True)
- **kw_args**: Keyword args for func
- **inv_kw_args**: Keyword args for inverse_func

### Usage

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# Log transform
log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
X_log = log_transformer.fit_transform(X)
X_restored = log_transformer.inverse_transform(X_log)

# Custom function
def add_const(X, c=1.0):
    return X + c

transformer = FunctionTransformer(
    func=add_const,
    inverse_func=lambda X, c=1.0: X - c,
    kw_args={'c': 5.0},
    inv_kw_args={'c': 5.0}
)
X_transformed = transformer.fit_transform(X)
```

### Use Cases

- Log, sqrt, or other mathematical transforms
- Custom cleaning (e.g., string processing)
- Wrapping existing functions for use in **Pipeline**
- **validate=False** to skip input checks (e.g., for non-numeric data)

---

## Summary

| Transformer | Input | Output | Typical Use |
|-------------|-------|--------|-------------|
| KBinsDiscretizer | Continuous | Binned (ordinal/onehot) | Non-linear, noise reduction |
| Binarizer | Continuous | Binary 0/1 | Threshold-based features |
| PolynomialFeatures | Numeric | Polynomial terms | Non-linear regression |
| FunctionTransformer | Any | Transformed | Custom transforms in pipelines |
