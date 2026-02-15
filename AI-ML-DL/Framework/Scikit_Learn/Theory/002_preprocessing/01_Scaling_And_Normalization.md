# Scaling and Normalization

---

## Table of Contents

- [Overview](#overview)
- [StandardScaler](#standardscaler)
- [MinMaxScaler](#minmaxscaler)
- [RobustScaler](#robustscaler)
- [MaxAbsScaler](#maxabsscaler)
- [Normalizer](#normalizer)
- [PowerTransformer](#powertransformer)
- [QuantileTransformer](#quantiletransformer)
- [When to Use Each](#when-to-use-each)

---

## Overview

**Scaling** and **normalization** transform features to comparable ranges, which is critical for algorithms sensitive to feature magnitude (e.g., SVM, neural networks, distance-based methods). Unscaled features can dominate the model and cause poor performance.

| Transformer | Formula | Use Case |
|-------------|---------|----------|
| StandardScaler | (x - mean) / std | Gaussian-like data |
| MinMaxScaler | (x - min) / (max - min) | Bounded [0, 1] output |
| RobustScaler | (x - median) / IQR | Outlier-resistant |
| MaxAbsScaler | x / max(abs(x)) | Sparse data |
| Normalizer | x / norm(x) | Per-sample scaling |
| PowerTransformer | Yeo-Johnson / Box-Cox | Skewed distributions |
| QuantileTransformer | Rank-based mapping | Non-linear, robust |

---

## StandardScaler

**StandardScaler** centers data to mean 0 and scales to unit variance (z-score normalization). It is the most common choice for many ML algorithms.

### Key Attributes

- **mean_**: Per-feature mean computed during `fit`
- **scale_**: Per-feature standard deviation (scale)

### Formula

\[
z = \frac{x - \mu}{\sigma}
\]

### Usage

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# Access fitted parameters
print(scaler.mean_)
print(scaler.scale_)

# Inverse transform restores original scale
X_restored = scaler.inverse_transform(X_scaled)
```

### Important Notes

- **fit** computes mean and std from training data only
- **transform** applies the same parameters to new data
- Sensitive to outliers; use **RobustScaler** if outliers exist

---

## MinMaxScaler

**MinMaxScaler** scales features to a specified range, default [0, 1]. Useful when you need bounded output or interpretable feature ranges.

### Key Attributes

- **data_min_**: Per-feature minimum
- **data_max_**: Per-feature maximum

### Formula

\[
x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}
\]

### Usage

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
```

### Important Notes

- Highly sensitive to outliers (min/max can be skewed)
- Use **feature_range** to set custom bounds, e.g. (-1, 1)

---

## RobustScaler

**RobustScaler** uses median and interquartile range (IQR), making it robust to outliers. Preferred when data contains extreme values.

### Formula

\[
x_{scaled} = \frac{x - \text{median}}{IQR}
\]

### Usage

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

### Important Notes

- IQR = Q3 - Q1 (75th - 25th percentile)
- Outliers have limited impact on median and IQR

---

## MaxAbsScaler

**MaxAbsScaler** scales by the maximum absolute value per feature. Output range is [-1, 1]. Preserves sparsity (zeros remain zero).

### Usage

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)
```

---

## Normalizer

**Normalizer** scales each sample (row) independently to unit norm. Used when the magnitude of a sample vector matters, not individual features.

### Norm Options

| norm | Description |
|------|-------------|
| l2 | Euclidean (default): divide by L2 norm |
| l1 | Manhattan: divide by L1 norm (sum of absolutes) |
| max | Divide by max absolute value |

### Usage

```python
from sklearn.preprocessing import Normalizer, normalize

# Using the transformer
norm = Normalizer(norm='l2')
X_norm = norm.fit_transform(X)

# Using the function (no fit needed)
X_norm = normalize(X, norm='l2')
```

### When to Use

- Text classification (TF-IDF vectors)
- Clustering (cosine similarity)
- When sample-wise scaling is required

---

## PowerTransformer

**PowerTransformer** reduces skewness by applying power transforms. Two methods:

### Yeo-Johnson

- Works with positive and negative values
- Default method

### Box-Cox

- Requires strictly positive data
- Often produces more symmetric distributions

### Usage

```python
from sklearn.preprocessing import PowerTransformer

# Yeo-Johnson (handles any values)
pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X)

# Box-Cox (positive data only)
pt_bc = PowerTransformer(method='box-cox')
X_transformed = pt_bc.fit_transform(X_positive)
```

### Key Attribute

- **lambdas_**: Optimal lambda per feature (Box-Cox / Yeo-Johnson parameter)

---

## QuantileTransformer

**QuantileTransformer** maps data to a uniform or normal distribution using quantile information. Non-linear and robust to outliers.

### Parameters

- **output_distribution**: 'uniform' or 'normal'
- **n_quantiles**: Number of quantiles to compute (default 1000)

### Usage

```python
from sklearn.preprocessing import QuantileTransformer

# Map to uniform [0, 1]
qt = QuantileTransformer(output_distribution='uniform')
X_uniform = qt.fit_transform(X)

# Map to standard normal
qt_norm = QuantileTransformer(output_distribution='normal')
X_normal = qt_norm.fit_transform(X)
```

### Important Notes

- Non-linear transformation
- Can distort relationships; use when distribution shape matters
- More computationally expensive than linear scalers

---

## When to Use Each

| Scenario | Recommended Transformer |
|----------|-------------------------|
| General purpose, no outliers | StandardScaler |
| Need [0, 1] range | MinMaxScaler |
| Many outliers | RobustScaler |
| Sparse data | MaxAbsScaler |
| Per-sample normalization | Normalizer |
| Skewed distributions | PowerTransformer |
| Non-linear, robust mapping | QuantileTransformer |
