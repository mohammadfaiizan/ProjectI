# Validation, Missing Data, and Statistics

## Table of Contents

1. [Data Validation](#1-data-validation)
2. [Schema and Range Checks](#2-schema-and-range-checks)
3. [Missing Data Handling](#3-missing-data-handling)
4. [Outlier Detection](#4-outlier-detection)
5. [Statistical Summaries](#5-statistical-summaries)
6. [Adaptation and Batch Statistics](#6-adaptation-and-batch-statistics)
7. [Data Quality Pipelines](#7-data-quality-pipelines)
8. [Best Practices](#8-best-practices)

---

## 1. Data Validation

**Data validation** ensures that inputs conform to expected schemas, types, and value ranges before processing or model training.

**Key concept:** Validating early prevents silent failures and improves model robustness.

```python
def validate_shape(data, expected_features):
    tf.debugging.assert_equal(tf.shape(data)[1], expected_features)
def validate_dtype(data, dtype=tf.float32):
    tf.debugging.assert_type(data, dtype)
```

---

## 2. Schema and Range Checks

### Shape Validation

```python
data = tf.constant([[1.0, 2.0], [3.0, 4.0]])
tf.debugging.assert_equal(tf.rank(data), 2)
tf.debugging.assert_equal(tf.shape(data)[1], 2)
```

### Range Validation

Use **tf.clip_by_value** to enforce bounds:

```python
x = tf.constant([0.5, 1.5, -0.2, 2.5])
clipped = tf.clip_by_value(x, 0.0, 1.0)
```

### Assert Positive / Non-Negative

```python
tf.debugging.assert_positive(x)
tf.debugging.assert_non_negative(x)
```

### NaN and Inf Checks

```python
tf.debugging.check_numerics(tensor, "Tensor contains NaN/Inf")
has_nan = tf.reduce_any(tf.math.is_nan(tensor))
has_inf = tf.reduce_any(tf.math.is_inf(tensor))
```

| Check | Function |
|-------|----------|
| Shape | tf.debugging.assert_equal(tf.shape(x), expected) |
| Dtype | tf.debugging.assert_type |
| Range | tf.clip_by_value |
| NaN/Inf | tf.debugging.check_numerics |

---

## 3. Missing Data Handling

### Imputation Strategies

**Mean imputation:** Replace missing values with the feature mean.

```python
mean = tf.reduce_mean(data, axis=0)
mask = tf.math.is_nan(data)
data_filled = tf.where(mask, tf.broadcast_to(mean, tf.shape(data)), data)
```

**Median imputation:** Use the median for robustness to outliers.

```python
sorted_data = tf.sort(data, axis=0)
n = tf.shape(data)[0]
median_idx = n // 2
median = tf.gather(sorted_data, median_idx, axis=0)
```

**Constant imputation:** Fill with a fixed value (e.g., 0 or -1).

```python
data_filled = tf.where(tf.math.is_nan(data), 0.0, data)
```

### Forward/Backward Fill (Sequences)

For time series, propagate the last valid value:

```python
# Simplified: use tf.scan or loop for full implementation
```

---

## 4. Outlier Detection

### IQR Method

**Interquartile range (IQR)** defines outliers as values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.

```python
sorted_vals = tf.sort(vals)
n = tf.size(vals)
q1_idx = tf.cast(tf.cast(n, tf.float32) * 0.25, tf.int32)
q3_idx = tf.cast(tf.cast(n, tf.float32) * 0.75, tf.int32)
q1 = tf.gather(sorted_vals, q1_idx)
q3 = tf.gather(sorted_vals, q3_idx)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
mask = (vals >= lower) & (vals <= upper)
filtered = tf.boolean_mask(vals, mask)
```

### Z-Score Method

Values beyond 3 standard deviations are often considered outliers.

```python
mean = tf.reduce_mean(data, axis=0)
std = tf.math.reduce_std(data, axis=0)
z_scores = tf.abs((data - mean) / (std + 1e-8))
outlier_mask = tf.reduce_any(z_scores > 3, axis=1)
```

---

## 5. Statistical Summaries

### Per-Feature Statistics

```python
mean = tf.reduce_mean(data, axis=0)
std = tf.math.reduce_std(data, axis=0)
var = tf.math.reduce_variance(data, axis=0)
min_val = tf.reduce_min(data, axis=0)
max_val = tf.reduce_max(data, axis=0)
```

### Percentiles

```python
sorted_data = tf.sort(tf.reshape(data, [-1]))
idx = tf.cast(tf.size(sorted_data) * 0.5, tf.int32)
median = tf.gather(sorted_data, tf.minimum(idx, tf.size(sorted_data) - 1))
```

### Correlation (Concept)

For correlation matrices, use `tf.linalg.matmul` on centered data or specialized ops.

---

## 6. Adaptation and Batch Statistics

**Keras preprocessing layers** use `adapt()` to compute statistics from a data sample. This supports incremental adaptation from batches.

```python
norm_layer = tf.keras.layers.Normalization(axis=-1)
for batch in dataset.batch(32):
    norm_layer.adapt(batch)
```

**Key concept:** Adaptation learns mean/variance (or other stats) from data. Use the same adapted layer at inference.

---

## 7. Data Quality Pipelines

Combine validation, imputation, and scaling in a pipeline:

```python
def preprocess_pipeline(data):
    data = tf.cast(data, tf.float32)
    data = tf.where(tf.math.is_nan(data), 0.0, data)
    norm_layer = tf.keras.layers.Normalization(axis=-1)
    norm_layer.adapt(data)
    return norm_layer(data)
```

---

## 8. Best Practices

| Practice | Description |
|----------|-------------|
| Validate early | Check shape, dtype, range before training |
| Handle missing data | Choose imputation strategy per feature |
| Detect outliers | Use IQR or z-score; clip or remove |
| Adapt on training data only | Never adapt on test data |
| Document assumptions | Record validation rules and imputation choices |
