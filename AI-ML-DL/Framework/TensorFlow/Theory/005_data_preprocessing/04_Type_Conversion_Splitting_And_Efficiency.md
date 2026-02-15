# Type Conversion, Splitting, and Efficiency

## Table of Contents

1. [Type Conversion](#1-type-conversion)
2. [Data Splitting Strategies](#2-data-splitting-strategies)
3. [Train/Validation/Test Splits](#3-trainvalidationtest-splits)
4. [Stratified and Time-Based Splits](#4-stratified-and-time-based-splits)
5. [tf.data Integration](#5-tfdata-integration)
6. [Efficient Preprocessing Pipelines](#6-efficient-preprocessing-pipelines)
7. [Caching and Prefetching](#7-caching-and-prefetching)
8. [Best Practices](#8-best-practices)

---

## 1. Type Conversion

### tf.cast

**tf.cast** converts tensors to a different dtype. Essential for ensuring compatibility with layers and loss functions.

```python
x = tf.constant([1, 2, 3])
x_float = tf.cast(x, tf.float32)
x_int64 = tf.cast(x, tf.int64)
```

### Common Conversions

| From | To | Use Case |
|------|-----|----------|
| int32 | float32 | Model input, normalization |
| float64 | float32 | Reduce memory, GPU compatibility |
| bool | float32 | Mask to weights |
| string | int32 | After StringLookup |

```python
labels = tf.constant([0, 1, 2])
labels_onehot = tf.one_hot(tf.cast(labels, tf.int32), depth=3)
```

---

## 2. Data Splitting Strategies

### Manual Index-Based Split

```python
n = len(data)
train_end = int(0.7 * n)
val_end = int(0.85 * n)
train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]
```

### Shuffle Then Split

```python
indices = tf.range(n)
shuffled = tf.random.shuffle(indices)
split_idx = int(0.8 * n)
train_idx = shuffled[:split_idx]
test_idx = shuffled[split_idx:]
train_data = tf.gather(data, train_idx)
test_data = tf.gather(data, test_idx)
```

---

## 3. Train/Validation/Test Splits

**Typical ratios:** 70/15/15 or 80/10/10 for train/val/test.

```python
n = 1000
train_size = int(0.7 * n)
val_size = int(0.15 * n)
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]
```

**Key concept:** Validation data is used for early stopping and hyperparameter tuning. Test data is held out for final evaluation only.

---

## 4. Stratified and Time-Based Splits

### Stratified Split

Preserve class distribution in each split. Use `tf.unique` and manual indexing per class.

```python
# Simplified: group by class, split each group, concatenate
classes, idx = tf.unique(labels)
# Implement per-class split logic
```

### Time-Based Split

For time series, split by time order (no shuffle):

```python
train_data = data[:int(0.8 * n)]
test_data = data[int(0.8 * n):]
```

---

## 5. tf.data Integration

### Dataset.take and skip

```python
ds = tf.data.Dataset.from_tensor_slices((X, y))
ds = ds.shuffle(1000)
train_size = int(0.7 * n)
ds_train = ds.take(train_size)
ds_val = ds.skip(train_size).take(int(0.15 * n))
ds_test = ds.skip(train_size + int(0.15 * n))
```

### Batch After Split

```python
ds_train = ds_train.batch(32).prefetch(tf.data.AUTOTUNE)
ds_val = ds_val.batch(32)
```

---

## 6. Efficient Preprocessing Pipelines

**Key concept:** Apply preprocessing inside `dataset.map()` so it runs on the fly during training. Use `tf.function` for map functions when possible.

```python
def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    return x, y

ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
```

### Preprocessing Layers in Model

For Keras preprocessing layers, include them in the model so they run on GPU and are saved with the model:

```python
model = tf.keras.Sequential([
    norm_layer,
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

---

## 7. Caching and Prefetching

### cache()

**cache** stores the dataset in memory (or on disk) after the first epoch. Use when preprocessing is expensive and data fits in memory.

```python
ds = ds.map(preprocess).cache().batch(32)
```

### prefetch()

**prefetch** overlaps data loading with training. While the model trains on batch N, the next batch is prepared.

```python
ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
```

### Pipeline Order

Recommended order: `map` -> `cache` (optional) -> `shuffle` -> `batch` -> `prefetch`.

```python
ds = (ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
      .cache()
      .shuffle(1000)
      .batch(32)
      .prefetch(tf.data.AUTOTUNE))
```

---

## 8. Best Practices

| Practice | Description |
|----------|-------------|
| Cast early | Ensure float32 for model inputs |
| Shuffle before split | Avoid temporal bias |
| Use AUTOTUNE | Let tf.data tune parallelism |
| Cache when possible | Avoid recomputing preprocessing |
| Prefetch | Overlap I/O with compute |
