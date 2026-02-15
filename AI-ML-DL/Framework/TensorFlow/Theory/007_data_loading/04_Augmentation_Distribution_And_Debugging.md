# Augmentation, Distribution, and Debugging

## Table of Contents

1. [Data Augmentation in the Pipeline](#1-data-augmentation-in-the-pipeline)
2. [Distributed Data Loading](#2-distributed-data-loading)
3. [Data Validation Pipeline](#3-data-validation-pipeline)
4. [Debugging Data Pipelines](#4-debugging-data-pipelines)

---

## 1. Data Augmentation in the Pipeline

### Augmentation via map

Apply augmentation inside a **map** transformation. This runs on CPU in parallel with GPU training when using **prefetch**:

```python
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
```

### Keras Preprocessing Layers

Use **tf.keras.layers** for resizing, rescaling, and augmentation:

```python
resize = tf.keras.layers.Resizing(224, 224)
rescale = tf.keras.layers.Rescaling(1.0 / 255.0)

def preprocess(image, label):
    image = resize(image)
    image = rescale(image)
    return image, label

ds = ds.map(preprocess)
```

### Conditional Augmentation (Train vs Eval)

Apply augmentation only during training:

```python
def process(image, label, is_training):
    image = resize(image)
    image = rescale(image)
    if is_training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label

# Training
ds_train = ds.map(lambda x, y: process(x, y, True))

# Validation
ds_val = ds.map(lambda x, y: process(x, y, False))
```

### Best Practices

- Use **num_parallel_calls=tf.data.AUTOTUNE** for augmentation map
- Place augmentation **before** batch (per-image) or **after** batch (per-batch) depending on the operation
- Ensure **clip_by_value** or similar to keep values in valid range
- Use **tf.image** for image augmentation; it is GPU-compatible when used in model

---

## 2. Distributed Data Loading

### Sharding

When using **tf.distribute**, each worker should process a different shard of the data. Use **shard**:

```python
ds = tf.data.Dataset.list_files("data/*.tfrecord")
ds = ds.shard(num_shards=num_workers, index=worker_index)
ds = ds.interleave(tf.data.TFRecordDataset)
```

With **tf.distribute.Strategy**, the strategy's **distribute_datasets_from_function** or **experimental_distribute_dataset** handles sharding automatically when the dataset is created inside the strategy scope.

### options() for Performance

```python
options = tf.data.Options()
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.map_and_batch_fusion = True
ds = ds.with_options(options)
```

### Determinism

For reproducible training, set **deterministic=True**:

```python
options = tf.data.Options()
options.deterministic = True
ds = ds.with_options(options)
```

Note: Some operations (e.g., shuffle) may still vary across runs if seeds differ. Use fixed seeds for full reproducibility.

### Global Shuffle

For distributed training, shuffle before sharding so each worker gets a random subset:

```python
ds = ds.shuffle(buffer_size=dataset_size, seed=42)
ds = ds.shard(num_shards=num_workers, index=worker_index)
```

---

## 3. Data Validation Pipeline

### filter

Remove invalid or unwanted samples:

```python
ds = ds.filter(lambda x, y: tf.reduce_all(tf.math.is_finite(x)))
ds = ds.filter(lambda x, y: y >= 0)
```

### tf.debugging.assert

Add assertions inside a map:

```python
def validate(x, y):
    tf.debugging.assert_non_negative(x, message="x must be non-negative")
    tf.debugging.assert_less_equal(y, num_classes - 1, message="Invalid label")
    return x, y

ds = ds.map(validate)
```

If an assertion fails, the pipeline raises an error. Use with care in production; prefer **filter** for robustness.

### assert_cardinality

Ensure the dataset has exactly N elements:

```python
ds = ds.apply(tf.data.experimental.assert_cardinality(1000))
```

Useful when the dataset size must match expectations (e.g., steps per epoch).

### Filter Invalid Labels

```python
ds = ds.filter(lambda x, y: tf.logical_and(y >= 0, y < num_classes))
```

### Summary

| Method | Purpose |
|--------|---------|
| filter | Remove invalid samples |
| tf.debugging.assert | Fail fast on invalid data |
| assert_cardinality | Verify dataset size |

---

## 4. Debugging Data Pipelines

### take

Limit the dataset to a few elements for quick inspection:

```python
ds = ds.take(5)
for batch in ds:
    print(batch)
```

### as_numpy_iterator

Convert to Python/NumPy for inspection:

```python
ds = tf.data.Dataset.range(5)
values = list(ds.as_numpy_iterator())
print(values)
```

### element_spec

Inspect the structure of each element:

```python
print(ds.element_spec)
```

### reduce

Count elements or aggregate:

```python
count = ds.reduce(0, lambda state, x: state + 1)
print(count.numpy())
```

### skip

Skip the first N elements (e.g., to bypass a problematic prefix):

```python
ds = ds.skip(10)
```

### Step-by-Step Debugging

Build the pipeline incrementally and inspect after each transformation:

```python
ds = tf.data.Dataset.range(10)
print(list(ds.take(3).as_numpy_iterator()))

ds = ds.map(lambda x: x * 2)
print(list(ds.take(3).as_numpy_iterator()))

ds = ds.batch(2)
for batch in ds.take(2):
    print(batch.numpy())
```

### Common Issues

| Issue | Check |
|-------|-------|
| Wrong shape | Print element_spec, batch shapes |
| Empty dataset | Use reduce to count, check filter |
| Slow pipeline | Add prefetch, increase num_parallel_calls |
| OOM | Reduce batch size, use cache to disk |
| Non-determinism | Set options.deterministic = True |

---

## Summary Table

| Topic | Key Concepts |
|-------|--------------|
| Augmentation | map with tf.image, conditional for train/eval |
| Distribution | shard, options, deterministic |
| Validation | filter, tf.debugging.assert, assert_cardinality |
| Debugging | take, as_numpy_iterator, element_spec, reduce |
