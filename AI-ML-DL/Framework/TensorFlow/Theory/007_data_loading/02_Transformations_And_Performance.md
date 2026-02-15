# Dataset Transformations and Performance

## Table of Contents

1. [Core Transformations: map, batch, shuffle, repeat](#1-core-transformations-map-batch-shuffle-repeat)
2. [prefetch and Pipeline Ordering](#2-prefetch-and-pipeline-ordering)
3. [Caching and AUTOTUNE](#3-caching-and-autotune)
4. [interleave and Parallel map](#4-interleave-and-parallel-map)
5. [padded_batch and bucket_by_sequence_length](#5-padded_batch-and-bucket_by_sequence_length)

---

## 1. Core Transformations: map, batch, shuffle, repeat

### map

**map** applies a function to each element. The function can be a Python function or a `tf.function`:

```python
ds = tf.data.Dataset.range(5)
ds = ds.map(lambda x: x * 2)
# Elements: 0, 2, 4, 6, 8
```

For (features, labels) pairs:

```python
def preprocess(x, y):
    return tf.cast(x, tf.float32) / 255.0, y

ds = ds.map(preprocess)
```

### batch

**batch** combines consecutive elements into batches. The batch dimension is added as the first axis:

```python
ds = tf.data.Dataset.range(10)
ds = ds.batch(3)
# Batches: [0,1,2], [3,4,5], [6,7,8], [9]
```

Use **drop_remainder=True** when the last partial batch would cause issues (e.g., distributed training):

```python
ds = ds.batch(4, drop_remainder=True)
# Drops last 2 elements if total is 10
```

### shuffle

**shuffle** randomly shuffles elements using a buffer. Elements are drawn from the buffer; larger buffers improve randomness but use more memory:

```python
ds = tf.data.Dataset.range(100)
ds = ds.shuffle(buffer_size=50, seed=42)
```

For full-shuffle of a finite dataset, set `buffer_size` >= dataset size. Use **reshuffle_each_iteration=True** (default) to reshuffle each epoch.

### repeat

**repeat** repeats the dataset. With no argument, repeats infinitely (common for training):

```python
ds = tf.data.Dataset.range(3)
ds = ds.repeat(2)
# Elements: 0,1,2,0,1,2
```

```python
ds = ds.repeat()  # Infinite
```

| Transformation | Purpose |
|----------------|---------|
| map | Apply per-element function |
| batch | Group elements into batches |
| shuffle | Randomize order |
| repeat | Repeat dataset N times or infinitely |

---

## 2. prefetch and Pipeline Ordering

**prefetch** overlaps data preparation with model execution. While the model trains on batch N, the pipeline prepares batch N+1.

```python
ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
```

### Recommended Pipeline Order

1. **Read** (from files, etc.)
2. **Parse** (decode, parse)
3. **map** (preprocess, augment)
4. **shuffle** (if training)
5. **batch**
6. **repeat** (if infinite epochs)
7. **prefetch**

```python
pipeline = (
    tf.data.Dataset.list_files("data/*.tfrecord")
    .interleave(tf.data.TFRecordDataset)
    .map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1000)
    .batch(32)
    .repeat()
    .prefetch(tf.data.AUTOTUNE)
)
```

---

## 3. Caching and AUTOTUNE

### cache

**cache** caches elements in memory (or on disk) so expensive operations (e.g., decoding, parsing) run only once:

```python
ds = tf.data.Dataset.range(5)
ds = ds.map(expensive_fn).cache()
# First epoch: runs expensive_fn. Later epochs: use cache.
```

Cache to a file for datasets that do not fit in memory:

```python
ds = ds.cache("/tmp/cache_dir")
```

### AUTOTUNE

**tf.data.AUTOTUNE** lets TensorFlow choose parallelism based on available CPU:

```python
ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.prefetch(tf.data.AUTOTUNE)
```

Use AUTOTUNE for `num_parallel_calls` in `map` and for `buffer_size` in `prefetch` to tune performance automatically.

### options()

Set global options for the pipeline:

```python
options = tf.data.Options()
options.experimental_optimization.map_parallelization = True
ds = ds.with_options(options)
```

---

## 4. interleave and Parallel map

### interleave

**interleave** reads from multiple datasets concurrently and interleaves their elements. Useful for reading from many files:

```python
files = ["a.tfrecord", "b.tfrecord", "c.tfrecord"]
ds = tf.data.Dataset.from_tensor_slices(files)
ds = ds.interleave(
    tf.data.TFRecordDataset,
    cycle_length=3,
    block_length=1,
    num_parallel_calls=tf.data.AUTOTUNE
)
```

- **cycle_length**: Number of datasets to read from concurrently
- **block_length**: Number of consecutive elements from each dataset before switching

### num_parallel_calls in map

Parallelize the map transformation:

```python
ds = ds.map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)
```

This uses multiple threads to apply the map function, improving throughput when the function is CPU-bound.

---

## 5. padded_batch and bucket_by_sequence_length

### padded_batch

When elements have variable length (e.g., sequences), **padded_batch** batches them and pads to a common shape:

```python
sequences = [
    tf.constant([1, 2, 3]),
    tf.constant([4, 5]),
    tf.constant([6, 7, 8, 9])
]
ds = tf.data.Dataset.from_tensor_slices(sequences)
ds = ds.padded_batch(
    batch_size=2,
    padded_shapes=[4],
    padding_values=0
)
# Batches: [[1,2,3,0], [4,5,0,0]], [[6,7,8,9]]
```

**padded_shapes** can use `None` or `-1` for variable dimensions:

```python
ds = ds.padded_batch(
    batch_size=2,
    padded_shapes=[None],
    padding_values=0
)
```

### bucket_by_sequence_length

For variable-length sequences, **bucket_by_sequence_length** groups similar-length sequences into buckets to minimize padding:

```python
def element_length(x, y):
    return tf.cast(tf.shape(x)[0], tf.int64)

ds = ds.apply(
    tf.data.experimental.bucket_by_sequence_length(
        element_length,
        bucket_boundaries=[10, 20, 40],
        bucket_batch_sizes=[32, 16, 8, 4],
        padded_shapes=([None], []),
        padding_values=(0, -1)
    )
)
```

- **bucket_boundaries**: [10, 20, 40] creates buckets: [0,10), [10,20), [20,40), [40,inf)
- **bucket_batch_sizes**: Batch size per bucket (often larger for shorter sequences)
- **padded_shapes**: Shape for each component; use `[None]` for variable-length
- **padding_values**: Padding value per component

| Method | Use Case |
|--------|----------|
| batch | Fixed-shape elements |
| padded_batch | Variable-length, simple padding |
| bucket_by_sequence_length | Variable-length, minimize padding waste |
