# tf.data Fundamentals and Data Sources

## Table of Contents

1. [Introduction to tf.data.Dataset](#1-introduction-to-tfdatadataset)
2. [element_spec and Cardinality](#2-element_spec-and-cardinality)
3. [from_tensor_slices](#3-from_tensor_slices)
4. [from_generator](#4-from_generator)
5. [TFRecord Format](#5-tfrecord-format)

---

## 1. Introduction to tf.data.Dataset

The **tf.data.Dataset** API provides a flexible and efficient way to build input pipelines for TensorFlow models. It abstracts over various data sources and supports lazy, composable transformations.

**Key concepts:**
- **Lazy evaluation**: Transformations are not executed until data is consumed
- **Composability**: Chain multiple transformations into a pipeline
- **Performance**: Designed for high-throughput training with prefetching and parallelization

```python
import tensorflow as tf

ds = tf.data.Dataset.range(5)
for elem in ds:
    print(elem.numpy())
```

A dataset represents a sequence of elements. Each element can be a single tensor, a tuple of tensors, or a nested structure (e.g., dict).

---

## 2. element_spec and Cardinality

### element_spec

**element_spec** describes the structure and type of each element in the dataset. It is a nested structure of `TensorSpec` objects matching the element structure.

```python
ds = tf.data.Dataset.from_tensor_slices(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
print(ds.element_spec)
# TensorSpec(shape=(2,), dtype=tf.float32, name=None)
```

For nested structures (tuples, dicts), `element_spec` mirrors that structure:

```python
ds = tf.data.Dataset.from_tensor_slices({
    "features": tf.constant([[1.0], [2.0]]),
    "labels": tf.constant([0, 1])
})
print(ds.element_spec)
# {'features': TensorSpec(...), 'labels': TensorSpec(...)}
```

### cardinality

**cardinality** returns the number of elements in the dataset. For finite datasets, it is an integer. For infinite datasets (e.g., after `repeat()`), it returns `tf.data.INFINITE_CARDINALITY` (internally -1).

```python
ds_finite = tf.data.Dataset.range(10)
print(ds_finite.cardinality().numpy())  # 10

ds_infinite = tf.data.Dataset.range(5).repeat()
print(ds_infinite.cardinality())  # -1 (INFINITE_CARDINALITY)
```

| Property | Description |
|----------|-------------|
| element_spec | Structure and dtype of each element |
| cardinality | Number of elements (or INFINITE) |

---

## 3. from_tensor_slices

**Dataset.from_tensor_slices** creates a dataset by slicing the first dimension of the input tensors. Each slice along axis 0 becomes one element.

### From NumPy or Tensor Arrays

```python
import numpy as np

arr = np.array([[1, 2], [3, 4], [5, 6]])
ds = tf.data.Dataset.from_tensor_slices(arr)
# Elements: [1,2], [3,4], [5,6]
```

### From Dict

When passing a dict, the first dimension of each value must match. Each element is a dict of slices:

```python
data = {
    "features": tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    "labels": tf.constant([0, 1, 0])
}
ds = tf.data.Dataset.from_tensor_slices(data)
# Each element: {"features": [1,2], "labels": 0}, ...
```

### From Tuples

Useful for (features, labels) pairs:

```python
X = tf.constant([[1.0], [2.0], [3.0]])
y = tf.constant([0, 1, 0])
ds = tf.data.Dataset.from_tensor_slices((X, y))
# Elements: ([1.0], 0), ([2.0], 1), ([3.0], 0)
```

### Important Notes

- All tensors must have the same size in the first dimension
- Data is embedded in the graph when using `from_tensor_slices`; for large data, prefer TFRecord or `from_generator`
- Supports nested structures (dict, tuple, nested tuple)

---

## 4. from_generator

**Dataset.from_generator** creates a dataset from a Python generator. Use it when data cannot fit in memory or comes from external sources (files, APIs, databases).

### Basic Usage

```python
def gen():
    for i in range(5):
        yield i * 2

ds = tf.data.Dataset.from_generator(
    gen,
    output_signature=tf.TensorSpec(shape=(), dtype=tf.int32)
)
```

**output_signature** is required and must match the structure of yielded elements. Use `tf.TensorSpec` for tensors.

### Yielding Tuples

```python
def gen_pairs():
    for i in range(3):
        yield (tf.constant([float(i), float(i+1)]), tf.constant(i % 2))

ds = tf.data.Dataset.from_generator(
    gen_pairs,
    output_signature=(
        tf.TensorSpec(shape=(2,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)
```

### Yielding Dicts

```python
def gen_dict():
    for i in range(3):
        yield {"id": i, "value": float(i * 10)}

ds = tf.data.Dataset.from_generator(
    gen_dict,
    output_signature={
        "id": tf.TensorSpec(shape=(), dtype=tf.int32),
        "value": tf.TensorSpec(shape=(), dtype=tf.float32)
    }
)
```

### Infinite Generators

Generators can yield indefinitely. Use `take()` or `repeat()` with care:

```python
def infinite_gen():
    n = 0
    while True:
        yield n
        n += 1

ds = tf.data.Dataset.from_generator(
    infinite_gen,
    output_signature=tf.TensorSpec(shape=(), dtype=tf.int32)
)
limited = ds.take(10)
```

### When to Use

| Use Case | Recommended Source |
|----------|-------------------|
| Small in-memory data | from_tensor_slices |
| Large or streaming data | from_generator |
| File-based, distributed | TFRecordDataset |

---

## 5. TFRecord Format

**TFRecord** is TensorFlow's binary format for storing sequences of records. It is efficient for large datasets and works well with distributed training.

### Writing TFRecords

Use **tf.io.TFRecordWriter** to write serialized **tf.train.Example** protos:

```python
with tf.io.TFRecordWriter("data.tfrecord") as writer:
    for i in range(3):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                    "value": tf.train.Feature(float_list=tf.train.FloatList(value=[float(i*10)])),
                    "name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[f"item_{i}".encode()]))
                }
            )
        )
        writer.write(example.SerializeToString())
```

### Feature Types

| Feature Type | TensorFlow Type | Use Case |
|--------------|-----------------|----------|
| Int64List | tf.int64 | Integers, IDs |
| FloatList | tf.float32 | Floats |
| BytesList | tf.string | Strings, raw bytes, serialized images |

### Reading TFRecords

**tf.data.TFRecordDataset** reads TFRecord files. Parse each record with **tf.io.parse_single_example** or **tf.io.parse_example**:

```python
raw_ds = tf.data.TFRecordDataset("data.tfrecord")

def parse_fn(serialized):
    features = {
        "id": tf.io.FixedLenFeature([], tf.int64),
        "value": tf.io.FixedLenFeature([], tf.float32),
        "name": tf.io.FixedLenFeature([], tf.string)
    }
    return tf.io.parse_single_example(serialized, features)

ds = raw_ds.map(parse_fn)
```

### FixedLenFeature vs VarLenFeature

- **FixedLenFeature**: Fixed shape. Use `[]` for scalars, `[n]` for vectors.
- **VarLenFeature**: Variable-length. Returns `tf.SparseTensor`; use `tf.sparse.to_dense` to convert.

```python
features = {
    "ids": tf.io.VarLenFeature(tf.int64)
}
parsed = tf.io.parse_single_example(serialized, features)
ids = tf.sparse.to_dense(parsed["ids"])
```

### Multiple Files

```python
files = ["file1.tfrecord", "file2.tfrecord"]
ds = tf.data.TFRecordDataset(files)
```

TFRecord is ideal for large-scale, distributed training because files can be sharded across workers.
