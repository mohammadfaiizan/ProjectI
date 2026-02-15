# Shapes, Indexing, and Slicing

## Table of Contents

1. [Shape Properties](#1-shape-properties)
2. [Rank and Dimensions](#2-rank-and-dimensions)
3. [Reshaping Tensors](#3-reshaping-tensors)
4. [Basic Indexing and Slicing](#4-basic-indexing-and-slicing)
5. [tf.gather and tf.gather_nd](#5-tfgather-and-tfgather_nd)
6. [tf.boolean_mask](#6-tfboolean_mask)
7. [Transpose and Dimension Manipulation](#7-transpose-and-dimension-manipulation)
8. [expand_dims and squeeze](#8-expand_dims-and-squeeze)
9. [tf.tile](#9-tftile)

---

## 1. Shape Properties

The **shape** of a tensor describes the size of each dimension. It is a `TensorShape` object that can be converted to a list.

```python
t = tf.constant([[1, 2, 3], [4, 5, 6]])
print(t.shape)
print(t.shape.as_list())
```

For partially known shapes (e.g., batch dimension unknown), some dimensions may be `None`.

---

## 2. Rank and Dimensions

**Rank** (or **ndim**) is the number of dimensions. A scalar has rank 0, a vector has rank 1, a matrix has rank 2.

```python
t = tf.constant([[1, 2], [3, 4]])
print(t.ndim)
print(tf.rank(t))
```

**size** returns the total number of elements.

```python
print(tf.size(t).numpy())
```

| Concept | Description |
|---------|-------------|
| shape | Size per dimension |
| rank/ndim | Number of dimensions |
| size | Total element count |

---

## 3. Reshaping Tensors

### tf.reshape

Changes the shape while preserving the total number of elements. The new shape must be compatible.

```python
t = tf.constant([[1, 2, 3], [4, 5, 6]])
flat = tf.reshape(t, [6])
row = tf.reshape(t, [1, 6])
col = tf.reshape(t, [6, 1])
```

### Inferred dimension with -1

At most one dimension can be `-1`, which is inferred from the total size.

```python
t = tf.constant([[1, 2], [3, 4], [5, 6]])
r = tf.reshape(t, [-1])
s = tf.reshape(t, [2, -1])
```

---

## 4. Basic Indexing and Slicing

TensorFlow supports NumPy-style indexing and slicing.

### Indexing

```python
t = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(t[0])
print(t[1, 2])
print(t[-1])
```

### Slicing

`start:stop:step` syntax applies per dimension.

```python
print(t[1:3])
print(t[:, 1])
print(t[0:2, 1:3])
print(t[::2, ::2])
```

| Syntax | Meaning |
|--------|---------|
| t[i] | Row i |
| t[i, j] | Element at (i, j) |
| t[i:j] | Rows i to j-1 |
| t[:, k] | Column k |
| t[::2] | Every other row |

---

## 5. tf.gather and tf.gather_nd

### tf.gather

Gathers slices along an axis using indices.

```python
t = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
indices = tf.constant([0, 2])
rows = tf.gather(t, indices, axis=0)
cols = tf.gather(t, [1, 2], axis=1)
```

### tf.gather_nd

Gathers elements using multi-dimensional indices. Each index in the indices tensor specifies a full coordinate.

```python
indices = tf.constant([[0, 0], [1, 1], [2, 2]])
values = tf.gather_nd(t, indices)
```

---

## 6. tf.boolean_mask

Extracts elements where the mask is True. The mask must be broadcastable to the tensor shape.

```python
t = tf.constant([[1, 2], [3, 4], [5, 6]])
mask = tf.constant([True, False, True])
rows = tf.boolean_mask(t, mask, axis=0)
```

For element-wise masking, the mask shape must match the tensor shape.

```python
mask = t > 3
values = tf.boolean_mask(t, mask)
```

---

## 7. Transpose and Dimension Manipulation

### tf.transpose

Permutes dimensions. By default, reverses all dimensions.

```python
t = tf.constant([[1, 2], [3, 4]])
tp = tf.transpose(t)
```

For higher-rank tensors, use `perm` to specify the new order.

```python
t = tf.ones([2, 3, 4])
tp = tf.transpose(t, perm=[1, 0, 2])
```

---

## 8. expand_dims and squeeze

### tf.expand_dims

Adds a dimension of size 1 at the specified axis.

```python
t = tf.constant([1, 2, 3])
t_batch = tf.expand_dims(t, axis=0)
t_feat = tf.expand_dims(t, axis=-1)
```

### tf.squeeze

Removes dimensions of size 1. By default removes all such dimensions.

```python
t = tf.constant([[[1], [2], [3]]])
s = tf.squeeze(t)
```

To squeeze only specific dimensions, use `axis`.

```python
t = tf.constant([[[1, 2, 3]]])
s = tf.squeeze(t, axis=0)
```

---

## 9. tf.tile

Repeats a tensor along each dimension. The multiples specify how many times to repeat along each axis.

```python
t = tf.constant([[1, 2], [3, 4]])
tiled = tf.tile(t, [2, 2])
```

For broadcasting-like behavior, first expand dimensions then tile.

```python
t = tf.constant([[1], [2], [3]])
tiled = tf.tile(t, [1, 4])
```

| Operation | Purpose |
|-----------|---------|
| reshape | Change layout, same elements |
| transpose | Permute dimensions |
| expand_dims | Add size-1 dimension |
| squeeze | Remove size-1 dimensions |
| tile | Repeat along axes |
