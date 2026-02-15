# Broadcasting, Concatenation, and Math

## Table of Contents

1. [Broadcasting Rules](#1-broadcasting-rules)
2. [Compatible Shapes](#2-compatible-shapes)
3. [Concatenation and Stacking](#3-concatenation-and-stacking)
4. [Splitting Tensors](#4-splitting-tensors)
5. [Element-wise Mathematical Operations](#5-element-wise-mathematical-operations)
6. [tf.math Functions](#6-tfmath-functions)
7. [Reduction Operations](#7-reduction-operations)
8. [Comparison and Logical Operations](#8-comparison-and-logical-operations)
9. [tf.where and Sorting](#9-tfwhere-and-sorting)

---

## 1. Broadcasting Rules

**Broadcasting** allows operations between tensors of different shapes by virtually expanding the smaller tensor. No extra memory is allocated for the expanded form.

Rules:
1. Dimensions are compared from right to left.
2. Two dimensions are compatible if they are equal or one is 1.
3. The smaller tensor is broadcast to match the larger.

```python
a = tf.constant([[1], [2], [3]])
b = tf.constant([10, 20, 30])
c = a + b
```

---

## 2. Compatible Shapes

| Shape A | Shape B | Result |
|---------|---------|--------|
| (3, 4) | (4,) | (3, 4) |
| (3, 1) | (1, 4) | (3, 4) |
| (5, 3, 4) | (3, 4) | (5, 3, 4) |
| (3,) | (4,) | Incompatible |

### tf.broadcast_to

Explicitly broadcast a tensor to a target shape.

```python
t = tf.constant([1, 2, 3])
b = tf.broadcast_to(t, [2, 3])
```

---

## 3. Concatenation and Stacking

### tf.concat

Joins tensors along an existing axis. All dimensions except the concatenation axis must match.

```python
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
cat0 = tf.concat([a, b], axis=0)
cat1 = tf.concat([a, b], axis=1)
```

### tf.stack

Creates a new dimension and stacks tensors along it. All input shapes must be identical.

```python
st = tf.stack([a, b], axis=0)
```

| Operation | New dimension | Input requirement |
|-----------|---------------|-------------------|
| concat | No | Same shape except axis |
| stack | Yes | Identical shapes |

---

## 4. Splitting Tensors

### tf.unstack

Removes a dimension and returns a list of tensors.

```python
t = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
parts = tf.unstack(t, axis=0)
```

### tf.split

Splits a tensor along an axis into multiple tensors.

```python
t = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
parts = tf.split(t, 2, axis=1)
parts = tf.split(t, [1, 3], axis=1)
```

---

## 5. Element-wise Mathematical Operations

Standard arithmetic operators apply element-wise: `+`, `-`, `*`, `/`, `**`.

```python
a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])
print(a + b)
print(a * b)
print(a ** 2)
```

---

## 6. tf.math Functions

| Function | Description |
|----------|-------------|
| sin, cos | Trigonometric |
| exp, log | Exponential, natural log |
| sqrt | Square root |
| abs | Absolute value |
| sign | Sign (-1, 0, 1) |
| floor, ceil, round | Rounding |
| clip_by_value | Clamp to range |

```python
x = tf.constant([0.0, 3.14159/2])
print(tf.math.sin(x))
print(tf.math.exp(tf.constant([1.0, 2.0])))
print(tf.clip_by_value(x, 0.5, 2.0))
```

---

## 7. Reduction Operations

Reductions collapse one or more dimensions. Common parameter: `axis` (dimension to reduce), `keepdims` (whether to keep reduced dimensions as 1).

### tf.reduce_sum, reduce_mean, reduce_max, reduce_min

```python
t = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(tf.reduce_sum(t))
print(tf.reduce_mean(t, axis=1))
print(tf.reduce_max(t, axis=0))
```

### tf.reduce_prod

Product of elements.

```python
print(tf.reduce_prod(tf.constant([1.0, 2.0, 3.0])))
```

### tf.math.reduce_std

Standard deviation.

```python
print(tf.math.reduce_std(tf.constant([1.0, 2.0, 3.0, 4.0])))
```

### keepdims

Preserves reduced dimensions as size 1 for broadcasting.

```python
s = tf.reduce_sum(t, axis=1, keepdims=True)
```

---

## 8. Comparison and Logical Operations

### Comparison

| Operation | Symbol/Function |
|-----------|-----------------|
| Equal | tf.equal |
| Not equal | tf.not_equal |
| Greater | tf.greater |
| Less | tf.less |
| Greater or equal | tf.greater_equal |
| Less or equal | tf.less_equal |

```python
a = tf.constant([1, 2, 3])
b = tf.constant([1, 3, 2])
print(tf.equal(a, b))
print(tf.greater(a, b))
```

### Logical

| Operation | Function |
|-----------|----------|
| AND | tf.logical_and |
| OR | tf.logical_or |
| NOT | tf.logical_not |

```python
x = tf.constant([True, False])
y = tf.constant([True, True])
print(tf.logical_and(x, y))
```

---

## 9. tf.where and Sorting

### tf.where (three-argument form)

Selects from two tensors based on a condition.

```python
cond = tf.constant([True, False, True])
t = tf.constant([1.0, 2.0, 3.0])
f = tf.constant([0.0, 0.0, 0.0])
result = tf.where(cond, t, f)
```

### tf.where (one-argument form)

Returns indices where condition is True.

```python
t = tf.constant([1, 2, 3, 4, 5])
indices = tf.where(tf.greater(t, 3))
```

### tf.sort and tf.argsort

```python
vals = tf.constant([3, 1, 4, 1, 5])
sorted_vals = tf.sort(vals)
sorted_desc = tf.sort(vals, direction="DESCENDING")
indices = tf.argsort(vals)
ordered = tf.gather(vals, indices)
```
