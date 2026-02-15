# Linear Algebra and Random

## Table of Contents

1. [Matrix Multiplication](#1-matrix-multiplication)
2. [Determinant and Inverse](#2-determinant-and-inverse)
3. [Eigenvalue Decomposition](#3-eigenvalue-decomposition)
4. [Singular Value Decomposition](#4-singular-value-decomposition)
5. [Solving Linear Systems](#5-solving-linear-systems)
6. [Norms and Trace](#6-norms-and-trace)
7. [Cholesky and QR Decomposition](#7-cholesky-and-qr-decomposition)
8. [Random Number Generation](#8-random-number-generation)
9. [Seeds and Reproducibility](#9-seeds-and-reproducibility)
10. [Random Distributions](#10-random-distributions)

---

## 1. Matrix Multiplication

### tf.linalg.matmul

Performs matrix multiplication. For 2D tensors: `C[i,j] = sum_k A[i,k] * B[k,j]`. Supports batched multiplication for higher ranks.

```python
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
c = tf.linalg.matmul(a, b)
```

The `@` operator is equivalent for 2D tensors: `a @ b`.

---

## 2. Determinant and Inverse

### tf.linalg.det

Computes the determinant of a square matrix. The matrix must be invertible (non-zero determinant) for the inverse to exist.

```python
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
det = tf.linalg.det(a)
```

### tf.linalg.inv

Computes the matrix inverse. For `A`, returns `A^(-1)` such that `A @ A^(-1) = I`.

```python
inv = tf.linalg.inv(a)
identity = tf.linalg.matmul(a, inv)
```

---

## 3. Eigenvalue Decomposition

### tf.linalg.eigh

Eigenvalue decomposition for symmetric (or Hermitian) matrices. Returns eigenvalues and eigenvectors.

```python
sym = tf.constant([[4.0, 1.0], [1.0, 3.0]])
eigenvalues, eigenvectors = tf.linalg.eigh(sym)
```

For general matrices, use `tf.linalg.eig`.

---

## 4. Singular Value Decomposition

### tf.linalg.svd

Decomposes a matrix into `A = U @ diag(s) @ V^T`. Returns `U`, singular values `s`, and `V` (V is `V^H` for complex).

```python
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
u, s, v = tf.linalg.svd(a)
```

Singular values are returned in descending order. Useful for low-rank approximation and dimensionality reduction.

---

## 5. Solving Linear Systems

### tf.linalg.solve

Solves `A @ x = b` for `x`. More efficient than computing the inverse when only the solution is needed.

```python
A = tf.constant([[3.0, 1.0], [1.0, 2.0]])
b = tf.constant([[9.0], [8.0]])
x = tf.linalg.solve(A, b)
```

For batch solving, `A` and `b` can have a leading batch dimension.

---

## 6. Norms and Trace

### tf.linalg.norm

Computes vector or matrix norms.

```python
v = tf.constant([3.0, 4.0])
l2 = tf.linalg.norm(v)
l1 = tf.linalg.norm(v, ord=1)
```

| ord | Norm |
|-----|------|
| 2 (default) | L2 (Euclidean) |
| 1 | L1 (sum of abs) |
| "fro" | Frobenius (matrices) |

### tf.linalg.trace

Sum of diagonal elements of a matrix.

```python
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
tr = tf.linalg.trace(a)
```

---

## 7. Cholesky and QR Decomposition

### tf.linalg.cholesky

Cholesky decomposition for positive definite matrices: `A = L @ L^T` where `L` is lower triangular.

```python
pos_def = tf.constant([[4.0, 2.0], [2.0, 3.0]])
L = tf.linalg.cholesky(pos_def)
```

### tf.linalg.qr

QR decomposition: `A = Q @ R` where `Q` is orthogonal and `R` is upper triangular.

```python
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
q, r = tf.linalg.qr(a)
```

| Decomposition | Use case |
|---------------|----------|
| Cholesky | Positive definite systems |
| QR | Least squares, orthogonalization |
| SVD | Rank, approximation |
| Eigh | Symmetric eigenvalue problems |

---

## 8. Random Number Generation

TensorFlow provides random number generation through `tf.random` and `tf.random.Generator`.

### tf.random.uniform

Uniform distribution in `[minval, maxval)`.

```python
u = tf.random.uniform(shape=[2, 3], minval=0, maxval=1)
u_int = tf.random.uniform(shape=[5], minval=1, maxval=10, dtype=tf.int32)
```

### tf.random.normal

Normal (Gaussian) distribution with given mean and stddev.

```python
n = tf.random.normal(shape=[2, 3], mean=0.0, stddev=1.0)
```

### tf.random.truncated_normal

Normal distribution truncated to `[mean - 2*stddev, mean + 2*stddev]`. Useful for weight initialization to avoid extreme values.

```python
tn = tf.random.truncated_normal(shape=[5], mean=0.0, stddev=1.0)
```

### tf.random.shuffle

Randomly shuffles along the first dimension.

```python
arr = tf.constant([1, 2, 3, 4, 5])
shuffled = tf.random.shuffle(arr)
```

---

## 9. Seeds and Reproducibility

### tf.random.set_seed

Sets the global random seed. Same seed produces same sequence of random numbers.

```python
tf.random.set_seed(42)
a = tf.random.normal([2])
tf.random.set_seed(42)
b = tf.random.normal([2])
```

### tf.random.Generator

Stateful generator with its own seed. Better control over reproducibility and independence of streams.

```python
gen = tf.random.Generator.from_seed(123)
r1 = gen.uniform([3])
r2 = gen.normal([3])
```

`from_non_deterministic_state()` creates a generator with random initial state for non-reproducible runs.

---

## 10. Random Distributions

| Distribution | Function | Parameters |
|--------------|----------|------------|
| Uniform | tf.random.uniform | shape, minval, maxval |
| Normal | tf.random.normal | shape, mean, stddev |
| Truncated normal | tf.random.truncated_normal | shape, mean, stddev |
| Shuffle | tf.random.shuffle | tensor |

Generator-based API provides additional distributions and more control:

```python
gen = tf.random.Generator.from_seed(42)
uniform = gen.uniform(shape=[3])
normal = gen.normal(shape=[3])
```

For reproducibility in experiments, always set a seed before random operations. For production or security-sensitive code, use non-deterministic generation.
