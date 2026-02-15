# Broadcasting, Concatenation, and Mathematical Operations

## Table of Contents
1. [Broadcasting Rules](#broadcasting-rules)
2. [Concatenation and Stacking](#concatenation-and-stacking)
3. [Splitting Operations](#splitting-operations)
4. [Arithmetic Operations](#arithmetic-operations)
5. [Mathematical Functions](#mathematical-functions)
6. [Reduction Operations](#reduction-operations)
7. [Comparison and Logical Operations](#comparison-and-logical-operations)
8. [Sorting and Selection](#sorting-and-selection)

---

## Broadcasting Rules

Broadcasting allows operations between tensors of different shapes by virtually expanding smaller tensors to match larger ones, without copying data.

### The Three Rules

1. Compare dimensions from **right to left** (trailing dimensions first)
2. Two dimensions are compatible if they are **equal**, or one of them is **1**
3. Missing dimensions on the left are treated as **1**

### Examples

```
Shape A      Shape B      Result      Compatible?
(3, 4)    +  (4,)      = (3, 4)      Yes (4 == 4)
(3, 1)    +  (1, 4)    = (3, 4)      Yes (1 expands)
(3, 1, 4) +  (2, 1, 1) = (3, 2, 4)   Yes
(3, 4)    +  (3,)      = Error        No (4 != 3)
(3, 4)    +  (2, 4)    = Error        No (3 != 2)
```

### Code Examples

```python
# Scalar with tensor
5 + torch.tensor([1, 2, 3])                          # [6, 7, 8]

# Vector with matrix
torch.tensor([1, 2, 3]) + torch.tensor([[10, 20, 30],
                                         [40, 50, 60]])
# [[11, 22, 33], [41, 52, 63]]

# Column vector with row vector -> outer-product-like
col = torch.tensor([[1], [2], [3]])                   # (3, 1)
row = torch.tensor([[10, 20, 30, 40]])                # (1, 4)
col + row                                              # (3, 4)
```

### Explicit Broadcasting

```python
# Check compatibility
torch.broadcast_shapes((3, 1), (1, 4))   # torch.Size([3, 4])

# Materialize broadcasted tensors
x, y = torch.broadcast_tensors(col, row)
# x.shape == y.shape == (3, 4)
```

### Memory Behavior

Broadcasting does not copy data. Internally, PyTorch uses stride=0 along expanded dimensions. `expand()` creates this view explicitly; `repeat()` creates actual copies.

---

## Concatenation and Stacking

### cat (Concatenate)

Joins tensors along an **existing** dimension. All tensors must match in every dimension except the concatenation dimension.

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

torch.cat([a, b], dim=0)     # shape (4, 2) - stack rows
torch.cat([a, b], dim=1)     # shape (2, 4) - stack columns
```

### stack

Creates a **new** dimension and joins tensors along it. All tensors must have the same shape.

```python
torch.stack([a, b], dim=0)   # shape (2, 2, 2) - new batch dim
torch.stack([a, b], dim=1)   # shape (2, 2, 2) - new dim at pos 1
torch.stack([a, b], dim=2)   # shape (2, 2, 2) - new dim at pos 2
```

### cat vs stack

| | cat | stack |
|---|---|---|
| Dimension | Existing | New |
| Input shapes | Same except along cat dim | Identical |
| Output ndim | Same as input | Input ndim + 1 |

### Common Patterns

```python
# Batch creation from samples
samples = [torch.randn(3, 224, 224) for _ in range(32)]
batch = torch.stack(samples, dim=0)              # (32, 3, 224, 224)

# Merging batches
batch1 = torch.randn(16, 3, 32, 32)
batch2 = torch.randn(8, 3, 32, 32)
combined = torch.cat([batch1, batch2], dim=0)    # (24, 3, 32, 32)

# Feature fusion (channel-wise)
feat_a = torch.randn(32, 64, 56, 56)
feat_b = torch.randn(32, 128, 56, 56)
fused = torch.cat([feat_a, feat_b], dim=1)       # (32, 192, 56, 56)
```

### Performance

Always concatenate all tensors in a single `torch.cat` call rather than iteratively concatenating in a loop.

```python
# Bad: O(n^2) memory allocations
result = tensors[0]
for t in tensors[1:]:
    result = torch.cat([result, t], dim=0)

# Good: single allocation
result = torch.cat(tensors, dim=0)
```

---

## Splitting Operations

### chunk

Splits a tensor into a specified number of equal (or near-equal) chunks.

```python
tensor = torch.arange(24).reshape(4, 6)
chunks = torch.chunk(tensor, chunks=2, dim=0)     # 2 chunks of shape (2, 6)
```

If the tensor does not divide evenly, the last chunk is smaller.

### split

Splits by specifying either a uniform size or a list of sizes.

```python
# Uniform size
torch.split(tensor, 2, dim=0)                     # chunks of size 2

# Custom sizes
torch.split(tensor, [2, 3, 1], dim=1)             # sizes 2, 3, 1 along dim 1
```

### Split and Reconstruct

```python
parts = torch.split(tensor, 2, dim=0)
reconstructed = torch.cat(parts, dim=0)
torch.equal(tensor, reconstructed)                 # True
```

---

## Arithmetic Operations

### Element-Wise Operators

```python
a = torch.tensor([1.0, 2.0, 3.0, 4.0])
b = torch.tensor([2.0, 3.0, 4.0, 5.0])

a + b               # addition
a - b               # subtraction
a * b               # multiplication
a / b               # division
a // b              # floor division
a % b               # modulo
a ** 2              # power
```

### Function Equivalents

```python
torch.add(a, b)
torch.sub(a, b)
torch.mul(a, b)
torch.div(a, b)
torch.pow(a, 2)
torch.add(a, 10)        # scalar operand
```

### In-Place Operations

In-place operations (suffix `_`) modify the tensor directly without allocating new memory.

```python
a.add_(1)
a.mul_(2)
a.div_(2)
a.sqrt_()
```

**Warning**: In-place operations on tensors that require gradients can cause issues with autograd. Avoid in-place ops on leaf tensors in training loops.

---

## Mathematical Functions

### Trigonometric

```python
angles = torch.tensor([0, 3.14159/4, 3.14159/2])
torch.sin(angles)
torch.cos(angles)
torch.tan(angles)
torch.asin(torch.tensor([0.0, 0.5, 1.0]))
torch.acos(torch.tensor([1.0, 0.5, 0.0]))
torch.atan(torch.tensor([0.0, 1.0]))
```

### Hyperbolic

```python
torch.sinh(x)
torch.cosh(x)
torch.tanh(x)         # commonly used as activation function
```

### Exponential and Logarithmic

```python
torch.exp(x)           # e^x
torch.exp2(x)          # 2^x
torch.log(x)           # ln(x)
torch.log2(x)          # log2(x)
torch.log10(x)         # log10(x)
torch.log1p(x)         # ln(1 + x), numerically stable for small x
torch.expm1(x)         # e^x - 1, numerically stable for small x
```

### Power and Root

```python
torch.pow(base, exponent)
torch.sqrt(x)
torch.rsqrt(x)        # 1 / sqrt(x)
torch.square(x)
```

### Rounding

```python
values = torch.tensor([-2.7, -1.3, 0.8, 1.5, 2.9])
torch.floor(values)    # [-3, -2,  0,  1,  2]
torch.ceil(values)     # [-2, -1,  1,  2,  3]
torch.round(values)    # [-3, -1,  1,  2,  3]
torch.trunc(values)    # [-2, -1,  0,  1,  2]
torch.frac(values)     # fractional part
```

### Absolute, Sign, and Clamp

```python
torch.abs(x)
torch.sign(x)          # -1, 0, or 1
torch.clamp(x, min=-1, max=1)
torch.clamp(x, min=0)  # equivalent to ReLU
```

### Special Value Checks

```python
torch.isnan(x)
torch.isinf(x)
torch.isfinite(x)
torch.isposinf(x)
torch.isneginf(x)
```

---

## Reduction Operations

Reductions collapse one or more dimensions of a tensor into a single value.

### Sum, Mean, Product

```python
tensor = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]])

torch.sum(tensor)               # 45.0 (all elements)
torch.sum(tensor, dim=0)        # [12, 15, 18] (column sums)
torch.sum(tensor, dim=1)        # [6, 15, 24] (row sums)

torch.mean(tensor)
torch.mean(tensor, dim=0)

torch.prod(tensor, dim=1)       # row-wise product
```

### keepdim

`keepdim=True` preserves the reduced dimension as size 1, useful for broadcasting back.

```python
row_sums = torch.sum(tensor, dim=1, keepdim=True)   # shape (3, 1)
normalized = tensor / row_sums                        # broadcasts correctly
```

### Min, Max, Argmin, Argmax

```python
torch.min(tensor)                         # scalar minimum
torch.max(tensor)                         # scalar maximum

values, indices = torch.min(tensor, dim=0)  # column-wise min + indices
values, indices = torch.max(tensor, dim=1)  # row-wise max + indices

torch.argmin(tensor)                       # flat index of min
torch.argmax(tensor, dim=1)               # row-wise argmax
```

### Variance and Standard Deviation

```python
torch.var(tensor)                          # population variance
torch.std(tensor)                          # population std
torch.var(tensor, dim=0, unbiased=True)    # Bessel's correction
```

### Cumulative Operations

```python
torch.cumsum(tensor, dim=0)    # cumulative sum along rows
torch.cumsum(tensor, dim=1)    # cumulative sum along columns
torch.cumprod(tensor, dim=1)   # cumulative product
```

### Norms

```python
vector = torch.tensor([3.0, 4.0])
torch.norm(vector, p=1)                    # L1 norm: 7.0
torch.norm(vector, p=2)                    # L2 norm: 5.0
torch.norm(vector, p=float('inf'))         # max norm: 4.0

matrix = torch.randn(3, 4)
torch.norm(matrix, p='fro')               # Frobenius norm
torch.norm(matrix, p='nuc')               # nuclear norm
torch.norm(matrix, dim=1)                 # row-wise L2 norm
```

### Multi-Dimension Reduction

```python
tensor_4d = torch.randn(2, 3, 4, 5)
torch.sum(tensor_4d, dim=(1, 3))           # reduce dims 1 and 3
torch.mean(tensor_4d, dim=(0, 2))
```

### Quantile and Median

```python
torch.median(tensor)
torch.quantile(tensor, 0.5)                # median
torch.quantile(tensor, torch.tensor([0.25, 0.75]))
```

### Counting

```python
torch.count_nonzero(tensor)
torch.count_nonzero(tensor, dim=0)
torch.unique(tensor)                        # unique values
torch.unique(tensor, return_counts=True)   # with counts
```

### Differences

```python
seq = torch.tensor([1, 4, 7, 11, 16])
torch.diff(seq)         # [3, 3, 4, 5]
torch.diff(seq, n=2)    # [0, 1, 1]
```

---

## Comparison and Logical Operations

### Element-Wise Comparison

```python
a = torch.tensor([1, 2, 3, 4, 5])
b = torch.tensor([2, 2, 3, 3, 6])

torch.eq(a, b)    # a == b
torch.ne(a, b)    # a != b
torch.gt(a, b)    # a > b
torch.ge(a, b)    # a >= b
torch.lt(a, b)    # a < b
torch.le(a, b)    # a <= b

# Operators work identically
a == b
a > b
```

### Floating-Point Comparison

Direct `==` is unreliable for floats due to precision. Use tolerance-based comparison.

```python
torch.allclose(tensor_a, tensor_b, rtol=1e-5, atol=1e-8)   # whole tensor
torch.isclose(tensor_a, tensor_b, rtol=1e-5, atol=1e-8)     # element-wise
```

### Tensor-Level Equality

```python
torch.equal(tensor_a, tensor_b)    # True only if all elements match exactly
```

### Logical Operations

```python
bool_a = torch.tensor([True, False, True, False])
bool_b = torch.tensor([True, True, False, False])

torch.logical_and(bool_a, bool_b)   # or: bool_a & bool_b
torch.logical_or(bool_a, bool_b)    # or: bool_a | bool_b
torch.logical_xor(bool_a, bool_b)   # or: bool_a ^ bool_b
torch.logical_not(bool_a)           # or: ~bool_a
```

### Chained Conditions

```python
values = torch.arange(1, 11)
between = (values >= 3) & (values <= 7)
outside = (values < 3) | (values > 8)
even_gt5 = ((values % 2) == 0) & (values > 5)
```

### Boolean Reductions

```python
torch.any(bool_tensor)              # True if any element is True
torch.all(bool_tensor)              # True if all elements are True
torch.any(bool_tensor, dim=0)       # along dimension
torch.all(bool_tensor, dim=1)
```

### Conditional Selection

```python
# torch.where: element-wise if-then-else
torch.where(data > 0, data, torch.zeros_like(data))
torch.where(data > 0, data, -1)                      # scalar fallback

# Element-wise max/min
torch.maximum(tensor_x, tensor_y)
torch.minimum(tensor_x, tensor_y)
```

---

## Sorting and Selection

### Sort

```python
values, indices = torch.sort(tensor)
values, indices = torch.sort(tensor, descending=True)
values, indices = torch.sort(tensor_2d, dim=0)

indices = torch.argsort(tensor)
```

### Top-K

```python
values, indices = torch.topk(tensor, k=3)                # largest k
values, indices = torch.topk(tensor, k=3, largest=False)  # smallest k
```

### Unique and Mode

```python
unique_vals = torch.unique(tensor)
unique_vals, counts = torch.unique(tensor, return_counts=True)
unique_vals, inverse = torch.unique(tensor, return_inverse=True)

mode_val, mode_idx = torch.mode(tensor)
```
