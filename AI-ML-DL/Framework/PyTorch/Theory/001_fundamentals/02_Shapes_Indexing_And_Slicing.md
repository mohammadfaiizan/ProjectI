# Tensor Shapes, Indexing, Slicing, and Reshaping

## Table of Contents
1. [Tensor Shape Properties](#tensor-shape-properties)
2. [Basic Indexing](#basic-indexing)
3. [Advanced Indexing](#advanced-indexing)
4. [Slicing Operations](#slicing-operations)
5. [Reshape and View](#reshape-and-view)
6. [Squeeze and Unsqueeze](#squeeze-and-unsqueeze)
7. [Transpose and Permute](#transpose-and-permute)
8. [Flatten, Expand, and Repeat](#flatten-expand-and-repeat)
9. [Contiguity and Memory Layout](#contiguity-and-memory-layout)

---

## Tensor Shape Properties

### Shape, Size, and Dimensions

```python
tensor = torch.randn(2, 3, 4)

tensor.shape          # torch.Size([2, 3, 4])
tensor.size()         # torch.Size([2, 3, 4])  (identical to .shape)
tensor.size(0)        # 2  (size of specific dimension)
tensor.ndim           # 3  (number of dimensions)
tensor.numel()        # 24 (total element count)
```

`shape` is a property, `size()` is a method. Both return `torch.Size`, which is a subclass of `tuple`.

### Strides

Strides indicate the number of elements to skip in memory to move one step along each dimension.

```python
tensor = torch.randn(3, 4)
tensor.stride()       # (4, 1) for row-major layout
# stride[0]=4: skip 4 elements to move to next row
# stride[1]=1: skip 1 element to move to next column
```

---

## Basic Indexing

### Single Element and Row/Column Access

```python
tensor = torch.arange(24).reshape(4, 6)

tensor[2, 3]          # single element at row 2, col 3
tensor[1]             # entire row 1
tensor[:, 2]          # entire column 2
tensor[1:3]           # rows 1 and 2
tensor[:, 1:4]        # columns 1, 2, 3
```

### Negative Indexing

```python
tensor[-1]            # last row
tensor[:, -1]         # last column
tensor[-2:]           # last two rows
```

### Step Slicing

```python
tensor[::2]           # every other row
tensor[:, ::2]        # every other column
tensor[::-1]          # rows in reverse order
tensor[:, ::-1]       # columns in reverse order
```

---

## Advanced Indexing

### Ellipsis Indexing

`...` (ellipsis) replaces any number of `:` dimensions.

```python
tensor_3d = torch.randn(3, 4, 5)

tensor_3d[..., 2]     # same as tensor_3d[:, :, 2]
tensor_3d[1, ...]     # same as tensor_3d[1, :, :]
```

### Boolean (Mask) Indexing

Boolean indexing selects elements where the mask is `True`. The result is always 1D.

```python
data = torch.randn(4, 4)

mask = data > 0
positive_values = data[mask]               # 1D tensor of positive values

complex_mask = (data > -0.5) & (data < 0.5)
data[complex_mask]                          # values in range

# Row-level boolean mask
row_mask = torch.tensor([True, False, True, False])
data[row_mask]                              # rows 0 and 2
```

### Fancy (Integer Tensor) Indexing

Indexing with an integer tensor selects elements at those indices. Unlike slicing, fancy indexing returns a **copy** (not a view).

```python
indices = torch.tensor([0, 2, 1])
tensor[indices]                             # rows in order [0, 2, 1]

# 2D fancy indexing: selects (row_i, col_i) pairs
row_idx = torch.tensor([0, 1, 2])
col_idx = torch.tensor([1, 3, 5])
tensor[row_idx, col_idx]                    # elements at (0,1), (1,3), (2,5)
```

### index_select and masked_select

```python
# Select along a dimension
torch.index_select(tensor, dim=0, index=torch.tensor([0, 2, 3]))

# Masked select (returns 1D)
torch.masked_select(data, data > 0.5)
```

### Gather and Scatter

**gather**: Collects values from a source tensor using an index tensor along a given dimension.

```python
source = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
index = torch.tensor([[0, 1, 2], [2, 0, 1]])
torch.gather(source, dim=0, index=index)
```

**scatter**: Inverse of gather. Distributes values from a source into a target tensor at specified indices.

```python
target = torch.zeros(3, 3)
src = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
index = torch.tensor([[0, 1, 2], [2, 0, 1]])
target.scatter(dim=0, index=index, src=src)
```

### where and nonzero

```python
# Conditional selection
torch.where(data > 0, data, torch.zeros_like(data))

# Indices of non-zero elements
torch.nonzero(sparse_tensor)

# Top-k values and indices
values, indices = torch.topk(tensor_1d, k=3)
```

---

## Slicing Operations

### Multi-Dimensional Slicing

```python
tensor_3d = torch.arange(60).reshape(3, 4, 5)

tensor_3d[0:2, 1:3, 2:4]       # slice across all dims
tensor_3d[::2, ::2, ::2]        # every other element in all dims
tensor_3d[-1, -2:, -3:]         # negative indexing with slicing
```

### None Indexing (New Axis)

`None` inserts a new dimension of size 1, equivalent to `unsqueeze`.

```python
tensor_3d[None, 1, :, 2]        # adds batch dimension at front
tensor_3d[1, :, None, 2, None]  # adds dims at positions 2 and 4
```

### Narrow and Select

```python
# narrow: slice along a dimension (start, length)
torch.narrow(tensor, dim=0, start=1, length=3)

# select: index a single slice along a dimension (removes that dim)
torch.select(tensor, dim=0, index=2)
```

### Stride-Based Views

`as_strided` creates a view with custom strides, useful for sliding windows.

```python
# Sliding window of size 3 over a 1D tensor
data = torch.arange(10)
windows = torch.as_strided(data, size=(6, 3), stride=(1, 1))
```

### Unfold

```python
sequence = torch.arange(10)
# unfold(dimension, window_size, step)
sequence.unfold(0, 3, 1)    # shape (8, 3): 8 windows of size 3
```

### Slicing Is a View

Slicing never copies data. The result shares the same underlying storage as the original tensor.

```python
original = torch.randn(1000, 1000)
view_slice = original[100:200, 200:300]
# view_slice.storage().data_ptr() == original.storage().data_ptr()
```

Use `.clone()` to create an independent copy.

---

## Reshape and View

### reshape

Works on any tensor (contiguous or not). May return a view or a copy.

```python
original = torch.arange(24)
original.reshape(4, 6)         # 2D
original.reshape(2, 3, 4)     # 3D
original.reshape(4, -1)       # -1 inferred as 6
```

### view

Requires a **contiguous** tensor. Always returns a view (shared memory).

```python
tensor = torch.arange(12)
viewed = tensor.view(3, 4)      # shared memory
viewed[0, 0] = 999              # modifies original

# Non-contiguous tensor -> view fails
transposed = torch.randn(3, 4).t()
# transposed.view(-1)  -> RuntimeError
transposed.contiguous().view(-1)  # works after making contiguous
```

**When to use which**: Use `view` when you know the tensor is contiguous and want to guarantee no copy. Use `reshape` for safety when contiguity is uncertain.

---

## Squeeze and Unsqueeze

### squeeze

Removes dimensions of size 1.

```python
tensor = torch.randn(1, 3, 1, 4, 1)

tensor.squeeze()        # shape (3, 4) - removes all size-1 dims
tensor.squeeze(0)       # shape (3, 1, 4, 1) - removes dim 0 only
tensor.squeeze(2)       # shape (1, 3, 4, 1) - removes dim 2 only
```

### unsqueeze

Inserts a dimension of size 1 at the specified position.

```python
tensor = torch.randn(3, 4)

tensor.unsqueeze(0)     # shape (1, 3, 4)
tensor.unsqueeze(1)     # shape (3, 1, 4)
tensor.unsqueeze(-1)    # shape (3, 4, 1)
```

---

## Transpose and Permute

### transpose

Swaps exactly two dimensions. Returns a view (non-contiguous).

```python
matrix = torch.randn(3, 4)
matrix.transpose(0, 1)  # shape (4, 3)
matrix.t()               # shorthand for 2D transpose

tensor_3d = torch.randn(2, 3, 4)
tensor_3d.transpose(0, 2)   # shape (4, 3, 2)
```

### permute

Rearranges all dimensions at once.

```python
tensor = torch.randn(2, 3, 4, 5)
tensor.permute(3, 1, 0, 2)   # shape (5, 3, 2, 4)

# Common patterns
nchw = torch.randn(10, 3, 32, 32)
nhwc = nchw.permute(0, 2, 3, 1)         # NCHW -> NHWC

seq_first = torch.randn(20, 32, 512)     # Seq, Batch, Feat
batch_first = seq_first.permute(1, 0, 2) # Batch, Seq, Feat
```

---

## Flatten, Expand, and Repeat

### flatten

```python
tensor = torch.randn(2, 3, 4, 5)

tensor.flatten()                            # shape (120,)
tensor.flatten(start_dim=1)                 # shape (2, 60)
tensor.flatten(start_dim=1, end_dim=2)      # shape (2, 12, 5)
```

### expand

Broadcasts a tensor to a larger size **without copying data**. Only size-1 dimensions can be expanded.

```python
small = torch.tensor([[1], [2], [3]])       # shape (3, 1)
small.expand(3, 4)                           # shape (3, 4), no copy
small.expand(-1, 4)                          # -1 keeps original size
small.expand_as(torch.randn(3, 6))          # match target shape
```

### repeat

Actually **copies** data to tile the tensor.

```python
small = torch.tensor([[1], [2], [3]])
small.repeat(2, 3)     # shape (6, 3), new memory
```

**expand vs repeat**: `expand` creates a view (no memory cost, strides are 0 for expanded dims). `repeat` allocates new memory. Prefer `expand` when possible.

---

## Contiguity and Memory Layout

### What Is Contiguity

A tensor is **contiguous** if its elements are stored in a single, unbroken block of memory in row-major (C) order. Operations like `transpose` and `permute` change strides without moving data, resulting in non-contiguous tensors.

```python
tensor = torch.randn(3, 4)
tensor.is_contiguous()               # True
tensor.stride()                       # (4, 1)

transposed = tensor.t()
transposed.is_contiguous()           # False
transposed.stride()                   # (1, 4)
```

### Making Contiguous

```python
contiguous_copy = transposed.contiguous()   # allocates new memory
contiguous_copy.is_contiguous()             # True
```

### Why It Matters

- `view` requires contiguous tensors
- Some CUDA kernels are faster on contiguous data
- Non-contiguous tensors have suboptimal cache access patterns

### Views vs Copies

| Operation | Creates |
|-----------|---------|
| Slicing (`tensor[1:3]`) | View |
| `view`, `reshape` (contiguous) | View |
| `transpose`, `permute`, `t()` | View (non-contiguous) |
| `expand` | View |
| Fancy indexing (`tensor[[0, 2]]`) | Copy |
| Boolean indexing (`tensor[mask]`) | Copy |
| `.clone()`, `.contiguous()` | Copy |
| `repeat` | Copy |

### Checking Shared Memory

```python
original = torch.arange(12)
view = original.view(3, 4)
view.data_ptr() == original.data_ptr()           # True (same memory)
view.storage().data_ptr() == original.storage().data_ptr()  # True
```
