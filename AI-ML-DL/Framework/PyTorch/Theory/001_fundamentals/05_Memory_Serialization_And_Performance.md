# Memory Management, Serialization, Advanced Operations, and Performance

## Table of Contents
1. [Memory Management](#memory-management)
2. [In-Place Operations](#in-place-operations)
3. [Storage and Data Pointers](#storage-and-data-pointers)
4. [Tensor Serialization](#tensor-serialization)
5. [Advanced Tensor Operations](#advanced-tensor-operations)
6. [Performance Optimization](#performance-optimization)
7. [Debugging Utilities](#debugging-utilities)

---

## Memory Management

### Tensor Memory Usage

Each tensor's memory footprint is `numel() * element_size()` bytes.

```python
tensor = torch.randn(1000, 1000)
memory_bytes = tensor.numel() * tensor.element_size()
memory_mb = memory_bytes / 1e6
# 1,000,000 * 4 bytes (float32) = 4.0 MB
```

### GPU Memory Tracking

```python
torch.cuda.memory_allocated()      # bytes currently used by tensors
torch.cuda.memory_reserved()       # bytes reserved by caching allocator
torch.cuda.max_memory_allocated()  # peak allocated since start/reset
torch.cuda.memory_stats()          # detailed statistics dict
```

### GPU Memory Deallocation

```python
# Delete tensor reference
del large_tensor

# Force Python garbage collection
import gc
gc.collect()

# Release unused cached memory back to OS
torch.cuda.empty_cache()
```

`empty_cache` releases unused blocks from the caching allocator. It does not free memory still referenced by live tensors.

### Pinned Memory

Page-locked (pinned) memory enables faster, asynchronous CPU-to-GPU transfers.

```python
pinned = torch.randn(1000, 1000).pin_memory()
gpu_tensor = pinned.cuda(non_blocking=True)    # async transfer
```

Use pinned memory when repeatedly transferring data to GPU (e.g., in DataLoader with `pin_memory=True`).

### torch.no_grad Context

Disabling gradient tracking reduces memory by not storing intermediate values for backpropagation.

```python
with torch.no_grad():
    output = model(input_data)    # no computation graph stored
```

### Memory-Efficient Patterns

**Chunked Processing**: Process large tensors in chunks to limit peak memory.

```python
def chunked_operation(tensor, chunk_size=1000):
    results = []
    for i in range(0, tensor.size(0), chunk_size):
        chunk = tensor[i:i+chunk_size]
        results.append(chunk.pow(2).sum())
    return torch.stack(results)
```

**Detach for Inference**: Break the computation graph when gradients are not needed.

```python
features = model.backbone(x).detach()
```

---

## In-Place Operations

In-place operations modify tensors directly without allocating new memory. They are identified by the trailing underscore `_`.

### Available In-Place Methods

```python
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])

tensor.add_(1)          # tensor += 1
tensor.sub_(0.5)        # tensor -= 0.5
tensor.mul_(2)          # tensor *= 2
tensor.div_(3)          # tensor /= 3
tensor.pow_(2)          # tensor **= 2
tensor.sqrt_()          # tensor = sqrt(tensor)
tensor.abs_()
tensor.exp_()
tensor.log_()
tensor.clamp_(min=0)
tensor.zero_()          # fill with zeros
tensor.fill_(5.0)       # fill with value
tensor.uniform_(0, 1)   # fill with uniform random
tensor.normal_(0, 1)    # fill with normal random
```

### In-Place Normalization

```python
def normalize_inplace(tensor):
    mean = tensor.mean()
    std = tensor.std()
    tensor.sub_(mean).div_(std)
    return tensor
```

### Caveats

- In-place operations on tensors that require gradients will raise an error if they are needed for backward pass
- Views of tensors are affected by in-place operations on the original
- In-place ops on expanded tensors raise errors (expanded dimensions have stride 0)

---

## Storage and Data Pointers

### Tensor Storage

Every tensor has an underlying `Storage` object that holds the actual data in a contiguous 1D block. Multiple tensors can share the same storage (views).

```python
a = torch.arange(12)
b = a.view(3, 4)

a.storage().data_ptr() == b.storage().data_ptr()   # True (shared storage)
a.data_ptr()                                         # pointer to first element
b.data_ptr()                                         # may differ (different offset)
```

### Checking Memory Sharing

```python
# Two tensors share memory if they have the same storage
a.storage().data_ptr() == b.storage().data_ptr()

# Or use:
a.data_ptr() == b.data_ptr()    # only True if same element offset too
```

### Storage Utilization

When a view or slice references only a portion of the storage, the full storage remains in memory.

```python
large = torch.randn(10000)
small_view = large[5000:5010]
# small_view uses only 10 elements but keeps all 10000 in memory
# Use .clone() to free the original: small_clone = small_view.clone()
```

---

## Tensor Serialization

### Basic Save and Load

```python
tensor = torch.randn(3, 4, 5)
torch.save(tensor, 'tensor.pt')
loaded = torch.load('tensor.pt')
```

### State Dictionaries

```python
state = {
    'weights': torch.randn(10, 5),
    'bias': torch.randn(10),
    'epoch': 100,
    'loss': 0.1234,
}
torch.save(state, 'checkpoint.pt')
loaded_state = torch.load('checkpoint.pt')
```

### Device-Aware Loading

```python
# Load to CPU regardless of where it was saved
loaded = torch.load('model.pt', map_location='cpu')

# Load to specific GPU
loaded = torch.load('model.pt', map_location='cuda:0')

# Load GPU-saved tensor to CPU
loaded = torch.load('gpu_model.pt', map_location=torch.device('cpu'))
```

### File Formats

| Extension | Notes |
|-----------|-------|
| `.pt` | Standard PyTorch convention |
| `.pth` | Also common for model checkpoints |

Both are identical under the hood (Python pickle + tensor storage).

### Compressed Serialization

```python
import gzip

# Save compressed
with gzip.open('tensor.gz', 'wb') as f:
    torch.save(tensor, f)

# Load compressed
with gzip.open('tensor.gz', 'rb') as f:
    loaded = torch.load(f)
```

### Buffer I/O

```python
import io

buffer = io.BytesIO()
torch.save(tensor, buffer)
buffer.seek(0)
loaded = torch.load(buffer)
```

### Model Checkpoint Pattern

```python
# Save
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### Pickle Protocol

```python
torch.save(tensor, 'tensor.pt', pickle_protocol=4)   # higher = more efficient
```

---

## Advanced Tensor Operations

### Masked Operations

```python
data = torch.randn(4, 5)
mask = data > 0

# Masked select (returns 1D)
torch.masked_select(data, mask)

# Masked fill (in-place)
data.masked_fill_(mask, -999)

# Masked assignment
data[mask] = 0
```

### Flip, Roll, and Rotate

```python
tensor = torch.arange(12).reshape(3, 4)

torch.flip(tensor, dims=[0])             # flip rows
torch.flip(tensor, dims=[1])             # flip columns
torch.flip(tensor, dims=[0, 1])          # flip both

torch.roll(tensor, shifts=1, dims=1)     # circular shift right
torch.roll(tensor, shifts=(1, 1), dims=(0, 1))

torch.rot90(tensor, k=1, dims=[0, 1])    # 90-degree rotation
```

### Interpolation

```python
start = torch.zeros(3, 3)
end = torch.ones(3, 3)

torch.lerp(start, end, 0.5)              # linear interpolation at 50%
```

### Cumulative Operations

```python
data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

torch.cumsum(data, dim=0)     # [1, 3, 6, 10, 15]
torch.cumprod(data, dim=0)    # [1, 2, 6, 24, 120]
torch.cummax(data, dim=0)     # running maximum + indices
torch.cummin(data, dim=0)     # running minimum + indices
```

### Scatter and Gather

```python
# Gather: collect values using index tensor
source = torch.arange(20).float().reshape(4, 5)
index = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0]])
gathered = torch.gather(source, dim=1, index=index)

# Scatter: distribute values
target = torch.zeros(4, 5)
target.scatter_(dim=1, index=index, src=torch.ones(4, 3))

# Scatter add: accumulate
target.scatter_add_(dim=1, index=index, src=torch.ones(4, 3))
```

### Einstein Summation

```python
A = torch.randn(3, 4, 5)
B = torch.randn(5, 6, 7)
C = torch.randn(7, 8)

# Multi-tensor contraction
torch.einsum('ijk,klm,mn->ijln', A, B, C)

# Batch matrix multiply
torch.einsum('bij,bjk->bik', batch_A, batch_B)

# Multi-dimensional trace
torch.einsum('iijj->', torch.randn(3, 3, 4, 4))
```

### Constraints and Projections

```python
# Clamp to range
torch.clamp(x, min=-1, max=1)

# L2 normalize rows to unit vectors
torch.nn.functional.normalize(x, p=2, dim=1)
```

---

## Performance Optimization

### Contiguous Memory

Operations on contiguous tensors are faster due to better CPU cache utilization.

```python
tensor = torch.randn(1000, 1000)
non_contig = tensor.t()
contig = non_contig.contiguous()
# Operations on contig are faster than on non_contig
```

### Vectorization over Python Loops

```python
# Bad: Python loop
result = torch.zeros(10000)
for i in range(10000):
    if data[i] > 0:
        result[i] = data[i] ** 2

# Good: vectorized
mask = data > 0
result = torch.where(mask, data ** 2, torch.tensor(0.0))
```

### Batch Processing

```python
# Bad: process one by one
results = []
for sample in samples:
    results.append(torch.mm(sample, sample.t()))

# Good: batch processing
batch = torch.stack(samples)
results = torch.bmm(batch, batch.transpose(-2, -1))
```

### Broadcasting over Expansion

```python
a = torch.randn(1000, 1)
b = torch.randn(1, 1000)

# Bad: explicit expand
result = a.expand(1000, 1000) + b.expand(1000, 1000)

# Good: let broadcasting handle it
result = a + b
```

### Data Type Selection

| Type | Memory | Speed | Use Case |
|------|--------|-------|----------|
| `float32` | 4 bytes | Baseline | Default training |
| `float16` | 2 bytes | Faster on GPU | Mixed-precision training |
| `bfloat16` | 2 bytes | Faster on GPU | Training (better range than fp16) |

```python
float32_tensor = torch.randn(1000, 1000, dtype=torch.float32)  # 4 MB
float16_tensor = torch.randn(1000, 1000, dtype=torch.float16)  # 2 MB
```

### Efficient Tensor Creation

```python
# Fastest for filling with a value:
tensor = torch.empty(1000, 1000)
tensor.fill_(5.0)

# Slower alternatives:
torch.full((1000, 1000), 5.0)
torch.zeros(1000, 1000) + 5.0
```

### Operation Fusion

Chain operations in a single expression to allow the backend to fuse kernels.

```python
# Separate operations (3 kernel launches)
y = x + 1
y = y * 2
y = torch.relu(y)

# Fused (potentially 1 kernel)
y = torch.relu((x + 1) * 2)
```

### GPU-Specific Tips

- Create tensors directly on GPU instead of creating on CPU then transferring
- Use `non_blocking=True` with pinned memory for async transfers
- Minimize CPU-GPU synchronization points
- Use `torch.cuda.synchronize()` only when timing or debugging

### torch.no_grad for Inference

```python
with torch.no_grad():
    output = model(input)     # faster, less memory
```

### JIT Compilation

```python
@torch.jit.script
def fused_relu_scale(x, scale: float):
    return torch.relu(x) * scale
```

### Anti-Patterns to Avoid

- Converting tensors to Python lists in loops
- Using `.item()` inside loops
- Creating tensors inside tight loops
- Unnecessary `.cpu()` / `.cuda()` round-trips
- Not reusing pre-allocated tensors

---

## Debugging Utilities

### Tensor Inspection

```python
def inspect_tensor(tensor, name="tensor"):
    print(f"Shape: {tensor.shape}")
    print(f"DType: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Requires grad: {tensor.requires_grad}")
    print(f"Contiguous: {tensor.is_contiguous()}")
    print(f"Stride: {tensor.stride()}")
    print(f"Memory: {tensor.numel() * tensor.element_size() / 1024:.2f} KB")
    if tensor.is_floating_point() and tensor.numel() > 0:
        print(f"Min: {tensor.min().item()}")
        print(f"Max: {tensor.max().item()}")
        print(f"Mean: {tensor.mean().item():.6f}")
        print(f"Std: {tensor.std().item():.6f}")
        print(f"Has NaN: {torch.isnan(tensor).any().item()}")
        print(f"Has Inf: {torch.isinf(tensor).any().item()}")
```

### NaN and Inf Detection

```python
tensor = torch.tensor([1.0, float('nan'), 3.0, float('inf')])

torch.isnan(tensor)          # [False, True, False, False]
torch.isinf(tensor)          # [False, False, False, True]
torch.isfinite(tensor)       # [True, False, True, False]

# Locate problematic values
nan_indices = torch.nonzero(torch.isnan(tensor))
inf_indices = torch.nonzero(torch.isinf(tensor))
```

### Shape Compatibility Checking

```python
def check_broadcast(a, b):
    try:
        shape = torch.broadcast_shapes(a.shape, b.shape)
        return True, shape
    except RuntimeError:
        return False, None

def check_matmul(a, b):
    return a.shape[-1] == b.shape[-2] if a.ndim >= 2 and b.ndim >= 2 else False
```

### Gradient Debugging

```python
# Enable anomaly detection for gradient errors
torch.autograd.set_detect_anomaly(True)

# Check gradient info
x = torch.randn(3, 4, requires_grad=True)
y = x.pow(2).sum()
y.backward()

print(x.grad)           # gradient tensor
print(x.grad_fn)        # None (leaf tensor)
print(y.grad_fn)        # SumBackward0
```

### Print Options

```python
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
```

### Memory Layout Debugging

```python
a = torch.arange(12).reshape(3, 4)
b = a.t()

print(f"a contiguous: {a.is_contiguous()}, stride: {a.stride()}")
print(f"b contiguous: {b.is_contiguous()}, stride: {b.stride()}")
print(f"Same storage: {a.storage().data_ptr() == b.storage().data_ptr()}")
```

### Performance Profiling

```python
import time

def profile_op(fn, *args, n_runs=100, name="op"):
    for _ in range(10):    # warm up
        fn(*args)
    start = time.time()
    for _ in range(n_runs):
        fn(*args)
    elapsed = (time.time() - start) / n_runs
    print(f"{name}: {elapsed*1000:.4f} ms/op")
```
