# PyTorch Setup, Tensor Creation, and Type System

## Table of Contents
1. [Introduction to PyTorch](#introduction-to-pytorch)
2. [Installation and Environment](#installation-and-environment)
3. [Tensor Creation Methods](#tensor-creation-methods)
4. [DType System](#dtype-system)
5. [Type Casting](#type-casting)
6. [Device Management](#device-management)
7. [NumPy Interoperability](#numpy-interoperability)

---

## Introduction to PyTorch

PyTorch is an open-source deep learning framework developed by Meta AI. It provides a flexible, Pythonic interface for building and training neural networks, with strong GPU acceleration support.

### Why PyTorch

**Dynamic Computation Graphs**: PyTorch uses eager execution with dynamic computation graphs (define-by-run). Operations execute immediately as Python code runs, making debugging straightforward with standard Python tools like `pdb` and `print`.

**Tensor-Based Computation**: The core abstraction is the `torch.Tensor`, a multi-dimensional array similar to NumPy's `ndarray` but with GPU support and automatic differentiation. All PyTorch operations are expressed as tensor transformations.

**Autograd Engine**: PyTorch's autograd system automatically computes gradients through any computation graph. Every tensor operation is recorded, enabling backpropagation through arbitrary Python code including loops and conditionals.

**GPU Acceleration**: Tensors can be moved to CUDA-capable GPUs with a single method call. PyTorch manages memory allocation, kernel dispatch, and synchronization transparently.

---

## Installation and Environment

### Installation Commands

```python
# CPU-only
pip install torch torchvision torchaudio

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Environment Verification

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
```

### Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `CUDA_HOME` | Path to CUDA toolkit installation |
| `CUDA_VISIBLE_DEVICES` | Controls which GPUs are visible to PyTorch |
| `TORCH_HOME` | Directory for cached models and data |

---

## Tensor Creation Methods

### From Python Data

```python
# From list
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])

# From NumPy (shared memory)
import numpy as np
np_array = np.array([1.0, 2.0, 3.0])
tensor_from_numpy = torch.from_numpy(np_array)

# From another tensor
tensor_copy = torch.tensor(tensor_1d)   # new memory
tensor_clone = tensor_1d.clone()         # new memory, preserves grad
```

**torch.tensor vs torch.from_numpy**: `torch.tensor` always copies data. `torch.from_numpy` shares memory with the NumPy array; modifying one modifies the other.

### Zeros, Ones, and Fill

```python
torch.zeros(3, 4)              # 3x4 tensor of zeros
torch.ones(2, 3)               # 2x3 tensor of ones
torch.full((3, 4), 7.5)        # 3x4 tensor filled with 7.5
torch.empty(2, 3)              # 2x3 uninitialized tensor

# Match shape of existing tensor
torch.zeros_like(existing)
torch.ones_like(existing)
torch.full_like(existing, 3.14)
```

### Sequences

```python
torch.arange(0, 10, 2)         # [0, 2, 4, 6, 8]
torch.linspace(0, 10, 5)       # [0.0, 2.5, 5.0, 7.5, 10.0]
torch.logspace(0, 3, 4)        # [1, 10, 100, 1000] (base 10)
```

### Identity and Diagonal

```python
torch.eye(3)                   # 3x3 identity matrix
torch.eye(3, 5)                # 3x5 identity-like matrix
torch.diag(torch.tensor([1, 2, 3]))  # diagonal matrix from vector
```

### Random Tensors

```python
torch.rand(2, 3)               # uniform [0, 1)
torch.randn(2, 3)              # standard normal (0, 1)
torch.randint(0, 10, (3, 4))   # integers in [0, 10)
torch.randperm(10)             # random permutation of 0..9
```

### Grid Generation

```python
x = torch.arange(3)
y = torch.arange(4)
grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
# grid_x.shape == grid_y.shape == (3, 4)
```

---

## DType System

### Available Types

| Category | Types | Element Size |
|----------|-------|-------------|
| Boolean | `torch.bool` | 1 byte |
| Integer (unsigned) | `torch.uint8` | 1 byte |
| Integer (signed) | `torch.int8`, `torch.int16`, `torch.int32`, `torch.int64` | 1, 2, 4, 8 bytes |
| Floating point | `torch.float16`, `torch.bfloat16`, `torch.float32`, `torch.float64` | 2, 2, 4, 8 bytes |
| Complex | `torch.complex64`, `torch.complex128` | 8, 16 bytes |

### Type Aliases

| Alias | Full Name |
|-------|-----------|
| `torch.half` | `torch.float16` |
| `torch.float` | `torch.float32` |
| `torch.double` | `torch.float64` |
| `torch.short` | `torch.int16` |
| `torch.int` | `torch.int32` |
| `torch.long` | `torch.int64` |

### Creating with Specific Types

```python
torch.tensor([1, 2, 3], dtype=torch.int8)
torch.tensor([1.0, 2.0], dtype=torch.float16)
torch.tensor([True, False], dtype=torch.bool)
torch.tensor([1+2j, 3+4j], dtype=torch.complex64)
```

### Type Information

```python
torch.finfo(torch.float32)      # precision, min, max for floats
torch.iinfo(torch.int32)        # min, max for integers
tensor.element_size()            # bytes per element
tensor.is_floating_point()
tensor.is_complex()
```

---

## Type Casting

### Casting Methods

```python
base = torch.tensor([1.7, 2.3, 3.9])

# Method 1: .to()
base.to(torch.int32)
base.to(torch.long)

# Method 2: shorthand methods
base.float()    # float32
base.double()   # float64
base.int()      # int32
base.long()     # int64
base.half()     # float16
base.bool()     # bool

# Method 3: match another tensor's type
reference = torch.tensor([1], dtype=torch.int8)
base.type_as(reference)
```

### Automatic Type Promotion

PyTorch promotes types automatically in mixed-type operations. The result uses the higher-precision type.

```python
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
float_tensor = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
result = int_tensor + float_tensor   # result is float32
```

### Rounding Before Casting

```python
values = torch.tensor([1.2, 2.8, 3.1, 4.9])
values.round().int()     # [1, 3, 3, 5]
values.floor().int()     # [1, 2, 3, 4]
values.ceil().int()      # [2, 3, 4, 5]
values.trunc().int()     # [1, 2, 3, 4]
```

### Special Value Handling

```python
special = torch.tensor([float('inf'), float('-inf'), float('nan')])
special.to(torch.int32)                    # inf -> max int, nan -> 0
torch.tensor([0, 1, 2, -1]).bool()        # [False, True, True, True]
```

---

## Device Management

### Device Detection

```python
torch.cuda.is_available()         # CUDA GPU
torch.backends.mps.is_available() # Apple Silicon
torch.cuda.device_count()         # number of GPUs
torch.cuda.get_device_name(0)     # GPU name
```

### Device Objects

```python
cpu = torch.device('cpu')
gpu = torch.device('cuda')        # default CUDA device
gpu0 = torch.device('cuda:0')     # specific GPU
```

### Tensor Placement

```python
# Create on specific device
tensor_cpu = torch.randn(3, 4)
tensor_gpu = torch.randn(3, 4, device='cuda')

# Move between devices
tensor_gpu = tensor_cpu.to('cuda')
tensor_gpu = tensor_cpu.cuda()
tensor_cpu = tensor_gpu.to('cpu')
tensor_cpu = tensor_gpu.cpu()
```

### Cross-Device Rules

Tensors must be on the same device to participate in an operation. Attempting to operate on tensors from different devices raises `RuntimeError`.

```python
cpu_t = torch.randn(3, 3)
gpu_t = torch.randn(3, 3, device='cuda')
# cpu_t + gpu_t  -> RuntimeError
result = cpu_t.cuda() + gpu_t     # move to same device first
```

### GPU Memory Management

```python
torch.cuda.memory_allocated()      # bytes currently allocated
torch.cuda.memory_reserved()       # bytes reserved by caching allocator
torch.cuda.empty_cache()           # release unused cached memory
```

### Pinned Memory

Pinned (page-locked) memory enables faster CPU-to-GPU transfers with non-blocking copies.

```python
pinned = torch.randn(1000, 1000).pin_memory()
gpu_tensor = pinned.cuda(non_blocking=True)
```

---

## NumPy Interoperability

### Conversion

```python
# PyTorch -> NumPy (shared memory on CPU)
tensor = torch.tensor([1.0, 2.0, 3.0])
np_array = tensor.numpy()

# NumPy -> PyTorch (shared memory)
np_array = np.array([4.0, 5.0, 6.0])
tensor = torch.from_numpy(np_array)
```

### Shared Memory Behavior

`torch.from_numpy` and `.numpy()` share underlying memory. In-place modifications on one affect the other.

```python
tensor = torch.tensor([1.0, 2.0, 3.0])
np_arr = tensor.numpy()
tensor.add_(1)
print(np_arr)  # [2.0, 3.0, 4.0] - also changed
```

### GPU Tensors

GPU tensors cannot be directly converted to NumPy. Move to CPU first.

```python
gpu_tensor = torch.randn(3, device='cuda')
np_array = gpu_tensor.cpu().numpy()
```
