# Setup, Creation, and Types

## Table of Contents

1. [TensorFlow Installation and Environment Setup](#1-tensorflow-installation-and-environment-setup)
2. [Tensor Creation Methods](#2-tensor-creation-methods)
3. [Dtype System](#3-dtype-system)
4. [Type Conversion and Promotion](#4-type-conversion-and-promotion)
5. [Eager Execution vs Graph Mode](#5-eager-execution-vs-graph-mode)
6. [tf.function Introduction](#6-tffunction-introduction)
7. [tf.Variable vs tf.constant](#7-tfvariable-vs-tfconstant)

---

## 1. TensorFlow Installation and Environment Setup

TensorFlow can be installed via pip, conda, or Docker. The recommended approach for most users is pip with a virtual environment to isolate dependencies.

### pip Installation

```python
# CPU-only (smaller footprint)
pip install tensorflow

# GPU support (requires CUDA and cuDNN)
pip install tensorflow[and-cuda]
```

### Virtual Environment Best Practices

Create an isolated environment to avoid version conflicts:

```python
# Using venv
python -m venv tf_env
tf_env\Scripts\activate  # Windows
source tf_env/bin/activate  # Linux/Mac

# Using conda
conda create -n tf_env python=3.10
conda activate tf_env
conda install tensorflow
```

### Verifying Installation

```python
import tensorflow as tf
print(tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))
```

### Environment Variables

Key environment variables for TensorFlow configuration:

| Variable | Purpose |
|----------|---------|
| `TF_CPP_MIN_LOG_LEVEL` | Controls logging (0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR) |
| `CUDA_VISIBLE_DEVICES` | Restrict GPU visibility (e.g., `"0,1"` or `""` for CPU only) |
| `TF_FORCE_GPU_ALLOW_GROWTH` | Prevent TensorFlow from allocating all GPU memory upfront |

---

## 2. Tensor Creation Methods

TensorFlow provides several functions for creating tensors with different initialization patterns.

### tf.constant

Creates an immutable tensor from Python values or NumPy arrays.

```python
import tensorflow as tf

# Scalar
s = tf.constant(42)

# Vector
v = tf.constant([1.0, 2.0, 3.0])

# Matrix
m = tf.constant([[1, 2], [3, 4]])

# From NumPy
import numpy as np
arr = np.array([[1, 2], [3, 4]])
t = tf.constant(arr)
```

### tf.zeros and tf.ones

Create tensors filled with zeros or ones. Essential for weight initialization and placeholder shapes.

```python
# Zero tensor of shape (3, 4)
z = tf.zeros([3, 4])

# Ones with same shape as another tensor
ref = tf.constant([[1, 2], [3, 4]])
o = tf.ones_like(ref)

# With specific dtype
z_float = tf.zeros([2, 3], dtype=tf.float32)
```

### tf.fill

Creates a tensor of given shape filled with a scalar value.

```python
# 3x3 matrix filled with 7
f = tf.fill([3, 3], 7)

# 1D tensor of 100 elements, all -1
f2 = tf.fill([100], -1)
```

### tf.range

Creates a sequence of numbers, similar to Python's `range`.

```python
# Default: start=0, limit, delta=1
r1 = tf.range(10)           # [0, 1, 2, ..., 9]

# With start and limit
r2 = tf.range(2, 10)        # [2, 3, ..., 9]

# With delta
r3 = tf.range(0, 10, 2)     # [0, 2, 4, 6, 8]

# Float range
r4 = tf.range(0.0, 1.0, 0.1)
```

### tf.linspace

Creates evenly spaced values over an interval. Unlike `tf.range`, the number of values is specified.

```python
# 5 values from 0 to 1 (inclusive)
l = tf.linspace(0.0, 1.0, 5)   # [0.0, 0.25, 0.5, 0.75, 1.0]

# For grid creation
x = tf.linspace(-2.0, 2.0, 100)
```

### tf.eye

Creates an identity matrix (or batch of identity matrices).

```python
# 3x3 identity
i = tf.eye(3)

# Rectangular (3x4)
i2 = tf.eye(3, 4)

# Batch of identity matrices
i3 = tf.eye(3, batch_shape=[2])  # Shape: (2, 3, 3)
```

### Summary Table

| Function | Purpose |
|----------|---------|
| `tf.constant` | From explicit values or NumPy |
| `tf.zeros` | All zeros |
| `tf.ones` | All ones |
| `tf.fill` | Constant scalar value |
| `tf.range` | Integer/float sequence (start, limit, delta) |
| `tf.linspace` | N evenly spaced values |
| `tf.eye` | Identity matrix |

---

## 3. Dtype System

TensorFlow tensors have a **dtype** (data type) that determines precision and memory usage.

### Common dtypes

| dtype | Description | Typical Use |
|-------|-------------|-------------|
| `tf.float32` | 32-bit float | Default for most training |
| `tf.float64` | 64-bit float | High-precision numerics |
| `tf.int32` | 32-bit integer | Indices, counts |
| `tf.int64` | 64-bit integer | Large indices |
| `tf.bool` | Boolean | Masks, conditions |
| `tf.string` | Variable-length string | Text, tokens |
| `tf.complex64` | 64-bit complex | FFT, signal processing |
| `tf.complex128` | 128-bit complex | High-precision complex |

### Specifying dtype

```python
# At creation
t = tf.constant([1, 2, 3], dtype=tf.float32)
t2 = tf.zeros([2, 2], dtype=tf.int32)

# Inspecting dtype
print(t.dtype)
```

### Default dtypes

When dtype is not specified, TensorFlow infers from Python types: integers become `tf.int32`, floats become `tf.float32`.

```python
a = tf.constant(1)      # tf.int32
b = tf.constant(1.0)    # tf.float32
c = tf.constant(1+0j)   # tf.complex128
```

---

## 4. Type Conversion and Promotion

### tf.cast

Converts a tensor to a different dtype. Essential when mixing types in operations.

```python
t = tf.constant([1, 2, 3], dtype=tf.int32)
t_float = tf.cast(t, tf.float32)

# Bool to float (for loss masking)
mask = tf.constant([True, False, True])
mask_float = tf.cast(mask, tf.float32)  # [1.0, 0.0, 1.0]

# Float to int (truncation)
x = tf.constant([1.7, 2.3, 3.9])
x_int = tf.cast(x, tf.int32)  # [1, 2, 3]
```

### Type Promotion

When operating on tensors of different dtypes, TensorFlow promotes to the "wider" type to avoid precision loss.

```python
a = tf.constant(1, dtype=tf.int32)
b = tf.constant(1.0, dtype=tf.float32)
c = a + b  # Result is tf.float32
```

---

## 5. Eager Execution vs Graph Mode

### Eager Execution (Default)

In **eager execution**, operations execute immediately as they are called. Results are concrete tensors; debugging is straightforward.

```python
# Eager: immediate execution
x = tf.constant([[1, 2], [3, 4]])
y = tf.matmul(x, x)
print(y.numpy())  # Direct access to values
```

### Graph Mode

In **graph mode**, operations build a computational graph. Execution is deferred until the graph is run. Benefits include optimization, portability, and distributed training.

```python
# Graph mode via tf.function
@tf.function
def compute(x):
    return tf.matmul(x, x)

result = compute(tf.constant([[1., 2.], [3., 4.]]))
```

### When to Use Each

| Mode | Use Case |
|------|----------|
| Eager | Debugging, prototyping, small experiments |
| Graph | Production, performance, export (SavedModel) |

---

## 6. tf.function Introduction

**tf.function** traces a Python function and compiles it into a TensorFlow graph. It bridges eager and graph execution.

### Basic Usage

```python
@tf.function
def add_and_multiply(a, b):
    s = a + b
    p = a * b
    return s, p

x = tf.constant(3.0)
y = tf.constant(4.0)
sum_val, prod_val = add_and_multiply(x, y)
```

### Tracing and Retracing

The function is traced on first call. Retracing occurs when input signatures change (shape, dtype).

```python
@tf.function
def process(x):
    return tf.reduce_sum(x)

# First call: traces for shape (3,)
process(tf.constant([1., 2., 3.]))

# Same shape: reuses graph
process(tf.constant([4., 5., 6.]))

# Different shape: retraces
process(tf.constant([1., 2.]))
```

### input_signature for Stability

Specifying `input_signature` can reduce retracing and enforce expected inputs.

```python
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 10], dtype=tf.float32)])
def model(x):
    return tf.reduce_mean(x, axis=1)
```

---

## 7. tf.Variable vs tf.constant

### tf.constant

**tf.constant** creates immutable tensors. Values cannot be changed after creation.

```python
c = tf.constant([1, 2, 3])
# c.assign([4, 5, 6])  # AttributeError: constant has no assign
```

### tf.Variable

**tf.Variable** holds mutable state. Used for model weights, optimizer state, and trainable parameters.

```python
v = tf.Variable([1.0, 2.0, 3.0])
print(v.read_value())
```

### assign

Updates the variable to a new value. Must match shape and dtype.

```python
v = tf.Variable([1.0, 2.0, 3.0])
v.assign([4.0, 5.0, 6.0])
# v is now [4.0, 5.0, 6.0]
```

### assign_add and assign_sub

In-place addition or subtraction. Useful for accumulators and counters.

```python
v = tf.Variable(0.0)
v.assign_add(1.0)   # v becomes 1.0
v.assign_add(2.0)   # v becomes 3.0

v2 = tf.Variable([10.0, 20.0])
v2.assign_add([1.0, 2.0])  # v2 becomes [11.0, 22.0]
```

### read_value

Returns the current value as a tensor. Useful when you need a snapshot inside a `tf.function` to avoid graph capture issues.

```python
v = tf.Variable(1.0)

@tf.function
def use_var():
    return v.read_value() + 1.0
```

### Comparison Table

| Feature | tf.constant | tf.Variable |
|---------|-------------|-------------|
| Mutability | Immutable | Mutable |
| assign | No | Yes |
| assign_add/assign_sub | No | Yes |
| Typical use | Fixed data | Weights, state |
| Gradient tracking | No | Yes (trainable) |

### trainable Parameter

Variables can be marked non-trainable to exclude them from gradient updates.

```python
# Trainable (default)
w = tf.Variable([1.0, 2.0], trainable=True)

# Non-trainable (e.g., running statistics)
running_mean = tf.Variable([0.0], trainable=False)
```
