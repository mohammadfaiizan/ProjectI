# Performance, I/O, Interoperability, and Advanced NumPy Topics

## Table of Contents
1. [Memory and Performance](#memory-and-performance)
2. [File I/O](#file-io)
3. [Structured and Record Arrays](#structured-and-record-arrays)
4. [Special Array Types](#special-array-types)
5. [Interoperability](#interoperability)
6. [Common Pitfalls and Debugging](#common-pitfalls-and-debugging)

---

## Memory and Performance

Understanding NumPy's memory layout and performance characteristics is crucial for writing efficient code.

### Memory Layout Deep Dive

#### C-Contiguous vs Fortran-Contiguous

```python
import numpy as np

# C-contiguous (row-major): elements stored row by row
arr_c = np.array([[1, 2, 3], [4, 5, 6]], order='C')
print(arr_c.flags['C_CONTIGUOUS'])  # True
print(arr_c.strides)  # (24, 8) - bytes to next row, bytes to next column

# Fortran-contiguous (column-major): elements stored column by column
arr_f = np.array([[1, 2, 3], [4, 5, 6]], order='F')
print(arr_f.flags['F_CONTIGUOUS'])  # True
print(arr_f.strides)  # (8, 16) - bytes to next column, bytes to next row

# Check memory layout
print(arr_c.flags)
# C_CONTIGUOUS : True
# F_CONTIGUOUS : False
# OWNDATA : True
# WRITEABLE : True
# ALIGNED : True
```

**Performance Implications:**
- C-contiguous arrays are faster for row-wise operations
- Fortran-contiguous arrays are faster for column-wise operations
- Most NumPy operations expect C-contiguous arrays

#### Converting Between Layouts

```python
# Convert to C-contiguous
arr = np.array([[1, 2, 3], [4, 5, 6]], order='F')
arr_c = np.ascontiguousarray(arr)  # Creates copy if needed

# Convert to Fortran-contiguous
arr_f = np.asfortranarray(arr)  # Creates copy if needed

# Check if conversion created a copy
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr_c = np.ascontiguousarray(arr)
print(arr_c is arr)  # True if no copy needed, False if copy created
```

### Strides Explained

Strides define how to move through memory to access array elements:

```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(arr.shape)    # (3, 4)
print(arr.strides)  # (32, 8) - bytes to next row, bytes to next column

# For 3D array
arr_3d = np.random.rand(2, 3, 4)
print(arr_3d.strides)  # (96, 32, 8) - bytes to next along each axis

# Stride calculation:
# stride[i] = product of sizes of all dimensions after i * itemsize
# For shape (2, 3, 4) with float64 (8 bytes):
# stride[0] = 3 * 4 * 8 = 96
# stride[1] = 4 * 8 = 32
# stride[2] = 8
```

**Understanding Strides:**
- Strides map logical indices to memory offsets
- Offset = sum(index[i] * stride[i] for i in range(ndim))
- Non-contiguous arrays have non-standard strides

### Stride Tricks for Advanced Operations

#### Sliding Window View

```python
from numpy.lib.stride_tricks import sliding_window_view

# Create sliding windows
arr = np.array([1, 2, 3, 4, 5, 6])
windows = sliding_window_view(arr, window_shape=3)
# Result: [[1, 2, 3],
#          [2, 3, 4],
#          [3, 4, 5],
#          [4, 5, 6]]
# This is a view, not a copy!

# 2D sliding window
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
windows_2d = sliding_window_view(arr_2d, window_shape=(2, 2))
# Creates 2x2 sliding windows
```

#### Advanced Stride Manipulation

```python
from numpy.lib.stride_tricks import as_strided

# Create view with custom strides (dangerous but powerful)
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# Create view that reads every other element
stride_view = as_strided(arr, shape=(4,), strides=(16,))
# Result: [1, 3, 5, 7] - view, not copy!

# Warning: as_strided can access invalid memory if misused
# Always ensure: offset + shape[i] * stride[i] <= array size
```

**Use Cases:**
- Efficient sliding window operations
- Custom memory layouts
- Zero-copy transformations

### Views vs Copies Performance

```python
# View: shares memory, O(1) operation
arr = np.array([1, 2, 3, 4, 5])
view = arr[1:4]  # View, no copy
view[0] = 99
print(arr)  # [1, 99, 3, 4, 5] - original modified!

# Copy: new memory allocation, O(n) operation
arr = np.array([1, 2, 3, 4, 5])
copy = arr[1:4].copy()  # Explicit copy
copy[0] = 99
print(arr)  # [1, 2, 3, 4, 5] - original unchanged

# When views are created vs copies
arr = np.array([[1, 2, 3], [4, 5, 6]])
view = arr[0, :]  # View (1D slice of 2D)
view = arr[:, 0]  # View (1D slice of 2D)
view = arr[::2, :]  # View (strided slice)
copy = arr[[0, 2], :]  # Copy (fancy indexing)
```

**Performance Tips:**
- Prefer views over copies when possible
- Use `.copy()` explicitly when you need a copy
- Fancy indexing always creates copies

### In-Place Operations

```python
# In-place operations: modify array without creating new one
arr = np.array([1, 2, 3, 4, 5])

# In-place addition
arr += 10  # Modifies arr directly
np.add(arr, 10, out=arr)  # Explicit in-place

# In-place multiplication
arr *= 2
np.multiply(arr, 2, out=arr)

# In-place operations are faster and use less memory
```

**When to Use:**
- Large arrays where memory is a concern
- Loops where temporary arrays would accumulate
- Operations that don't need to preserve original

### Pre-Allocation Patterns

```python
# Pre-allocate arrays instead of appending
# Bad: grows array repeatedly
result = np.array([])
for i in range(1000):
    result = np.append(result, i**2)  # Creates new array each time!

# Good: pre-allocate
result = np.zeros(1000)
for i in range(1000):
    result[i] = i**2

# Even better: vectorize
result = np.arange(1000)**2
```

### Vectorization Patterns

#### Replacing Python Loops

```python
# Bad: Python loop
arr = np.random.rand(1000000)
result = np.zeros_like(arr)
for i in range(len(arr)):
    result[i] = arr[i] * 2 + 1  # Slow!

# Good: Vectorized
result = arr * 2 + 1  # Fast!

# Complex operations
result = np.sin(arr) * np.cos(arr) + np.exp(arr * 0.1)
```

#### Conditional Operations

```python
# Vectorized if-else with np.where
arr = np.array([1, 5, 3, 8, 2])
result = np.where(arr > 4, arr * 2, arr)  # Double if > 4, else keep

# Multiple conditions with np.select
conditions = [arr < 2, (arr >= 2) & (arr < 5), arr >= 5]
choices = [arr * 10, arr * 2, arr]
result = np.select(conditions, choices)

# Piecewise functions
def piecewise_func(x):
    return np.piecewise(x, 
                       [x < 0, (x >= 0) & (x < 5), x >= 5],
                       [lambda x: x**2, lambda x: 2*x, lambda x: x + 10])
```

#### np.vectorize (Convenience, Not Performance)

```python
# np.vectorize is for convenience, NOT performance
def my_func(x):
    return x**2 + 2*x + 1 if x > 0 else 0

# Vectorized version (still uses Python loop internally!)
vec_func = np.vectorize(my_func)
result = vec_func(arr)  # Slower than true vectorization

# Better: use np.where for true vectorization
result = np.where(arr > 0, arr**2 + 2*arr + 1, 0)
```

### Memory-Mapped Files

For arrays too large to fit in memory:

```python
# Create memory-mapped array
arr_mmap = np.memmap('large_array.dat', dtype='float64', mode='w+', shape=(1000000, 1000))
arr_mmap[:] = np.random.rand(1000000, 1000)  # Write to disk
del arr_mmap  # Flush to disk

# Read memory-mapped array
arr_read = np.memmap('large_array.dat', dtype='float64', mode='r', shape=(1000000, 1000))
# Access like normal array, but data stays on disk
result = arr_read[0:100, 0:100]  # Only loads accessed portion
```

**Use Cases:**
- Arrays larger than RAM
- Shared memory between processes
- Persistent storage of large arrays

### Chunked Processing

```python
# Process large arrays in chunks
def process_in_chunks(arr, chunk_size=10000):
    result = []
    for i in range(0, len(arr), chunk_size):
        chunk = arr[i:i+chunk_size]
        processed = chunk * 2 + 1  # Process chunk
        result.append(processed)
    return np.concatenate(result)

# For memory-mapped arrays
arr_mmap = np.memmap('large_array.dat', dtype='float64', mode='r', shape=(1000000,))
chunk_size = 100000
for i in range(0, len(arr_mmap), chunk_size):
    chunk = arr_mmap[i:i+chunk_size]
    process_chunk(chunk)  # Process without loading entire array
```

---

## File I/O

NumPy provides efficient methods for saving and loading arrays.

### Binary Formats

#### Save and Load

```python
# Save single array
arr = np.array([1, 2, 3, 4, 5])
np.save('array.npy', arr)

# Load single array
arr_loaded = np.load('array.npy')

# Save multiple arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
np.savez('arrays.npz', arr1=arr1, arr2=arr2)

# Load multiple arrays
data = np.load('arrays.npz')
arr1_loaded = data['arr1']
arr2_loaded = data['arr2']

# Compressed save (smaller file size)
np.savez_compressed('arrays_compressed.npz', arr1=arr1, arr2=arr2)
```

**Advantages:**
- Fast I/O
- Preserves dtype and shape exactly
- Cross-platform compatible
- Efficient for NumPy arrays

### Text Formats

#### savetxt and loadtxt

```python
# Save to text file
arr = np.array([[1, 2, 3], [4, 5, 6]])
np.savetxt('array.txt', arr, delimiter=',', fmt='%.2f')

# Load from text file
arr_loaded = np.loadtxt('array.txt', delimiter=',')

# With headers and custom formatting
np.savetxt('array.txt', arr, 
           header='Column1,Column2,Column3',
           delimiter=',',
           fmt='%d',
           comments='#')
```

#### genfromtxt (Advanced)

```python
# Load with missing value handling
data_str = """1,2,3
4,,6
7,8,9"""

with open('data.txt', 'w') as f:
    f.write(data_str)

# Load with missing values
arr = np.genfromtxt('data.txt', delimiter=',', missing_values='', filling_values=np.nan)

# With converters
def convert_func(x):
    return float(x.strip('$').replace(',', ''))

arr = np.genfromtxt('prices.txt', delimiter=',', converters={1: convert_func})

# Skip headers and specific rows
arr = np.genfromtxt('data.txt', delimiter=',', skip_header=1, skip_footer=1)
```

### Raw Binary I/O

```python
# Write raw binary
arr = np.array([1, 2, 3, 4, 5], dtype='float64')
arr.tofile('array.bin')

# Read raw binary (must know dtype and shape!)
arr_loaded = np.fromfile('array.bin', dtype='float64')

# For multi-dimensional arrays, shape is lost
arr_2d = np.array([[1, 2], [3, 4]], dtype='float64')
arr_2d.tofile('array_2d.bin')
# Must reshape when loading
arr_2d_loaded = np.fromfile('array_2d.bin', dtype='float64').reshape(2, 2)
```

### String Parsing

```python
# Parse from string
s = "1 2 3 4 5"
arr = np.fromstring(s, dtype=int, sep=' ')

# Note: fromstring is deprecated, use frombuffer or fromstring with sep
arr = np.fromstring(s, dtype=int, sep=' ')

# Character array operations
arr_char = np.char.array(['hello', 'world', 'numpy'])
upper = np.char.upper(arr_char)
lower = np.char.lower(arr_char)
split = np.char.split(arr_char, sep='l')
```

---

## Structured and Record Arrays

Structured arrays allow arrays with heterogeneous data types, similar to tables or DataFrames.

### Creating Structured Dtypes

```python
# Define structured dtype
dtype = [('name', 'U10'), ('age', 'i4'), ('weight', 'f4')]
arr = np.array([('Alice', 25, 55.5), ('Bob', 30, 70.2)], dtype=dtype)

# Access fields by name
names = arr['name']
ages = arr['age']
weights = arr['weight']

# Access individual records
person = arr[0]
print(person['name'])  # 'Alice'
```

### Nested Structured Arrays

```python
# Nested structures
dtype_nested = [('name', 'U10'), 
                ('address', [('street', 'U20'), ('city', 'U10'), ('zip', 'i4')]),
                ('age', 'i4')]

arr = np.array([('Alice', ('123 Main St', 'Boston', 02115), 25),
                ('Bob', ('456 Oak Ave', 'NYC', 10001), 30)],
               dtype=dtype_nested)

# Access nested fields
streets = arr['address']['street']
cities = arr['address']['city']
```

### Record Arrays (np.recarray)

```python
# Record arrays allow attribute-style access
arr = np.array([('Alice', 25, 55.5), ('Bob', 30, 70.2)],
               dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
rec_arr = arr.view(np.recarray)

# Attribute access
names = rec_arr.name  # Instead of rec_arr['name']
ages = rec_arr.age
```

**Use Cases:**
- Tabular data with mixed types
- Database-like operations
- Interfacing with structured binary formats

---

## Special Array Types

### Masked Arrays

Masked arrays allow handling missing or invalid data:

```python
import numpy.ma as ma

# Create masked array
data = np.array([1, 2, 3, 4, 5])
mask = np.array([False, False, True, False, False])  # Mask third element
masked_arr = ma.masked_array(data, mask=mask)

# Or create mask conditionally
data = np.array([1, 2, -999, 4, 5])  # -999 represents missing
masked_arr = ma.masked_array(data, mask=(data == -999))

# Operations ignore masked values
mean_val = masked_arr.mean()  # Computes mean ignoring masked values
sum_val = masked_arr.sum()

# Access underlying data and mask
print(masked_arr.data)  # Original data
print(masked_arr.mask)  # Boolean mask

# Fill masked values
filled = masked_arr.filled(0)  # Replace masked with 0
```

**Operations:**
- Arithmetic operations propagate masks
- Statistical functions ignore masked values
- Useful for handling missing data

### Datetime Arrays

```python
# Create datetime arrays
dates = np.array(['2023-01-01', '2023-01-02', '2023-01-03'], dtype='datetime64')
dates = np.array(['2023-01-01', '2023-01-02'], dtype='datetime64[D]')  # Day precision

# Current date/time
today = np.datetime64('today')
now = np.datetime64('now')

# Date arithmetic
dates = np.array(['2023-01-01', '2023-01-15'], dtype='datetime64[D]')
deltas = np.array([7, 14], dtype='timedelta64[D]')
future_dates = dates + deltas

# Time differences
date1 = np.datetime64('2023-01-01')
date2 = np.datetime64('2023-01-15')
diff = date2 - date1  # timedelta64[14D]

# Business days (requires pandas or custom logic)
# With pandas:
import pandas as pd
business_days = pd.bdate_range('2023-01-01', '2023-01-31')
```

### String Operations

```python
# Character array module
arr = np.array(['hello', 'world', 'numpy'])

# String operations
upper = np.char.upper(arr)
lower = np.char.lower(arr)
capitalize = np.char.capitalize(arr)

# String manipulation
strip = np.char.strip(arr, chars=' ')
lstrip = np.char.lstrip(arr, chars=' ')
rstrip = np.char.rstrip(arr, chars=' ')

# Finding and replacing
find_result = np.char.find(arr, 'l')  # Returns index of first occurrence
replace = np.char.replace(arr, 'l', 'L')

# Splitting and joining
split = np.char.split(arr, sep='l')
join = np.char.join('-', arr)

# Comparison
equal = np.char.equal(arr, 'hello')
contains = np.char.find(arr, 'o') >= 0
```

---

## Interoperability

NumPy arrays can be converted to and from various other data structures.

### NumPy <-> Python Lists

```python
# List to NumPy
python_list = [1, 2, 3, 4, 5]
arr = np.array(python_list)

# NumPy to list
arr = np.array([1, 2, 3, 4, 5])
python_list = arr.tolist()

# Multi-dimensional
arr_2d = np.array([[1, 2], [3, 4]])
list_2d = arr_2d.tolist()
```

### NumPy <-> Pandas

```python
import pandas as pd

# NumPy to Pandas
arr = np.array([[1, 2, 3], [4, 5, 6]])
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])

# Pandas to NumPy
arr_from_df = df.values  # Legacy, returns view if possible
arr_from_df = df.to_numpy()  # Modern, always returns array

# Series to NumPy
series = pd.Series([1, 2, 3, 4, 5])
arr_from_series = series.values
arr_from_series = series.to_numpy()
```

### NumPy <-> PIL/Pillow

```python
from PIL import Image

# NumPy to PIL Image
arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
img = Image.fromarray(arr)

# PIL Image to NumPy
img = Image.open('image.jpg')
arr = np.array(img)

# Grayscale
arr_gray = np.array(img.convert('L'))
```

### NumPy <-> PyTorch

```python
import torch

# NumPy to PyTorch
arr = np.array([1, 2, 3, 4, 5])
tensor = torch.from_numpy(arr)  # Shares memory!

# PyTorch to NumPy
tensor = torch.tensor([1, 2, 3, 4, 5])
arr = tensor.numpy()  # Shares memory if on CPU

# On GPU: must move to CPU first
tensor_gpu = tensor.cuda()
arr = tensor_gpu.cpu().numpy()
```

### NumPy <-> TensorFlow

```python
import tensorflow as tf

# NumPy to TensorFlow
arr = np.array([1, 2, 3, 4, 5])
tensor = tf.constant(arr)

# TensorFlow to NumPy
tensor = tf.constant([1, 2, 3, 4, 5])
arr = tensor.numpy()

# Eager execution required for .numpy()
# In graph mode, use session.run()
```

### Buffer Protocol and __array_interface__

```python
# NumPy arrays implement buffer protocol
arr = np.array([1, 2, 3, 4, 5])
buffer = memoryview(arr)  # Python buffer protocol

# __array_interface__ for custom array-like objects
class ArrayLike:
    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype
    
    def __array_interface__(self):
        return {
            'shape': self.shape,
            'typestr': self.dtype.str,
            'data': (self.data.ctypes.data, False),
            'version': 3
        }

# Can be converted to NumPy array
arr_like = ArrayLike(np.array([1, 2, 3]))
arr = np.asarray(arr_like)
```

---

## Common Pitfalls and Debugging

### Mutable Default Gotcha

```python
# WRONG: Mutable default argument
def bad_function(arr=[]):
    arr.append(1)
    return arr

# RIGHT: Use None as default
def good_function(arr=None):
    if arr is None:
        arr = []
    arr.append(1)
    return arr

# For NumPy arrays
def process_array(arr=None):
    if arr is None:
        arr = np.array([])
    # Process array
    return arr
```

### Integer Overflow

```python
# Small dtypes can overflow
arr = np.array([100, 200, 300], dtype=np.int8)
arr += 100  # Overflow! Values wrap around

# Use appropriate dtypes
arr = np.array([100, 200, 300], dtype=np.int16)  # Or int32/int64

# Check for overflow
arr = np.array([100, 200, 300], dtype=np.int8)
result = arr.astype(np.int16) + 100  # Safe operation
```

### Floating Point Comparison

```python
# WRONG: Direct comparison
a = 0.1 + 0.2
if a == 0.3:  # False! Floating point precision
    print("Equal")

# RIGHT: Use np.isclose or np.allclose
if np.isclose(a, 0.3):
    print("Equal")

# For arrays
arr1 = np.array([0.1, 0.2, 0.3])
arr2 = np.array([0.1, 0.2, 0.3000001])
if np.allclose(arr1, arr2):
    print("Arrays are close")

# With tolerance
if np.allclose(arr1, arr2, rtol=1e-5, atol=1e-8):
    print("Arrays are close")
```

### Shape Mismatch Debugging

```python
# Common shape errors
arr1 = np.array([1, 2, 3])  # Shape: (3,)
arr2 = np.array([[1, 2], [3, 4]])  # Shape: (2, 2)

# Debugging tips
print(f"arr1 shape: {arr1.shape}")
print(f"arr2 shape: {arr2.shape}")
print(f"arr1 ndim: {arr1.ndim}")
print(f"arr2 ndim: {arr2.ndim}")

# Reshape for compatibility
arr1_2d = arr1.reshape(1, -1)  # Shape: (1, 3)
arr1_2d = arr1[np.newaxis, :]  # Alternative: Shape: (1, 3)
```

### Silent Broadcasting Bugs

```python
# Broadcasting can hide bugs
arr1 = np.array([[1, 2, 3]])  # Shape: (1, 3)
arr2 = np.array([4, 5, 6])     # Shape: (3,)

result = arr1 + arr2  # Broadcasts to (1, 3) + (1, 3) = (1, 3)
# Works, but might not be intended!

# Be explicit about dimensions
arr2_expanded = arr2[np.newaxis, :]  # Shape: (1, 3)
result = arr1 + arr2_expanded  # More explicit

# Use np.broadcast_to to see what will happen
broadcast_shape = np.broadcast_shapes(arr1.shape, arr2.shape)
print(f"Broadcast shape: {broadcast_shape}")
```

### Copy vs View Confusion

```python
# Common confusion: when is it a copy vs view?
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Views (share memory)
view1 = arr[1:3, :]      # View
view2 = arr[:, 1:3]      # View
view3 = arr[::2, :]      # View (strided)

# Copies (new memory)
copy1 = arr[[0, 2], :]   # Copy (fancy indexing)
copy2 = arr[:, [0, 2]]   # Copy (fancy indexing)
copy3 = arr.copy()       # Explicit copy

# Check if view or copy
print(f"view1.base is arr: {view1.base is arr}")  # True if view
print(f"copy1.base is arr: {copy1.base is arr}")  # False if copy

# Modify to test
view1[0, 0] = 999
print(arr[1, 0])  # 999 if view, unchanged if copy
```

### Debugging Tips

```python
# Check array properties
arr = np.array([1, 2, 3])
print(f"Shape: {arr.shape}")
print(f"Strides: {arr.strides}")
print(f"Dtype: {arr.dtype}")
print(f"Size: {arr.size}")
print(f"Itemsize: {arr.itemsize}")
print(f"Flags: {arr.flags}")
print(f"Memory layout: C={arr.flags['C_CONTIGUOUS']}, F={arr.flags['F_CONTIGUOUS']}")

# Check for NaN/Inf
arr = np.array([1, 2, np.nan, 4, np.inf])
print(f"Has NaN: {np.isnan(arr).any()}")
print(f"Has Inf: {np.isinf(arr).any()}")
print(f"Finite: {np.isfinite(arr)}")

# Memory usage
print(f"Memory usage: {arr.nbytes} bytes")
print(f"Memory usage (MB): {arr.nbytes / 1024**2:.2f} MB")
```

---

## Summary

This module covered:

1. **Memory and Performance**: Memory layout, strides, views vs copies, vectorization, and memory-mapped files
2. **File I/O**: Binary and text formats, structured data loading
3. **Structured Arrays**: Heterogeneous data types, nested structures, record arrays
4. **Special Array Types**: Masked arrays, datetime arrays, string operations
5. **Interoperability**: Conversion between NumPy and other libraries
6. **Common Pitfalls**: Floating point comparison, broadcasting, copy vs view confusion

Understanding these topics is essential for writing efficient, correct NumPy code and integrating NumPy with other scientific computing libraries.
