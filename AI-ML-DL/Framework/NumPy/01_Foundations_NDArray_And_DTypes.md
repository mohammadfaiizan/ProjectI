# NumPy Foundations: NDArray and DTypes

## Table of Contents
1. [Introduction to NumPy](#introduction-to-numpy)
2. [NDArray Object Internals](#ndarray-object-internals)
3. [Memory Layout](#memory-layout)
4. [Array Creation Methods](#array-creation-methods)
5. [Complete DType System](#complete-dtype-system)
6. [Views vs Copies](#views-vs-copies)
7. [Array Attributes](#array-attributes)
8. [Array Protocols](#array-protocols)

---

## Introduction to NumPy

NumPy (Numerical Python) is the fundamental package for scientific computing in Python. It provides a powerful N-dimensional array object and tools for working with these arrays.

### Why NumPy Matters

**C-Backed Implementation**: NumPy arrays are implemented in C, making them significantly faster than Python lists. The core array operations are written in C and compiled to machine code, avoiding Python's interpreter overhead.

**Contiguous Memory**: NumPy arrays store data in contiguous blocks of memory, enabling efficient cache utilization and vectorized operations. This contrasts with Python lists, which store pointers to objects scattered throughout memory.

**SIMD Operations**: Single Instruction Multiple Data (SIMD) instructions allow modern CPUs to perform the same operation on multiple data points simultaneously. NumPy leverages SIMD through optimized BLAS (Basic Linear Algebra Subprograms) libraries.

**Vectorization**: Vectorization eliminates explicit loops by applying operations to entire arrays at once. This is both more readable and dramatically faster than Python loops.

```python
import numpy as np

# Python list approach (slow)
python_list = [i**2 for i in range(1000000)]

# NumPy vectorized approach (fast)
numpy_array = np.arange(1000000)**2
```

---

## NDArray Object Internals

The `ndarray` (N-dimensional array) is NumPy's core data structure. Understanding its internal components is crucial for efficient NumPy usage.

### Data Buffer

The data buffer is a contiguous block of memory storing the actual array elements. It's a raw C array, accessible via the array's data pointer.

```python
arr = np.array([1, 2, 3, 4, 5])
print(arr.data)  # <memory at 0x...>
```

### DType

The dtype (data type) describes how bytes in memory should be interpreted. It specifies:
- Type of data (integer, float, complex, boolean, string, object)
- Size in bytes
- Byte order (endianness)
- For structured types: field names and offsets

```python
arr = np.array([1, 2, 3])
print(arr.dtype)  # dtype('int64')
```

### Shape

Shape is a tuple indicating the size of each dimension. For a 2D array with 3 rows and 4 columns, shape is `(3, 4)`.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # (2, 3)
```

### Strides

Strides is a tuple indicating the number of bytes to step in each dimension when traversing the array. It determines how data is laid out in memory.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.strides)  # (24, 8) for int64: 3 elements * 8 bytes = 24 bytes per row
```

For a C-contiguous array of shape `(a, b, c)` with dtype size `s`:
- Strides = `(b*c*s, c*s, s)`

### Flags

Flags provide metadata about array properties:
- `C_CONTIGUOUS`: Data is in C-order (row-major)
- `F_CONTIGUOUS`: Data is in Fortran-order (column-major)
- `OWNDATA`: Array owns its data
- `WRITEABLE`: Data can be modified
- `ALIGNED`: Data is aligned for hardware requirements
- `WRITEBACKIFCOPY`: Array is a copy that will be written back

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.flags)
#   C_CONTIGUOUS : True
#   F_CONTIGUOUS : False
#   OWNDATA : True
#   WRITEABLE : True
#   ALIGNED : True
```

### Itemsize

Itemsize is the size in bytes of a single array element.

```python
arr_int32 = np.array([1, 2, 3], dtype=np.int32)
print(arr_int32.itemsize)  # 4 bytes

arr_float64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
print(arr_float64.itemsize)  # 8 bytes
```

### Nbytes

Nbytes is the total number of bytes consumed by the array data.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.nbytes)  # 48 bytes (6 elements * 8 bytes for int64)
```

---

## Memory Layout

### Row-Major (C Order)

In C-order (row-major), the last dimension varies fastest. Elements are stored row by row.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]], order='C')
# Memory layout: [1, 2, 3, 4, 5, 6]
```

For a 2D array `arr[i, j]`, the memory offset is:
```
offset = i * strides[0] + j * strides[1]
```

### Column-Major (Fortran Order)

In Fortran-order (column-major), the first dimension varies fastest. Elements are stored column by column.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]], order='F')
# Memory layout: [1, 4, 2, 5, 3, 6]
```

### Contiguity Flags

An array is C-contiguous if it's stored row-major and elements are packed without gaps. It's F-contiguous if stored column-major without gaps.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.flags.c_contiguous)  # True
print(arr.flags.f_contiguous)  # False

# Transpose is F-contiguous
arr_T = arr.T
print(arr_T.flags.c_contiguous)  # False
print(arr_T.flags.f_contiguous)  # True
```

---

## Array Creation Methods

### Basic Array Creation

#### np.array

Creates an array from a sequence-like object.

```python
# From list
arr1 = np.array([1, 2, 3, 4, 5])

# From nested list (2D)
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

# Specify dtype
arr3 = np.array([1, 2, 3], dtype=np.float32)

# Specify order
arr4 = np.array([[1, 2], [3, 4]], order='F')
```

#### np.zeros

Creates an array filled with zeros.

```python
# 1D array
arr = np.zeros(5)
# [0. 0. 0. 0. 0.]

# 2D array
arr = np.zeros((3, 4))
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# With dtype
arr = np.zeros((2, 3), dtype=np.int32)
```

#### np.ones

Creates an array filled with ones.

```python
arr = np.ones((2, 3))
# [[1. 1. 1.]
#  [1. 1. 1.]]

arr = np.ones((2, 3), dtype=np.int32)
```

#### np.full

Creates an array filled with a specified value.

```python
arr = np.full((2, 3), 7)
# [[7. 7. 7.]
#  [7. 7. 7.]]

arr = np.full((2, 3), np.pi, dtype=np.float32)
```

#### np.empty

Creates an array without initializing values. Faster than zeros/ones, but contains garbage data.

```python
arr = np.empty((2, 3))
# Values are uninitialized (garbage)
```

### Sequence Generation

#### np.arange

Creates evenly spaced values within a specified interval.

```python
# Start, stop, step
arr = np.arange(0, 10, 2)
# [0 2 4 6 8]

# Start, stop (step defaults to 1)
arr = np.arange(5)
# [0 1 2 3 4]

# With float step
arr = np.arange(0, 1, 0.1)
# [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
```

#### np.linspace

Creates evenly spaced numbers over a specified interval.

```python
# Start, stop, num_points
arr = np.linspace(0, 10, 5)
# [ 0.   2.5  5.   7.5 10. ]

# Exclude endpoint
arr = np.linspace(0, 10, 5, endpoint=False)
# [0. 2. 4. 6. 8.]

# Return step size
arr, step = np.linspace(0, 10, 5, retstep=True)
```

#### np.logspace

Creates numbers spaced evenly on a log scale.

```python
# Base 10: 10^start to 10^stop
arr = np.logspace(0, 2, 3)
# [  1.  10. 100.]

# Custom base
arr = np.logspace(0, 2, 3, base=2)
# [1. 2. 4.]

# Specify actual start/stop values
arr = np.logspace(np.log10(1), np.log10(100), 3)
```

#### np.geomspace

Creates numbers spaced evenly on a log scale (geometric progression).

```python
arr = np.geomspace(1, 100, 3)
# [  1.  10. 100.]
```

### Special Arrays

#### np.eye

Creates an identity matrix (2D).

```python
# NxN identity matrix
arr = np.eye(3)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# With offset
arr = np.eye(3, k=1)  # Offset diagonal up
# [[0. 1. 0.]
#  [0. 0. 1.]
#  [0. 0. 0.]]

# Rectangular
arr = np.eye(3, 4)
```

#### np.identity

Creates a square identity matrix (always NxN).

```python
arr = np.identity(3)
# Same as np.eye(3)
```

#### np.diag

Creates a diagonal array or extracts diagonal.

```python
# Create diagonal matrix from 1D array
arr = np.diag([1, 2, 3])
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]

# Extract diagonal from 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
diag = np.diag(arr_2d)
# [1 5 9]

# With offset
arr = np.diag([1, 2, 3], k=1)  # Offset up
```

#### np.tri

Creates a lower triangular matrix.

```python
arr = np.tri(3, 3)
# [[1. 0. 0.]
#  [1. 1. 0.]
#  [1. 1. 1.]]

# With offset
arr = np.tri(3, 3, k=-1)  # Offset down
```

### Like Functions

#### np.zeros_like

Creates an array of zeros with the same shape and dtype as another array.

```python
template = np.array([[1, 2], [3, 4]])
arr = np.zeros_like(template)
# [[0 0]
#  [0 0]]

# Override dtype
arr = np.zeros_like(template, dtype=np.float32)
```

#### np.ones_like

Creates an array of ones with the same shape and dtype.

```python
template = np.array([[1, 2], [3, 4]])
arr = np.ones_like(template)
```

#### np.full_like

Creates an array filled with a value, matching shape and dtype.

```python
template = np.array([[1, 2], [3, 4]])
arr = np.full_like(template, 5)
```

#### np.empty_like

Creates an uninitialized array with the same shape and dtype.

```python
template = np.array([[1, 2], [3, 4]])
arr = np.empty_like(template)
```

### From Functions and Iterables

#### np.fromfunction

Creates an array by executing a function over coordinate arrays.

```python
def func(i, j):
    return i + j

arr = np.fromfunction(func, (3, 3))
# [[0. 1. 2.]
#  [1. 2. 3.]
#  [2. 3. 4.]]
```

#### np.fromiter

Creates an array from an iterable.

```python
# From generator
gen = (x**2 for x in range(5))
arr = np.fromiter(gen, dtype=np.int32)
# [ 0  1  4  9 16]

# With count
arr = np.fromiter(range(5), dtype=np.int32, count=3)
# [0 1 2]
```

#### np.frombuffer

Creates an array from a buffer object (shared memory).

```python
import array
buf = array.array('i', [1, 2, 3, 4, 5])
arr = np.frombuffer(buf, dtype=np.int32)
# [1 2 3 4 5]
# Modifying arr modifies buf (shared memory)
```

#### np.fromstring

Creates an array from a string (deprecated, use frombuffer).

```python
# Binary string
s = b'\x01\x00\x00\x00\x02\x00\x00\x00'
arr = np.fromstring(s, dtype=np.int32)
# [1 2]
```

### Grid Generation

#### np.meshgrid

Creates coordinate matrices from coordinate vectors.

```python
x = np.array([1, 2, 3])
y = np.array([4, 5])
X, Y = np.meshgrid(x, y)
# X = [[1 2 3]
#      [1 2 3]]
# Y = [[4 4 4]
#      [5 5 5]]

# Sparse output
X, Y = np.meshgrid(x, y, sparse=True)
```

#### np.mgrid

Creates dense multi-dimensional coordinate arrays.

```python
# Slice notation
arr = np.mgrid[0:5, 0:3]
# Returns array of shape (2, 5, 3)

# Step notation
arr = np.mgrid[0:5:2, 0:3:1]
```

#### np.ogrid

Creates open (sparse) multi-dimensional coordinate arrays.

```python
arr = np.ogrid[0:5, 0:3]
# Returns list of 1D arrays
```

---

## Complete DType System

### Integer Types

| Type | Description | Range |
|------|-------------|-------|
| int8 | 8-bit signed integer | -128 to 127 |
| int16 | 16-bit signed integer | -32768 to 32767 |
| int32 | 32-bit signed integer | -2^31 to 2^31-1 |
| int64 | 64-bit signed integer | -2^63 to 2^63-1 |
| uint8 | 8-bit unsigned integer | 0 to 255 |
| uint16 | 16-bit unsigned integer | 0 to 65535 |
| uint32 | 32-bit unsigned integer | 0 to 2^32-1 |
| uint64 | 64-bit unsigned integer | 0 to 2^64-1 |

```python
# Signed integers
arr_int8 = np.array([-128, 0, 127], dtype=np.int8)
arr_int64 = np.array([1, 2, 3], dtype=np.int64)

# Unsigned integers
arr_uint8 = np.array([0, 128, 255], dtype=np.uint8)
arr_uint32 = np.array([1, 2, 3], dtype=np.uint32)
```

### Float Types

| Type | Description | Precision |
|------|-------------|-----------|
| float16 | Half precision | ~3 decimal digits |
| float32 | Single precision | ~7 decimal digits |
| float64 | Double precision | ~15 decimal digits |
| float128 | Extended precision | ~18 decimal digits |

```python
arr_float32 = np.array([1.5, 2.7, 3.9], dtype=np.float32)
arr_float64 = np.array([1.5, 2.7, 3.9], dtype=np.float64)
arr_float128 = np.array([1.5, 2.7, 3.9], dtype=np.float128)
```

### Complex Types

| Type | Description | Components |
|------|-------------|------------|
| complex64 | Complex with float32 components | 2 × float32 |
| complex128 | Complex with float64 components | 2 × float64 |

```python
arr_complex64 = np.array([1+2j, 3+4j], dtype=np.complex64)
arr_complex128 = np.array([1+2j, 3+4j], dtype=np.complex128)
```

### Boolean Type

```python
arr_bool = np.array([True, False, True], dtype=np.bool_)
# or
arr_bool = np.array([1, 0, 1], dtype=bool)
```

### String Types

| Type | Description |
|------|-------------|
| U | Unicode string (fixed length) |
| S | Byte string (fixed length) |

```python
# Unicode strings
arr_unicode = np.array(['hello', 'world'], dtype='U10')
arr_unicode = np.array(['hello', 'world'], dtype=np.unicode_)

# Byte strings
arr_bytes = np.array([b'hello', b'world'], dtype='S10')
arr_bytes = np.array([b'hello', b'world'], dtype=np.bytes_)
```

### Object Type

Stores Python objects (flexible but slower).

```python
arr_obj = np.array([1, 'hello', [1, 2, 3]], dtype=object)
```

### Void Type

For raw data or structured data.

```python
arr_void = np.array([b'\x01\x02\x03'], dtype='V3')
```

### Structured DTypes

Define custom structures with named fields.

```python
# Define structured dtype
dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])

# Create structured array
arr = np.array([('Alice', 25, 55.5), ('Bob', 30, 70.2)], dtype=dt)

# Access fields
print(arr['name'])  # ['Alice' 'Bob']
print(arr['age'])   # [25 30]
print(arr[0]['name'])  # 'Alice'
```

### Record Arrays

Alternative interface for structured arrays.

```python
arr = np.rec.array([('Alice', 25, 55.5), ('Bob', 30, 70.2)],
                   dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])

# Attribute access
print(arr.name)  # ['Alice' 'Bob']
print(arr.age)   # [25 30]
```

### Custom DType Creation

```python
# From Python type
dt = np.dtype(int)

# From string
dt = np.dtype('i4')  # 32-bit integer
dt = np.dtype('f8')  # 64-bit float

# With endianness
dt = np.dtype('>i4')  # Big-endian
dt = np.dtype('<i4')  # Little-endian

# Structured with offsets
dt = np.dtype({'names': ['x', 'y'], 'formats': ['f4', 'f4'], 'offsets': [0, 4]})
```

### Type Casting

#### astype()

Convert array to different dtype.

```python
arr = np.array([1, 2, 3], dtype=np.int32)
arr_float = arr.astype(np.float64)
arr_int16 = arr.astype(np.int16)
```

#### Casting Rules

NumPy supports three casting safety levels:

**Safe Casting**: Only allows casts that preserve values.

```python
arr = np.array([1, 2, 3], dtype=np.int32)
# Safe: int32 -> int64, int32 -> float64
arr.astype(np.int64, casting='safe')
arr.astype(np.float64, casting='safe')
```

**Same Kind Casting**: Allows casts within the same kind (int to int, float to float).

```python
arr = np.array([1, 2, 3], dtype=np.int32)
# Same kind: int32 -> int64, int32 -> int16
arr.astype(np.int16, casting='same_kind')
```

**Unsafe Casting**: Allows any cast (may lose precision or overflow).

```python
arr = np.array([1.5, 2.7, 3.9], dtype=np.float64)
# Unsafe: float64 -> int32 (truncates)
arr.astype(np.int32, casting='unsafe')
```

---

## Views vs Copies

Understanding when NumPy creates views versus copies is crucial for memory efficiency and avoiding bugs.

### Views

A view shares the same data buffer as the original array. Modifying a view modifies the original.

```python
arr = np.array([1, 2, 3, 4, 5])
view = arr[1:4]  # Slice creates a view
view[0] = 99
print(arr)  # [ 1 99  3  4  5] - original modified!
```

### Copies

A copy has its own data buffer. Modifying a copy does not affect the original.

```python
arr = np.array([1, 2, 3, 4, 5])
copy = arr.copy()  # Explicit copy
copy[0] = 99
print(arr)  # [1 2 3 4 5] - original unchanged
```

### When Views Are Created

- Slicing: `arr[1:4]`, `arr[:, 1:3]`
- Transpose: `arr.T`
- Reshape (if data is contiguous): `arr.reshape(2, 3)`
- Changing dtype (if compatible): `arr.view(np.int32)`

### When Copies Are Created

- Fancy indexing: `arr[[0, 2, 4]]`
- Boolean indexing: `arr[arr > 2]`
- Explicit copy: `arr.copy()`
- astype() (usually): `arr.astype(np.float32)`
- Reshape (if data is not contiguous)

### Checking Views vs Copies

#### .base Attribute

Views have a `.base` attribute pointing to the original array.

```python
arr = np.array([1, 2, 3, 4, 5])
view = arr[1:4]
print(view.base is arr)  # True

copy = arr.copy()
print(copy.base is None)  # True
```

#### np.shares_memory()

Checks if two arrays share memory.

```python
arr = np.array([1, 2, 3, 4, 5])
view = arr[1:4]
print(np.shares_memory(arr, view))  # True

copy = arr.copy()
print(np.shares_memory(arr, copy))  # False
```

---

## Array Attributes

### Shape

Tuple indicating size of each dimension.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # (2, 3)

# Modify shape (if compatible)
arr.shape = (3, 2)
```

### NDim

Number of dimensions (rank).

```python
arr_1d = np.array([1, 2, 3])
print(arr_1d.ndim)  # 1

arr_2d = np.array([[1, 2], [3, 4]])
print(arr_2d.ndim)  # 2
```

### Size

Total number of elements.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.size)  # 6
```

### DType

Data type descriptor.

```python
arr = np.array([1, 2, 3])
print(arr.dtype)  # dtype('int64')
```

### Itemsize

Size in bytes of a single element.

```python
arr_int32 = np.array([1, 2, 3], dtype=np.int32)
print(arr_int32.itemsize)  # 4

arr_float64 = np.array([1.0, 2.0], dtype=np.float64)
print(arr_float64.itemsize)  # 8
```

### Nbytes

Total bytes consumed by array data.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
print(arr.nbytes)  # 24 (6 elements × 4 bytes)
```

### Strides

Tuple of bytes to step in each dimension.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
print(arr.strides)  # (24, 8) - 3 elements × 8 bytes per row, 8 bytes per element
```

### Flags

Array flags providing metadata.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.flags.c_contiguous)  # True
print(arr.flags.f_contiguous)  # False
print(arr.flags.writeable)     # True
print(arr.flags.owndata)       # True
```

### Flat

1D iterator over array (flattened view).

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
for val in arr.flat:
    print(val)  # 1, 2, 3, 4, 5, 6

# Can also assign
arr.flat[0] = 99
```

### T

Transpose attribute (shorthand for `.transpose()`).

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.T)
# [[1 4]
#  [2 5]
#  [3 6]]
```

---

## Array Protocols

NumPy arrays support several protocols for interoperability with other libraries and custom classes.

### __array__

Allows objects to be converted to NumPy arrays.

```python
class CustomArray:
    def __init__(self, data):
        self.data = data
    
    def __array__(self):
        return np.array(self.data)

obj = CustomArray([1, 2, 3])
arr = np.asarray(obj)  # Uses __array__
```

### __array_interface__

Low-level protocol for exposing array data (C-level).

```python
arr = np.array([1, 2, 3])
print(arr.__array_interface__)
# {'data': (..., False), 'strides': None, 'descr': [('', '<i8')], 
#  'typestr': '<i8', 'shape': (3,), 'version': 3}
```

### __array_ufunc__

Allows custom classes to override NumPy ufunc behavior.

```python
class CustomArray:
    def __init__(self, data):
        self.data = np.asarray(data)
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            return CustomArray(ufunc(self.data))
        return NotImplemented

arr = CustomArray([1, 2, 3])
result = np.sqrt(arr)  # Uses __array_ufunc__
```

These protocols enable seamless integration between NumPy and other numerical computing libraries, allowing for efficient data sharing and operation overloading.
