# NumPy Mathematics, Statistics, and Broadcasting

## Table of Contents
1. [Arithmetic Operations](#arithmetic-operations)
2. [Universal Functions (Ufuncs)](#universal-functions-ufuncs)
3. [Trigonometric Functions](#trigonometric-functions)
4. [Exponential and Logarithmic Functions](#exponential-and-logarithmic-functions)
5. [Rounding Functions](#rounding-functions)
6. [Comparison Operations](#comparison-operations)
7. [Logical Operations](#logical-operations)
8. [Aggregation and Reduction Operations](#aggregation-and-reduction-operations)
9. [NaN-Safe Functions](#nan-safe-functions)
10. [Broadcasting](#broadcasting)
11. [Clipping Operations](#clipping-operations)
12. [Complex Number Operations](#complex-number-operations)

---

## Arithmetic Operations

NumPy supports element-wise arithmetic operations that are vectorized and highly optimized.

### Basic Arithmetic Operators

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Addition
result = a + b
print(result)  # [ 6  8 10 12]

# Subtraction
result = a - b
print(result)  # [-4 -4 -4 -4]

# Multiplication (element-wise, not matrix multiplication)
result = a * b
print(result)  # [ 5 12 21 32]

# Division
result = a / b
print(result)  # [0.2 0.33333333 0.42857143 0.5]

# Floor division
result = a // b
print(result)  # [0 0 0 0]

# Modulo
result = a % b
print(result)  # [1 2 3 4]

# Power
result = a ** b
print(result)  # [    1    64  2187 65536]
```

### Ufunc Equivalents

Each arithmetic operator has a corresponding ufunc:

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Addition
result = np.add(a, b)  # Equivalent to a + b

# Subtraction
result = np.subtract(a, b)  # Equivalent to a - b

# Multiplication
result = np.multiply(a, b)  # Equivalent to a * b

# Division
result = np.divide(a, b)  # Equivalent to a / b

# Floor division
result = np.floor_divide(a, b)  # Equivalent to a // b

# Modulo
result = np.mod(a, b)  # Equivalent to a % b

# Power
result = np.power(a, b)  # Equivalent to a ** b

# True division (always float)
result = np.true_divide(a, b)

# Remainder
result = np.remainder(a, b)  # Same as mod
```

### In-Place Operations

NumPy supports in-place operations for efficiency:

```python
a = np.array([1, 2, 3, 4])
a += 1  # In-place addition
print(a)  # [2 3 4 5]

a *= 2  # In-place multiplication
print(a)  # [ 4  6  8 10]
```

---

## Universal Functions (Ufuncs)

Universal functions (ufuncs) are functions that operate element-wise on arrays, providing vectorization and broadcasting capabilities.

### What Are Ufuncs

Ufuncs are NumPy functions that:
- Operate element-wise on arrays
- Support broadcasting
- Return arrays (not scalars for array inputs)
- Are implemented in C for speed
- Support various methods (reduce, accumulate, outer, etc.)

### Vectorization Concept

Vectorization eliminates Python loops by applying operations to entire arrays:

```python
# Slow: Python loop
arr = np.array([1, 2, 3, 4, 5])
result = []
for x in arr:
    result.append(x ** 2)

# Fast: Vectorized ufunc
arr = np.array([1, 2, 3, 4, 5])
result = np.power(arr, 2)  # or arr ** 2
```

### Unary Ufuncs

Unary ufuncs operate on a single array:

#### Absolute Value

```python
arr = np.array([-1, -2, 3, -4, 5])
result = np.abs(arr)  # or np.absolute(arr)
print(result)  # [1 2 3 4 5]
```

#### Square Root

```python
arr = np.array([1, 4, 9, 16, 25])
result = np.sqrt(arr)
print(result)  # [1. 2. 3. 4. 5.]
```

#### Exponential

```python
arr = np.array([0, 1, 2])
result = np.exp(arr)
print(result)  # [1.         2.71828183 7.3890561 ]
```

#### Logarithms

```python
arr = np.array([1, 10, 100])

# Natural logarithm (base e)
result = np.log(arr)
print(result)  # [0.         2.30258509 4.60517019]

# Base 2 logarithm
result = np.log2(arr)
print(result)  # [0.         3.32192809 6.64385619]

# Base 10 logarithm
result = np.log10(arr)
print(result)  # [0. 1. 2.]
```

#### Ceiling and Floor

```python
arr = np.array([1.2, 2.7, 3.1, 4.9])

# Ceiling (round up)
result = np.ceil(arr)
print(result)  # [2. 3. 4. 5.]

# Floor (round down)
result = np.floor(arr)
print(result)  # [1. 2. 3. 4.]
```

#### Rounding

```python
arr = np.array([1.23, 2.67, 3.45, 4.89])

# Round to nearest integer
result = np.round(arr)
print(result)  # [1. 3. 3. 5.]

# Round to specified decimals
result = np.round(arr, decimals=1)
print(result)  # [1.2 2.7 3.4 4.9]
```

#### Sign

```python
arr = np.array([-5, 0, 5, -3.5, 2.7])
result = np.sign(arr)
print(result)  # [-1.  0.  1. -1.  1.]
```

#### Negative

```python
arr = np.array([1, 2, -3, 4])
result = np.negative(arr)
print(result)  # [-1 -2  3 -4]
```

#### Reciprocal

```python
arr = np.array([1, 2, 4, 8])
result = np.reciprocal(arr)
print(result)  # [1.   0.5  0.25 0.125]
```

### Binary Ufuncs

Binary ufuncs operate on two arrays:

#### Maximum and Minimum

```python
a = np.array([1, 5, 3, 7])
b = np.array([2, 4, 6, 5])

# Element-wise maximum
result = np.maximum(a, b)
print(result)  # [2 5 6 7]

# Element-wise minimum
result = np.minimum(a, b)
print(result)  # [1 4 3 5]
```

#### Modulo Variants

```python
a = np.array([7, -7, 7, -7])
b = np.array([3, 3, -3, -3])

# Standard modulo (remainder)
result = np.mod(a, b)
print(result)  # [1 2 -2 -1]

# Floating point modulo
result = np.fmod(a, b)
print(result)  # [ 1 -1  1 -1]
```

### Ufunc Methods

Ufuncs support several methods for advanced operations:

#### reduce()

Applies ufunc along axis, reducing dimensions:

```python
arr = np.array([1, 2, 3, 4, 5])

# Sum reduction
result = np.add.reduce(arr)
print(result)  # 15

# Product reduction
result = np.multiply.reduce(arr)
print(result)  # 120

# With axis (2D array)
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
result = np.add.reduce(arr_2d, axis=0)  # Sum along rows
print(result)  # [5 7 9]

result = np.add.reduce(arr_2d, axis=1)  # Sum along columns
print(result)  # [ 6 15]
```

#### accumulate()

Returns accumulated result:

```python
arr = np.array([1, 2, 3, 4, 5])

# Cumulative sum
result = np.add.accumulate(arr)
print(result)  # [ 1  3  6 10 15]

# Cumulative product
result = np.multiply.accumulate(arr)
print(result)  # [  1   2   6  24 120]
```

#### outer()

Applies ufunc to all pairs of elements:

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Outer product
result = np.multiply.outer(a, b)
print(result)
# [[ 4  5  6]
#  [ 8 10 12]
#  [12 15 18]]
```

#### at()

Performs unbuffered in-place operation at specified indices:

```python
arr = np.array([1, 2, 3, 4, 5])
indices = [0, 2, 4]

# Add 10 at specified indices
np.add.at(arr, indices, 10)
print(arr)  # [11  2 13  4 15]
```

#### reduceat()

Reduces array at specified indices:

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
indices = [0, 3, 5]

# Reduce segments: [0:3), [3:5), [5:end]
result = np.add.reduceat(arr, indices)
print(result)  # [ 6  9 21]  # sum([1,2,3]), sum([4,5]), sum([6,7,8])
```

### Custom Ufuncs

Create custom ufuncs from Python functions:

```python
def custom_func(x, y):
    return x ** 2 + y ** 2

# Create ufunc from Python function
custom_ufunc = np.frompyfunc(custom_func, 2, 1)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = custom_ufunc(a, b)
print(result)  # [17 29 45]
```

---

## Trigonometric Functions

NumPy provides comprehensive trigonometric functions operating in radians.

### Basic Trigonometric Functions

```python
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])

# Sine
sine = np.sin(angles)
print(sine)  # [0.         0.5        0.70710678 0.8660254  1.        ]

# Cosine
cosine = np.cos(angles)
print(cosine)  # [1.00000000e+00 8.66025404e-01 7.07106781e-01 5.00000000e-01 6.12323400e-17]

# Tangent
tangent = np.tan(angles)
print(tangent)  # [0.00000000e+00 5.77350269e-01 1.00000000e+00 1.73205081e+00 1.63312394e+16]
```

### Inverse Trigonometric Functions

```python
values = np.array([0, 0.5, 1/np.sqrt(2), np.sqrt(3)/2, 1])

# Arcsine
arcsine = np.arcsin(values)
print(arcsine)  # [0.         0.52359878 0.78539816 1.04719755 1.57079633]

# Arccosine
arccosine = np.arccos(values)
print(arccosine)  # [1.57079633 1.04719755 0.78539816 0.52359878 0.        ]

# Arctangent
arctangent = np.arctan(values)
print(arctangent)  # [0.         0.46364761 0.78539816 0.85707195 0.78539816]
```

### Arctan2

Two-argument arctangent (returns angle in correct quadrant):

```python
y = np.array([1, 1, -1, -1])
x = np.array([1, -1, -1, 1])

angle = np.arctan2(y, x)
print(angle)  # [ 0.78539816  2.35619449 -2.35619449 -0.78539816]
```

### Hypotenuse

Computes hypotenuse of right triangle:

```python
a = np.array([3, 4, 5])
b = np.array([4, 3, 12])

hyp = np.hypot(a, b)
print(hyp)  # [ 5.  5. 13.]
```

### Degree-Radian Conversion

```python
degrees = np.array([0, 30, 45, 60, 90])

# Convert degrees to radians
radians = np.deg2rad(degrees)
print(radians)  # [0.         0.52359878 0.78539816 1.04719755 1.57079633]

# Convert radians to degrees
degrees_back = np.rad2deg(radians)
print(degrees_back)  # [ 0. 30. 45. 60. 90.]
```

### Hyperbolic Functions

```python
x = np.array([0, 1, 2])

# Hyperbolic sine
sinh = np.sinh(x)
print(sinh)  # [0.         1.17520119 3.62686041]

# Hyperbolic cosine
cosh = np.cosh(x)
print(cosh)  # [1.         1.54308063 3.76219569]

# Hyperbolic tangent
tanh = np.tanh(x)
print(tanh)  # [0.         0.76159416 0.96402758]
```

---

## Exponential and Logarithmic Functions

### Exponential Functions

```python
x = np.array([0, 1, 2])

# e^x
exp = np.exp(x)
print(exp)  # [1.         2.71828183 7.3890561 ]

# 2^x
exp2 = np.exp2(x)
print(exp2)  # [1. 2. 4.]

# e^x - 1 (more accurate for small x)
expm1 = np.expm1(x)
print(expm1)  # [0.         1.71828183 6.3890561 ]
```

### Logarithmic Functions

```python
x = np.array([1, 10, 100, 1000])

# Natural logarithm (ln)
log = np.log(x)
print(log)  # [0.         2.30258509 4.60517019 6.90775528]

# Base 2 logarithm
log2 = np.log2(x)
print(log2)  # [0.         3.32192809 6.64385619 9.96578428]

# Base 10 logarithm
log10 = np.log10(x)
print(log10)  # [0. 1. 2. 3.]

# ln(1 + x) (more accurate for small x)
log1p = np.log1p(x)
print(log1p)  # [0.69314718 2.39789527 4.61512052 6.90875478]
```

---

## Rounding Functions

NumPy provides multiple rounding functions with different behaviors:

```python
arr = np.array([1.23, 2.67, -3.45, -4.89])

# Round to nearest integer
rounded = np.round(arr)
print(rounded)  # [ 1.  3. -3. -5.]

# Round to specified decimals
rounded = np.round(arr, decimals=1)
print(rounded)  # [ 1.2  2.7 -3.4 -4.9]

# Floor (round down)
floored = np.floor(arr)
print(floored)  # [ 1.  2. -4. -5.]

# Ceiling (round up)
ceiled = np.ceil(arr)
print(ceiled)  # [ 2.  3. -3. -4.]

# Truncate (toward zero)
truncated = np.trunc(arr)
print(truncated)  # [ 1.  2. -3. -4.]

# Fix (same as trunc)
fixed = np.fix(arr)
print(fixed)  # [ 1.  2. -3. -4.]

# Around (alias for round)
around = np.around(arr)
print(around)  # [ 1.  3. -3. -5.]

# Rint (round to nearest integer)
rint = np.rint(arr)
print(rint)  # [ 1.  3. -3. -5.]
```

---

## Comparison Operations

Comparison operations return boolean arrays:

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])

# Greater than
result = np.greater(a, b)
print(result)  # [False False False  True  True]

# Less than
result = np.less(a, b)
print(result)  # [ True  True False False False]

# Equal
result = np.equal(a, b)
print(result)  # [False False  True False False]

# Not equal
result = np.not_equal(a, b)
print(result)  # [ True  True False  True  True]

# Greater than or equal
result = np.greater_equal(a, b)
print(result)  # [False False  True  True  True]

# Less than or equal
result = np.less_equal(a, b)
print(result)  # [ True  True  True False False]
```

### Operator Equivalents

```python
a = np.array([1, 2, 3])
b = np.array([2, 2, 2])

# All equivalent
result1 = a > b
result2 = np.greater(a, b)

result1 = a == b
result2 = np.equal(a, b)

result1 = a != b
result2 = np.not_equal(a, b)
```

---

## Logical Operations

Logical operations work with boolean arrays:

```python
a = np.array([True, True, False, False])
b = np.array([True, False, True, False])

# Logical AND
result = np.logical_and(a, b)
print(result)  # [ True False False False]

# Logical OR
result = np.logical_or(a, b)
print(result)  # [ True  True  True False]

# Logical NOT
result = np.logical_not(a)
print(result)  # [False False  True  True]

# Logical XOR
result = np.logical_xor(a, b)
print(result)  # [False  True  True False]
```

### All and Any

```python
arr = np.array([[True, True, False], [True, False, False]])

# Check if all elements are True
all_true = np.all(arr)
print(all_true)  # False

# Check along axis
all_axis0 = np.all(arr, axis=0)
print(all_axis0)  # [ True False False]

all_axis1 = np.all(arr, axis=1)
print(all_axis1)  # [False False]

# Check if any element is True
any_true = np.any(arr)
print(any_true)  # True

any_axis0 = np.any(arr, axis=0)
print(any_axis0)  # [ True  True False]
```

---

## Aggregation and Reduction Operations

Aggregation operations reduce arrays to scalar values or arrays with fewer dimensions.

### Sum and Product

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Sum all elements
total = np.sum(arr)
print(total)  # 21

# Sum along axis 0 (columns)
sum_cols = np.sum(arr, axis=0)
print(sum_cols)  # [5 7 9]

# Sum along axis 1 (rows)
sum_rows = np.sum(arr, axis=1)
print(sum_rows)  # [ 6 15]

# Product
product = np.prod(arr)
print(product)  # 720

# Product along axis
prod_cols = np.prod(arr, axis=0)
print(prod_cols)  # [ 4 10 18]
```

### Mean, Standard Deviation, and Variance

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Mean
mean_all = np.mean(arr)
print(mean_all)  # 3.5

mean_axis0 = np.mean(arr, axis=0)
print(mean_axis0)  # [2.5 3.5 4.5]

# Standard deviation
std_all = np.std(arr)
print(std_all)  # 1.707825127659933

std_axis0 = np.std(arr, axis=0)
print(std_axis0)  # [1.5 1.5 1.5]

# Variance
var_all = np.var(arr)
print(var_all)  # 2.9166666666666665

var_axis0 = np.var(arr, axis=0)
print(var_axis0)  # [2.25 2.25 2.25]

# With ddof (delta degrees of freedom)
std_sample = np.std(arr, ddof=1)  # Sample standard deviation
```

### Median

```python
arr = np.array([1, 3, 2, 5, 4])

median = np.median(arr)
print(median)  # 3.0

arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
median_axis0 = np.median(arr_2d, axis=0)
print(median_axis0)  # [2.5 3.5 4.5]
```

### Min and Max

```python
arr = np.array([[1, 5, 3], [7, 2, 6]])

# Minimum
min_all = np.min(arr)
print(min_all)  # 1

min_axis0 = np.min(arr, axis=0)
print(min_axis0)  # [1 2 3]

# Maximum
max_all = np.max(arr)
print(max_all)  # 7

max_axis0 = np.max(arr, axis=0)
print(max_axis0)  # [7 5 6]

# Peak to peak (max - min)
ptp = np.ptp(arr)
print(ptp)  # 6

ptp_axis0 = np.ptp(arr, axis=0)
print(ptp_axis0)  # [6 3 3]
```

### Percentile and Quantile

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Percentile (0-100 scale)
p50 = np.percentile(arr, 50)  # Median
print(p50)  # 5.5

p25 = np.percentile(arr, 25)
print(p25)  # 3.25

# Quantile (0-1 scale)
q50 = np.quantile(arr, 0.5)  # Median
print(q50)  # 5.5

q25 = np.quantile(arr, 0.25)
print(q25)  # 3.25

# Multiple percentiles
percentiles = np.percentile(arr, [25, 50, 75])
print(percentiles)  # [3.25 5.5  7.75]
```

### Argmin and Argmax

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# Index of minimum
min_idx = np.argmin(arr)
print(min_idx)  # 1

# Index of maximum
max_idx = np.argmax(arr)
print(max_idx)  # 5

# With axis
arr_2d = np.array([[3, 1, 4], [1, 5, 9]])
min_idx_axis0 = np.argmin(arr_2d, axis=0)
print(min_idx_axis0)  # [1 0 0] - row indices of minimums
```

### Cumulative Operations

```python
arr = np.array([1, 2, 3, 4, 5])

# Cumulative sum
cumsum = np.cumsum(arr)
print(cumsum)  # [ 1  3  6 10 15]

# Cumulative product
cumprod = np.cumprod(arr)
print(cumprod)  # [  1   2   6  24 120]

# With axis
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
cumsum_axis0 = np.cumsum(arr_2d, axis=0)
print(cumsum_axis0)
# [[1 2 3]
#  [5 7 9]]
```

### Difference and Gradient

```python
arr = np.array([1, 4, 6, 7, 12])

# First difference
diff = np.diff(arr)
print(diff)  # [3 2 1 5]

# Second difference
diff2 = np.diff(arr, n=2)
print(diff2)  # [-1 -1  4]

# Gradient (approximate derivative)
gradient = np.gradient(arr)
print(gradient)  # [3.  2.5 1.5 3.  5. ]

# Gradient for 2D
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
grad_y, grad_x = np.gradient(arr_2d)
print(grad_y)  # [[3. 3. 3.]
               #  [3. 3. 3.]]
print(grad_x)  # [[1. 1. 1.]
               #  [1. 1. 1.]]
```

### Histogram

```python
arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])

# 1D histogram
hist, bins = np.histogram(arr, bins=5)
print(hist)   # [1 2 3 2 1] - counts
print(bins)   # [1.  1.8 2.6 3.4 4.2 5. ] - bin edges

# 2D histogram
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])
hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=3)
print(hist_2d.shape)  # (3, 3)

# Multi-dimensional histogram
data = np.array([[1, 2], [2, 3], [3, 4]])
hist_dd, edges = np.histogramdd(data, bins=2)
```

### Axis Parameter Explained

The `axis` parameter determines along which dimension the operation is performed:

```python
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# Shape: (2, 2, 2)

# axis=0: along first dimension (depth)
result = np.sum(arr, axis=0)
# Shape: (2, 2)
# Result: [[6  8]
#          [10 12]]

# axis=1: along second dimension (rows)
result = np.sum(arr, axis=1)
# Shape: (2, 2)
# Result: [[ 4  6]
#          [12 14]]

# axis=2: along third dimension (columns)
result = np.sum(arr, axis=2)
# Shape: (2, 2)
# Result: [[ 3  7]
#          [11 15]]

# Multiple axes
result = np.sum(arr, axis=(0, 1))
# Shape: (2,)
# Result: [16 20]
```

### Keepdims Parameter

Preserves reduced dimensions:

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Without keepdims
result = np.sum(arr, axis=0)
print(result.shape)  # (3,)

# With keepdims
result = np.sum(arr, axis=0, keepdims=True)
print(result.shape)  # (1, 3)
print(result)  # [[5 7 9]]
```

---

## NaN-Safe Functions

NaN-safe functions ignore NaN values in calculations:

```python
arr = np.array([1, 2, np.nan, 4, 5])

# Regular sum (returns NaN if any NaN present)
regular_sum = np.sum(arr)
print(regular_sum)  # nan

# NaN-safe sum
nan_sum = np.nansum(arr)
print(nan_sum)  # 12.0

# NaN-safe mean
nan_mean = np.nanmean(arr)
print(nan_mean)  # 3.0

# NaN-safe standard deviation
nan_std = np.nanstd(arr)
print(nan_std)  # 1.5811388300841898

# NaN-safe minimum
nan_min = np.nanmin(arr)
print(nan_min)  # 1.0

# NaN-safe maximum
nan_max = np.nanmax(arr)
print(nan_max)  # 5.0

# NaN-safe median
nan_median = np.nanmedian(arr)
print(nan_median)  # 3.0
```

---

## Broadcasting

Broadcasting allows NumPy to perform operations on arrays of different shapes efficiently.

### Broadcasting Rules

NumPy compares dimensions from right to left (trailing dimensions first):

**Rule 1**: If arrays have different numbers of dimensions, prepend 1s to the shape of the array with fewer dimensions.

**Rule 2**: Arrays are compatible for broadcasting if, for each dimension, the sizes are equal or one of them is 1.

**Rule 3**: Arrays with size 1 in a dimension are "stretched" to match the size of the other array in that dimension.

### Visual Examples

#### Example 1: Scalar + Array

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
scalar = 10  # Shape: ()

# Broadcasting: scalar treated as (1, 1)
result = arr + scalar
print(result)
# [[11 12 13]
#  [14 15 16]]
```

#### Example 2: 1D + 2D

```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
arr_1d = np.array([10, 20, 30])            # Shape: (3,)

# Broadcasting: arr_1d treated as (1, 3), then stretched to (2, 3)
result = arr_2d + arr_1d
print(result)
# [[11 22 33]
#  [14 25 36]]
```

#### Example 3: 2D + 2D (Different Shapes)

```python
arr_a = np.array([[1], [2], [3]])  # Shape: (3, 1)
arr_b = np.array([10, 20, 30])     # Shape: (3,)

# arr_b treated as (1, 3)
# arr_a: (3, 1), arr_b: (1, 3) -> both stretched to (3, 3)
result = arr_a + arr_b
print(result)
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]
```

#### Example 4: 3D Broadcasting

```python
arr_3d = np.array([[[1, 2]], [[3, 4]]])  # Shape: (2, 1, 2)
arr_1d = np.array([10, 20])               # Shape: (2,)

# arr_1d treated as (1, 1, 2)
# Both compatible: (2, 1, 2) + (1, 1, 2) -> (2, 1, 2)
result = arr_3d + arr_1d
print(result)
# [[[11 22]]
#  [[13 24]]]
```

### Common Broadcasting Patterns

#### Pattern 1: Adding Bias Vector

```python
data = np.array([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
bias = np.array([0.1, 0.2, 0.3])         # Shape: (3,)

result = data + bias  # Broadcasting applies bias to each row
print(result)
# [[1.1 2.2 3.3]
#  [4.1 5.2 6.3]]
```

#### Pattern 2: Outer Product via Broadcasting

```python
a = np.array([1, 2, 3])  # Shape: (3,)
b = np.array([4, 5])     # Shape: (2,)

# Reshape for broadcasting
a_col = a[:, np.newaxis]  # Shape: (3, 1)
result = a_col + b         # Shape: (3, 2)
print(result)
# [[5 6]
#  [6 7]
#  [7 8]]
```

### Broadcasting Pitfalls

#### Pitfall 1: Incompatible Dimensions

```python
arr_a = np.array([[1, 2, 3]])  # Shape: (1, 3)
arr_b = np.array([[4], [5]])  # Shape: (2, 1)

# This works: (1, 3) + (2, 1) -> (2, 3)
result = arr_a + arr_b
print(result)
# [[5 6 7]
#  [6 7 8]]

# But this fails:
arr_c = np.array([4, 5, 6, 7])  # Shape: (4,)
# arr_a: (1, 3), arr_c: (4,) -> incompatible!
# result = arr_a + arr_c  # ValueError!
```

#### Pitfall 2: Unexpected Behavior

```python
# Be careful with 1D arrays
arr_1d = np.array([1, 2, 3])  # Shape: (3,)
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)

# This broadcasts correctly
result = arr_2d + arr_1d  # Works: (2, 3) + (3,) -> (2, 3)

# But this might not be what you expect:
arr_col = np.array([[1], [2]])  # Shape: (2, 1)
# arr_col + arr_1d broadcasts to (2, 3), not (2, 1)
```

### Broadcasting Functions

#### np.broadcast_to

Explicitly broadcasts array to specified shape:

```python
arr = np.array([1, 2, 3])  # Shape: (3,)

# Broadcast to (2, 3)
broadcasted = np.broadcast_to(arr, (2, 3))
print(broadcasted)
# [[1 2 3]
#  [1 2 3]]
```

#### np.broadcast_shapes

Determines output shape from broadcasting:

```python
shape1 = (2, 3)
shape2 = (3,)

output_shape = np.broadcast_shapes(shape1, shape2)
print(output_shape)  # (2, 3)
```

#### np.broadcast_arrays

Broadcasts multiple arrays to same shape:

```python
arr1 = np.array([[1], [2]])  # Shape: (2, 1)
arr2 = np.array([10, 20, 30])  # Shape: (3,)

broadcasted1, broadcasted2 = np.broadcast_arrays(arr1, arr2)
print(broadcasted1.shape)  # (2, 3)
print(broadcasted2.shape)  # (2, 3)
```

---

## Clipping Operations

Clipping constrains values to a specified range:

```python
arr = np.array([1, 5, 10, 15, 20])

# Clip values between 5 and 15
clipped = np.clip(arr, 5, 15)
print(clipped)  # [ 5  5 10 15 15]

# With minimum and maximum arrays
min_vals = np.array([2, 6, 11, 16, 21])
max_vals = np.array([4, 8, 13, 18, 23])
clipped = np.clip(arr, min_vals, max_vals)
print(clipped)  # [ 4  6 11 16 20]
```

### Minimum and Maximum as Clipping

```python
arr = np.array([1, 5, 10, 15, 20])

# Element-wise minimum (clips from above)
result = np.minimum(arr, 12)
print(result)  # [ 1  5 10 12 12]

# Element-wise maximum (clips from below)
result = np.maximum(arr, 8)
print(result)  # [ 8  8 10 15 20]

# Combined clipping
result = np.maximum(np.minimum(arr, 15), 5)
print(result)  # [ 5  5 10 15 15]  # Same as clip(arr, 5, 15)
```

---

## Complex Number Operations

NumPy provides functions for working with complex numbers:

```python
arr = np.array([1+2j, 3+4j, 5+6j])

# Real part
real = np.real(arr)
print(real)  # [1. 3. 5.]

# Imaginary part
imag = np.imag(arr)
print(imag)  # [2. 4. 6.]

# Complex conjugate
conj = np.conj(arr)
print(conj)  # [1.-2.j 3.-4.j 5.-6.j]

# Angle (phase) in radians
angle = np.angle(arr)
print(angle)  # [1.10714872 0.92729522 0.87605805]

# Angle in degrees
angle_deg = np.angle(arr, deg=True)
print(angle_deg)  # [63.43494882 53.13010235 50.19442891]

# Absolute value (magnitude) for complex numbers
magnitude = np.abs(arr)
print(magnitude)  # [2.23606798 5.         7.81024968]
```

### Complex Arithmetic

```python
a = np.array([1+2j, 3+4j])
b = np.array([5+6j, 7+8j])

# Addition
sum_complex = a + b
print(sum_complex)  # [ 6. +8.j 10.+12.j]

# Multiplication
prod_complex = a * b
print(prod_complex)  # [-7.+16.j -11.+52.j]

# Division
div_complex = a / b
print(div_complex)  # [0.26+0.03j 0.44+0.08j]
```

These mathematical and statistical operations, combined with broadcasting, form the foundation of efficient numerical computing in NumPy, enabling complex data analysis and scientific computations.
