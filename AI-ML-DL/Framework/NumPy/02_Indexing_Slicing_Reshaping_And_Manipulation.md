# NumPy Indexing, Slicing, Reshaping, and Manipulation

## Table of Contents
1. [Basic Indexing](#basic-indexing)
2. [Slicing](#slicing)
3. [Advanced/Fancy Indexing](#advancedfancy-indexing)
4. [Boolean Indexing](#boolean-indexing)
5. [Combined Indexing](#combined-indexing)
6. [Take and Put Functions](#take-and-put-functions)
7. [Reshaping Operations](#reshaping-operations)
8. [Stacking and Joining](#stacking-and-joining)
9. [Splitting](#splitting)
10. [Adding/Removing Elements](#addingremoving-elements)
11. [Sorting](#sorting)
12. [Searching](#searching)
13. [Set Operations](#set-operations)
14. [Tiling and Repeating](#tiling-and-repeating)
15. [Flipping and Rotating](#flipping-and-rotating)

---

## Basic Indexing

Basic indexing accesses single elements or subarrays using integer indices.

### Single Element Indexing

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])
print(arr[0])   # 10
print(arr[2])   # 30
print(arr[-1])  # 50 (negative indexing from end)
```

### Multi-Dimensional Indexing

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr[0, 0])  # 1
print(arr[1, 2])  # 6
print(arr[-1, -1])  # 9

# Can also use tuple
print(arr[(0, 0)])  # 1
```

### Negative Indices

Negative indices count from the end of the array.

```python
arr = np.array([10, 20, 30, 40, 50])
print(arr[-1])  # 50
print(arr[-2])  # 40

arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr_2d[-1, -1])  # 6
```

---

## Slicing

Slicing extracts subarrays using the `start:stop:step` notation.

### Basic Slicing Syntax

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# start:stop
print(arr[2:6])      # [2 3 4 5]

# start:stop:step
print(arr[0:10:2])   # [0 2 4 6 8]

# Omit start (defaults to 0)
print(arr[:5])       # [0 1 2 3 4]

# Omit stop (defaults to end)
print(arr[5:])       # [5 6 7 8 9]

# Omit both (entire array)
print(arr[:])        # [0 1 2 3 4 5 6 7 8 9]

# Negative step (reverse)
print(arr[::-1])     # [9 8 7 6 5 4 3 2 1 0]
```

### Multi-Dimensional Slicing

```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Row slicing
print(arr[1:3])      # [[ 5  6  7  8]
                     #  [ 9 10 11 12]]

# Column slicing
print(arr[:, 1:3])   # [[ 2  3]
                     #  [ 6  7]
                     #  [10 11]]

# Both dimensions
print(arr[0:2, 1:3]) # [[2 3]
                     #  [6 7]]
```

### Ellipsis (...)

Ellipsis expands to fill all remaining dimensions.

```python
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Equivalent ways to access first element of each subarray
print(arr[0, :, :])  # [[1 2]
                     #  [3 4]]
print(arr[0, ...])   # Same as above
print(arr[..., 0])   # [[1 3]
                     #  [5 7]]
```

### np.s_

`np.s_` creates slice objects programmatically.

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Create slice object
s = np.s_[2:8:2]
print(arr[s])  # [2 4 6]

# Useful for dynamic slicing
start, stop, step = 1, 9, 2
print(arr[np.s_[start:stop:step]])  # [1 3 5 7]
```

---

## Advanced/Fancy Indexing

Fancy indexing uses integer arrays to select elements. **Always returns a copy**, not a view.

### Integer Array Indexing

```python
arr = np.array([10, 20, 30, 40, 50])

# Single array of indices
indices = np.array([0, 2, 4])
print(arr[indices])  # [10 30 50]

# Using list
print(arr[[0, 2, 4]])  # [10 30 50]
```

### Multi-Dimensional Fancy Indexing

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Select specific rows
print(arr[[0, 2]])    # [[1 2 3]
                      #  [7 8 9]]

# Select specific elements
print(arr[[0, 1, 2], [0, 1, 2]])  # [1 5 9] (diagonal)

# Select rectangular region
print(arr[np.ix_([0, 2], [0, 2])])  # [[1 3]
                                    #  [7 9]]
```

### Fancy Indexing Always Returns Copy

```python
arr = np.array([10, 20, 30, 40, 50])
fancy = arr[[0, 2, 4]]
fancy[0] = 99
print(arr)  # [10 20 30 40 50] - original unchanged (copy)
```

---

## Boolean Indexing

Boolean indexing uses boolean arrays to select elements.

### Boolean Masks

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Create boolean mask
mask = arr > 5
print(mask)  # [False False False False False  True  True  True  True  True]

# Apply mask
print(arr[mask])  # [ 6  7  8  9 10]

# Inline
print(arr[arr > 5])  # [ 6  7  8  9 10]
```

### Multiple Conditions

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# AND: use & (not 'and')
mask = (arr > 3) & (arr < 8)
print(arr[mask])  # [4 5 6 7]

# OR: use | (not 'or')
mask = (arr < 3) | (arr > 8)
print(arr[mask])  # [ 1  2  9 10]

# NOT: use ~ (not 'not')
mask = ~(arr > 5)
print(arr[mask])  # [1 2 3 4 5]
```

### np.where

Returns indices where condition is True, or values based on condition.

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Get indices where condition is True
indices = np.where(arr > 5)
print(indices)  # (array([5, 6, 7, 8, 9]),)

# Conditional assignment
result = np.where(arr > 5, arr, -1)
print(result)  # [-1 -1 -1 -1 -1  6  7  8  9 10]

# Three-argument form: condition, value_if_true, value_if_false
result = np.where(arr > 5, arr * 2, arr)
print(result)  # [ 1  2  3  4  5 12 14 16 18 20]
```

### np.nonzero

Returns indices of non-zero elements (or non-False for boolean).

```python
arr = np.array([0, 1, 0, 2, 0, 3, 0])
indices = np.nonzero(arr)
print(indices)  # (array([1, 3, 5]),)
print(arr[np.nonzero(arr)])  # [1 2 3]
```

### np.argwhere

Returns indices as array of shape (N, ndim) where N is number of True elements.

```python
arr = np.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]])
indices = np.argwhere(arr > 0)
print(indices)
# [[0 0]
#  [0 2]
#  [1 1]
#  [2 0]
#  [2 2]]
```

---

## Combined Indexing

You can combine basic indexing, slicing, and fancy indexing.

```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Slice + fancy indexing
print(arr[0:2, [0, 2]])  # [[1 3]
                         #  [5 7]]

# Fancy + slice
print(arr[[0, 2], 1:3])  # [[ 2  3]
                         #  [10 11]]

# Boolean + slice
mask = np.array([True, False, True])
print(arr[mask, 1:3])    # [[ 2  3]
                         #  [10 11]]
```

---

## Take and Put Functions

### np.take

Takes elements along a flattened version of the array.

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Take elements by flat indices
print(np.take(arr, [0, 4, 8]))  # [1 5 9]

# With axis
print(np.take(arr, [0, 2], axis=1))  # [[1 3]
                                     #  [4 6]
                                     #  [7 9]]
```

### np.take_along_axis

Takes values from array using indices array along specified axis.

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
indices = np.array([[2, 0, 1], [1, 2, 0], [0, 1, 2]])

result = np.take_along_axis(arr, indices, axis=1)
print(result)
# [[3 1 2]
#  [5 6 4]
#  [7 8 9]]
```

### np.put

Puts values into array at flat indices.

```python
arr = np.array([1, 2, 3, 4, 5])
np.put(arr, [0, 2, 4], [10, 30, 50])
print(arr)  # [10  2 30  4 50]
```

### np.put_along_axis

Puts values into array using indices along specified axis.

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
indices = np.array([[0], [1], [2]])
values = np.array([[10], [50], [90]])

np.put_along_axis(arr, indices, values, axis=1)
print(arr)
# [[10  2  3]
#  [ 4 50  6]
#  [ 7  8 90]]
```

### np.choose

Chooses elements from arrays based on indices.

```python
choices = np.array([[0, 1, 2], [10, 11, 12], [20, 21, 22]])
indices = np.array([0, 1, 2])

result = np.choose(indices, choices)
print(result)  # [ 0 11 22]
```

---

## Reshaping Operations

### reshape

Returns array with new shape (doesn't modify original if possible).

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# Reshape to 2x3
reshaped = arr.reshape(2, 3)
print(reshaped)
# [[1 2 3]
#  [4 5 6]]

# Auto-dimension with -1
reshaped = arr.reshape(2, -1)  # -1 inferred as 3
print(reshaped.shape)  # (2, 3)

# Flatten with -1
reshaped = arr.reshape(-1)  # Flattens to 1D
print(reshaped)  # [1 2 3 4 5 6]
```

### resize

Resizes array in-place (modifies original).

```python
arr = np.array([1, 2, 3, 4, 5, 6])
arr.resize(2, 3)  # Modifies arr
print(arr)
# [[1 2 3]
#  [4 5 6]]

# Can also use np.resize (returns new array)
arr = np.array([1, 2, 3, 4])
new_arr = np.resize(arr, (2, 3))  # Repeats if needed
print(new_arr)
# [[1 2 3]
#  [4 1 2]]
```

### ravel

Returns flattened view (if possible) or copy.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
flat = arr.ravel()
print(flat)  # [1 2 3 4 5 6]

# Modifying ravel may modify original (if view)
flat[0] = 99
print(arr)  # [[99  2  3]
            #  [ 4  5  6]]
```

### flatten

Returns flattened copy (always).

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
flat = arr.flatten()
print(flat)  # [1 2 3 4 5 6]

# Modifying flatten never modifies original
flat[0] = 99
print(arr)  # [[1 2 3]
            #  [4 5 6]]
```

### transpose

Transposes array (swaps dimensions).

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.transpose())
# [[1 4]
#  [2 5]
#  [3 6]]

# Shorthand
print(arr.T)
# [[1 4]
#  [2 5]
#  [3 6]]

# Multi-dimensional transpose
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(arr_3d.transpose(2, 0, 1))  # Permute axes
```

### swapaxes

Swaps two axes.

```python
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
swapped = arr.swapaxes(0, 2)
print(swapped.shape)  # (2, 2, 2) - swapped axes 0 and 2
```

### moveaxis

Moves axes to new positions.

```python
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
moved = np.moveaxis(arr, 0, -1)  # Move axis 0 to end
print(moved.shape)  # (2, 2, 2)
```

### rollaxis

Rolls axis backward (deprecated, use moveaxis).

```python
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
rolled = np.rollaxis(arr, 0, 3)  # Roll axis 0 to position 3
```

### np.expand_dims

Adds new dimension at specified position.

```python
arr = np.array([1, 2, 3])

# Add dimension at axis 0
expanded = np.expand_dims(arr, axis=0)
print(expanded.shape)  # (1, 3)

# Add dimension at axis 1
expanded = np.expand_dims(arr, axis=1)
print(expanded.shape)  # (3, 1)
```

### np.squeeze

Removes dimensions of size 1.

```python
arr = np.array([[[1, 2, 3]]])  # Shape (1, 1, 3)

# Remove all size-1 dimensions
squeezed = np.squeeze(arr)
print(squeezed.shape)  # (3,)

# Remove specific axis
squeezed = np.squeeze(arr, axis=0)
print(squeezed.shape)  # (1, 3)
```

### np.newaxis

Alias for None, used to add new axis.

```python
arr = np.array([1, 2, 3])

# Add dimension
row_vec = arr[np.newaxis, :]
print(row_vec.shape)  # (1, 3)

col_vec = arr[:, np.newaxis]
print(col_vec.shape)  # (3, 1)
```

---

## Stacking and Joining

### np.concatenate

Joins arrays along existing axis.

```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# Concatenate along axis 0 (rows)
result = np.concatenate([arr1, arr2], axis=0)
print(result)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# Concatenate along axis 1 (columns)
result = np.concatenate([arr1, arr2], axis=1)
print(result)
# [[1 2 5 6]
#  [3 4 7 8]]
```

### np.stack

Stacks arrays along new axis.

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Stack along new axis 0
result = np.stack([arr1, arr2], axis=0)
print(result)
# [[1 2 3]
#  [4 5 6]]

# Stack along new axis 1
result = np.stack([arr1, arr2], axis=1)
print(result)
# [[1 4]
#  [2 5]
#  [3 6]]
```

### np.vstack

Stacks arrays vertically (row-wise).

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

result = np.vstack([arr1, arr2])
print(result)
# [[1 2 3]
#  [4 5 6]]

# Works with 2D arrays too
arr1_2d = np.array([[1, 2], [3, 4]])
arr2_2d = np.array([[5, 6]])
result = np.vstack([arr1_2d, arr2_2d])
```

### np.hstack

Stacks arrays horizontally (column-wise).

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

result = np.hstack([arr1, arr2])
print(result)  # [1 2 3 4 5 6]

# Works with 2D arrays
arr1_2d = np.array([[1], [2]])
arr2_2d = np.array([[3], [4]])
result = np.hstack([arr1_2d, arr2_2d])
print(result)
# [[1 3]
#  [2 4]]
```

### np.dstack

Stacks arrays depth-wise (along third axis).

```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

result = np.dstack([arr1, arr2])
print(result.shape)  # (2, 2, 2)
print(result)
# [[[1 5]
#   [2 6]]
#  [[3 7]
#   [4 8]]]
```

### np.column_stack

Stacks 1D arrays as columns.

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

result = np.column_stack([arr1, arr2])
print(result)
# [[1 4]
#  [2 5]
#  [3 6]]
```

### np.row_stack

Stacks arrays as rows (same as vstack).

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

result = np.row_stack([arr1, arr2])
print(result)
# [[1 2 3]
#  [4 5 6]]
```

### np.block

Assembles arrays from blocks.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5], [6]])
C = np.array([[7, 8]])
D = np.array([[9]])

result = np.block([[A, B], [C, D]])
print(result)
# [[1 2 5]
#  [3 4 6]
#  [7 8 9]]
```

---

## Splitting

### np.split

Splits array into multiple sub-arrays.

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# Split into 3 equal parts
result = np.split(arr, 3)
print(result)
# [array([1, 2]), array([3, 4]), array([5, 6])]

# Split at specific indices
result = np.split(arr, [2, 4])
print(result)
# [array([1, 2]), array([3, 4]), array([5, 6])]
```

### np.hsplit

Splits array horizontally (column-wise).

```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# Split into 2 equal parts
result = np.hsplit(arr, 2)
print(result)
# [array([[1, 2],
#         [5, 6]]),
#  array([[3, 4],
#         [7, 8]])]
```

### np.vsplit

Splits array vertically (row-wise).

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Split into 3 equal parts
result = np.vsplit(arr, 3)
print(result)
# [array([[1, 2, 3]]),
#  array([[4, 5, 6]]),
#  array([[7, 8, 9]])]
```

### np.dsplit

Splits array depth-wise (along third axis).

```python
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

result = np.dsplit(arr, 2)
print(result)
# [array([[[1],
#          [3]],
#         [[5],
#          [7]]]),
#  array([[[2],
#          [4]],
#         [[6],
#          [8]]])]
```

### np.array_split

Splits array into sub-arrays (allows unequal division).

```python
arr = np.array([1, 2, 3, 4, 5])

# Split into 3 parts (unequal)
result = np.array_split(arr, 3)
print(result)
# [array([1, 2]), array([3, 4]), array([5])]
```

---

## Adding/Removing Elements

### np.append

Appends values to array (returns new array).

```python
arr = np.array([1, 2, 3])

# Append single value
result = np.append(arr, 4)
print(result)  # [1 2 3 4]

# Append multiple values
result = np.append(arr, [4, 5, 6])
print(result)  # [1 2 3 4 5 6]

# Append along axis
arr_2d = np.array([[1, 2], [3, 4]])
result = np.append(arr_2d, [[5, 6]], axis=0)
print(result)
# [[1 2]
#  [3 4]
#  [5 6]]
```

### np.insert

Inserts values before given indices.

```python
arr = np.array([1, 2, 3, 4, 5])

# Insert at index 2
result = np.insert(arr, 2, 99)
print(result)  # [ 1  2 99  3  4  5]

# Insert multiple values
result = np.insert(arr, [1, 3], [10, 20])
print(result)  # [ 1 10  2  3 20  4  5]

# Insert along axis
arr_2d = np.array([[1, 2], [3, 4]])
result = np.insert(arr_2d, 1, [5, 6], axis=0)
print(result)
# [[1 2]
#  [5 6]
#  [3 4]]
```

### np.delete

Deletes elements at given indices.

```python
arr = np.array([1, 2, 3, 4, 5])

# Delete at index 2
result = np.delete(arr, 2)
print(result)  # [1 2 4 5]

# Delete multiple indices
result = np.delete(arr, [0, 2, 4])
print(result)  # [2 4]

# Delete along axis
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = np.delete(arr_2d, 1, axis=0)  # Delete row 1
print(result)
# [[1 2 3]
#  [7 8 9]]
```

### np.unique

Finds unique elements, returns sorted unique array.

```python
arr = np.array([3, 1, 2, 2, 3, 1, 4])

# Get unique values
unique = np.unique(arr)
print(unique)  # [1 2 3 4]

# Get unique with indices
unique, indices = np.unique(arr, return_index=True)
print(indices)  # [1 2 0 6] - first occurrence of each unique value

# Get unique with counts
unique, counts = np.unique(arr, return_counts=True)
print(counts)  # [2 2 2 1] - count of each unique value

# Get inverse indices
unique, inverse = np.unique(arr, return_inverse=True)
print(inverse)  # [2 0 1 1 2 0 3] - indices to reconstruct original
```

---

## Sorting

### np.sort

Returns sorted copy of array.

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# Sort ascending
sorted_arr = np.sort(arr)
print(sorted_arr)  # [1 1 2 3 4 5 6 9]

# Sort descending
sorted_arr = np.sort(arr)[::-1]
print(sorted_arr)  # [9 6 5 4 3 2 1 1]

# Sort along axis
arr_2d = np.array([[3, 1, 4], [1, 5, 9]])
sorted_arr = np.sort(arr_2d, axis=1)
print(sorted_arr)
# [[1 3 4]
#  [1 5 9]]
```

### np.argsort

Returns indices that would sort array.

```python
arr = np.array([3, 1, 4, 1, 5])

indices = np.argsort(arr)
print(indices)  # [1 3 0 2 4] - indices to sort arr

# Use indices to sort
print(arr[indices])  # [1 1 3 4 5]
```

### np.lexsort

Sorts using multiple keys (lexicographic sort).

```python
first_names = np.array(['Alice', 'Bob', 'Charlie'])
ages = np.array([25, 30, 25])
salaries = np.array([50000, 60000, 55000])

# Sort by age, then by salary
indices = np.lexsort((salaries, ages))
print(indices)  # [0 2 1] - Alice and Charlie same age, sorted by salary
```

### np.partition

Partially sorts array (elements smaller than k-th element come before, larger after).

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# Partition around 4th element (0-indexed)
partitioned = np.partition(arr, 4)
print(partitioned)  # [1 1 2 3 4 9 5 6] - first 4 are smallest
```

### np.argpartition

Returns indices that would partition array.

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

indices = np.argpartition(arr, 4)
print(indices)  # [1 3 6 0 2 7 4 5]
```

### Sort Algorithms

NumPy supports different sorting algorithms:

- **quicksort**: Fast, not stable, O(n log n) average
- **mergesort**: Stable, O(n log n) worst case
- **heapsort**: O(n log n) worst case, not stable
- **stable**: Alias for mergesort

```python
arr = np.array([3, 1, 4, 1, 5])

# Use mergesort (stable)
sorted_arr = np.sort(arr, kind='mergesort')

# Use quicksort (default)
sorted_arr = np.sort(arr, kind='quicksort')
```

---

## Searching

### np.searchsorted

Finds indices where elements should be inserted to maintain order.

```python
arr = np.array([1, 3, 5, 7, 9])

# Find insertion point for value 4
idx = np.searchsorted(arr, 4)
print(idx)  # 2 - insert at index 2 to maintain order

# With side='right'
idx = np.searchsorted(arr, 5, side='right')
print(idx)  # 3 - rightmost insertion point

# Multiple values
indices = np.searchsorted(arr, [2, 4, 6, 8])
print(indices)  # [1 2 3 4]
```

### np.digitize

Returns indices of bins to which each value belongs.

```python
arr = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
bins = np.array([1, 2, 3, 4])

indices = np.digitize(arr, bins)
print(indices)  # [0 1 2 3 4] - bin indices for each value
```

### np.bincount

Counts occurrences of each value in array of non-negative integers.

```python
arr = np.array([0, 1, 1, 2, 2, 2, 3])

counts = np.bincount(arr)
print(counts)  # [1 2 3 1] - count of 0, 1, 2, 3

# With weights
weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
weighted_counts = np.bincount(arr, weights=weights)
print(weighted_counts)  # [0.1 0.5 1.5 0.7] - weighted sums
```

---

## Set Operations

### np.union1d

Finds union of two arrays.

```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])

union = np.union1d(arr1, arr2)
print(union)  # [1 2 3 4 5 6]
```

### np.intersect1d

Finds intersection of two arrays.

```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])

intersection = np.intersect1d(arr1, arr2)
print(intersection)  # [3 4]
```

### np.setdiff1d

Finds set difference (elements in arr1 but not in arr2).

```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])

diff = np.setdiff1d(arr1, arr2)
print(diff)  # [1 2]
```

### np.setxor1d

Finds symmetric difference (elements in either arr1 or arr2, but not both).

```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])

xor = np.setxor1d(arr1, arr2)
print(xor)  # [1 2 5 6]
```

### np.in1d / np.isin

Tests whether each element of first array is in second array.

```python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([2, 4, 6])

mask = np.in1d(arr1, arr2)
print(mask)  # [False  True False  True False]

# np.isin is alias (preferred)
mask = np.isin(arr1, arr2)
print(mask)  # [False  True False  True False]
```

---

## Tiling and Repeating

### np.tile

Constructs array by repeating input array.

```python
arr = np.array([1, 2, 3])

# Tile 2 times
tiled = np.tile(arr, 2)
print(tiled)  # [1 2 3 1 2 3]

# Tile in 2D pattern
tiled = np.tile(arr, (2, 3))
print(tiled)
# [[1 2 3 1 2 3 1 2 3]
#  [1 2 3 1 2 3 1 2 3]]
```

### np.repeat

Repeats elements of array.

```python
arr = np.array([1, 2, 3])

# Repeat each element 2 times
repeated = np.repeat(arr, 2)
print(repeated)  # [1 1 2 2 3 3]

# Repeat with different counts
repeated = np.repeat(arr, [2, 3, 1])
print(repeated)  # [1 1 2 2 2 3]

# Repeat along axis
arr_2d = np.array([[1, 2], [3, 4]])
repeated = np.repeat(arr_2d, 2, axis=0)
print(repeated)
# [[1 2]
#  [1 2]
#  [3 4]
#  [3 4]]
```

---

## Flipping and Rotating

### np.flip

Reverses array along specified axis.

```python
arr = np.array([1, 2, 3, 4, 5])

# Flip along axis 0
flipped = np.flip(arr)
print(flipped)  # [5 4 3 2 1]

# Flip 2D array along axis 0
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
flipped = np.flip(arr_2d, axis=0)
print(flipped)
# [[4 5 6]
#  [1 2 3]]

# Flip along multiple axes
flipped = np.flip(arr_2d, axis=(0, 1))
print(flipped)
# [[6 5 4]
#  [3 2 1]]
```

### np.fliplr

Flips array left-right (along axis 1).

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

flipped = np.fliplr(arr)
print(flipped)
# [[3 2 1]
#  [6 5 4]]
```

### np.flipud

Flips array up-down (along axis 0).

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

flipped = np.flipud(arr)
print(flipped)
# [[4 5 6]
#  [1 2 3]]
```

### np.rot90

Rotates array 90 degrees counter-clockwise.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Rotate 90 degrees (default)
rotated = np.rot90(arr)
print(rotated)
# [[3 6]
#  [2 5]
#  [1 4]]

# Rotate 180 degrees
rotated = np.rot90(arr, k=2)
print(rotated)
# [[6 5 4]
#  [3 2 1]]

# Rotate 270 degrees (or -90)
rotated = np.rot90(arr, k=3)
```

### np.roll

Rolls array elements along axis.

```python
arr = np.array([1, 2, 3, 4, 5])

# Roll 2 positions to the right
rolled = np.roll(arr, 2)
print(rolled)  # [4 5 1 2 3]

# Roll along axis
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
rolled = np.roll(arr_2d, 1, axis=0)
print(rolled)
# [[4 5 6]
#  [1 2 3]]
```

These operations provide comprehensive tools for manipulating NumPy arrays, enabling efficient data transformation and analysis workflows.
