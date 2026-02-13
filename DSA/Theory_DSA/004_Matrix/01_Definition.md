# Matrix Definition and Fundamentals

## 2D Array as Matrix

A matrix is a rectangular arrangement of numbers (or elements) in rows and columns. An m x n matrix has m rows and n columns. Element at position (i, j) is in row i and column j (0-indexed).

```
Matrix M (3 x 4):
     col0  col1  col2  col3
row0  a00   a01   a02   a03
row1  a10   a11   a12   a13
row2  a20   a21   a22   a23
```

## Row-Major vs Column-Major Storage

### Row-Major Order

Elements are stored row by row. For matrix M[m][n], element M[i][j] is at linear index: i * n + j.

```
Storage: [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23]
```

Used in: C, C++, Python (list of lists), most programming languages.

### Column-Major Order

Elements are stored column by column. Element M[i][j] is at linear index: j * m + i.

```
Storage: [a00, a10, a20, a01, a11, a21, a02, a12, a22, a03, a13, a23]
```

Used in: Fortran, MATLAB, R, Julia.

### Cache Implications

Row-major: Sequential access along rows is cache-friendly. Column traversal causes cache misses.
Column-major: Sequential access along columns is cache-friendly. Row traversal causes cache misses.

## Sparse Matrix Representations

A sparse matrix has most elements zero. Storing all elements wastes space. Special representations store only non-zero entries.

### Triplet Representation (COO - Coordinate Format)

Store (row, col, value) for each non-zero element.

```
Matrix:         Triplet:
0 0 3 0         (0, 2, 3)
0 0 0 4         (1, 3, 4)
5 0 0 0         (2, 0, 5)
```

```python
def triplet_to_matrix(rows, cols, triplets):
    matrix = [[0] * cols for _ in range(rows)]
    for r, c, v in triplets:
        matrix[r][c] = v
    return matrix

def matrix_to_triplet(matrix):
    triplets = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] != 0:
                triplets.append((i, j, matrix[i][j]))
    return triplets
```

### CSR (Compressed Sparse Row)

Three arrays: values (non-zero values), col_ind (column index of each value), row_ptr (cumulative count of non-zeros per row).

```
Matrix:         values:  [3, 4, 5]
0 0 3 0         col_ind: [2, 3, 0]
0 0 0 4         row_ptr: [0, 1, 2, 3]
5 0 0 0
```

```python
def matrix_to_csr(matrix):
    values, col_ind, row_ptr = [], [], [0]
    for row in matrix:
        for j, v in enumerate(row):
            if v != 0:
                values.append(v)
                col_ind.append(j)
        row_ptr.append(len(values))
    return values, col_ind, row_ptr

def csr_to_matrix(values, col_ind, row_ptr, cols):
    matrix = []
    for i in range(len(row_ptr) - 1):
        row = [0] * cols
        for k in range(row_ptr[i], row_ptr[i + 1]):
            row[col_ind[k]] = values[k]
        matrix.append(row)
    return matrix
```

### CSC (Compressed Sparse Column)

Same idea as CSR but column-wise: values, row_ind, col_ptr.

```python
def matrix_to_csc(matrix):
    values, row_ind, col_ptr = [], [], [0]
    for j in range(len(matrix[0])):
        for i in range(len(matrix)):
            if matrix[i][j] != 0:
                values.append(matrix[i][j])
                row_ind.append(i)
        col_ptr.append(len(values))
    return values, row_ind, col_ptr
```

## Special Matrix Types

### Identity Matrix

Square matrix with 1s on main diagonal, 0s elsewhere. I[i][i] = 1, I[i][j] = 0 for i != j.

### Symmetric Matrix

M[i][j] = M[j][i] for all i, j. Square matrix equal to its transpose.

### Diagonal Matrix

Non-zero elements only on main diagonal. M[i][j] = 0 for i != j.

### Triangular Matrix

**Upper triangular**: All elements below main diagonal are zero. M[i][j] = 0 for i > j.
**Lower triangular**: All elements above main diagonal are zero. M[i][j] = 0 for i < j.

## Time Complexity of Matrix Operations

| Operation | Time | Space |
|-----------|------|-------|
| Access M[i][j] | O(1) | O(1) |
| Traverse all | O(m * n) | O(1) |
| Transpose | O(m * n) | O(m * n) for new matrix |
| Transpose in-place (square) | O(m * n) | O(1) |
| Matrix addition | O(m * n) | O(m * n) |
| Matrix multiplication (naive) | O(m * n * p) for A(m,n) * B(n,p) | O(m * p) |
| Matrix multiplication (Strassen) | O(n^2.807) | O(n^2) |
| Search in sorted matrix | O(m + n) staircase | O(1) |

## When to Use Matrix vs 1D Array vs Graph

| Use Case | Matrix | 1D Array | Graph |
|----------|--------|----------|-------|
| 2D grid (image, board) | Yes | Flatten with index mapping | No |
| Adjacency of nodes | Adjacency matrix | No | Adjacency list preferred |
| Linear algebra | Yes | No | No |
| Tabular data (rows/cols) | Yes | Awkward | No |
| Sparse connections | Sparse matrix or graph | No | Yes |
| Sequential data | No | Yes | No |
| Relationships between entities | Small dense | No | Yes |

**Use Matrix when**: 2D spatial structure, linear algebra, dense 2D data, grid-based problems.

**Use 1D Array when**: Sequential data, flattenable structure, no 2D semantics.

**Use Graph when**: Sparse connections, traversal/connectivity, relationships matter more than coordinates.

## Python Representation

### List of Lists

```python
def create_matrix_list(m, n, default=0):
    return [[default] * n for _ in range(m)]

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(matrix[1][2])
```

Warning: `[[0] * n] * m` creates m references to the same row. Use list comprehension.

### NumPy Overview

```python
import numpy as np

arr = np.array([[1, 2], [3, 4]])
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
identity = np.eye(3)
random = np.random.rand(2, 2)

transpose = arr.T
matmul = np.dot(arr, arr)
```

NumPy provides: vectorized operations, efficient storage, BLAS/LAPACK for linear algebra, broadcasting.
