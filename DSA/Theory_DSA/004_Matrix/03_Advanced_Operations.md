# Advanced Matrix Operations

## Transpose In-Place (Square Matrix)

Swap elements above diagonal with below. Only process upper triangle to avoid double-swapping.

```python
def transpose_in_place(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
```

## Set Row and Column to Zero (O(1) Space)

Use first row and first column as markers. If matrix[i][j] == 0, set matrix[i][0] = 0 and matrix[0][j] = 0. Handle (0,0) overlap with separate flag.

```python
def set_zeroes(matrix):
    if not matrix or not matrix[0]:
        return
    m, n = len(matrix), len(matrix[0])
    first_row_zero = any(matrix[0][j] == 0 for j in range(n))
    first_col_zero = any(matrix[i][0] == 0 for i in range(m))
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0
    for i in range(1, m):
        if matrix[i][0] == 0:
            for j in range(1, n):
                matrix[i][j] = 0
    for j in range(1, n):
        if matrix[0][j] == 0:
            for i in range(1, m):
                matrix[i][j] = 0
    if first_row_zero:
        for j in range(n):
            matrix[0][j] = 0
    if first_col_zero:
        for i in range(m):
            matrix[i][0] = 0
```

## Rotate 90 Clockwise In-Place

Transpose then reverse each row. For square matrix.

```python
def rotate_90_clockwise_in_place(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for i in range(n):
        matrix[i].reverse()
```

## Rotate 90 Counter-Clockwise In-Place

Transpose then reverse each column (or reverse rows then transpose).

```python
def rotate_90_counter_clockwise_in_place(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for j in range(n):
        for i in range(n // 2):
            matrix[i][j], matrix[n - 1 - i][j] = matrix[n - 1 - i][j], matrix[i][j]
```

## Rotate 180

Reverse each row then reverse row order (or swap symmetric elements).

```python
def rotate_180(matrix):
    m = len(matrix)
    for i in range(m // 2):
        matrix[i], matrix[m - 1 - i] = matrix[m - 1 - i], matrix[i]
    for row in matrix:
        row.reverse()
```

## Reflect Horizontally

Reverse each row.

```python
def reflect_horizontal(matrix):
    for row in matrix:
        row.reverse()
```

## Reflect Vertically

Reverse row order.

```python
def reflect_vertical(matrix):
    matrix.reverse()
```

## Matrix Exponentiation (Fast Power)

Compute A^k using binary exponentiation. A^k = (A^(k/2))^2 when k even, A * A^(k-1) when k odd.

```python
def matrix_multiply_square(A, B):
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matrix_power(matrix, k):
    n = len(matrix)
    if k == 0:
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    if k == 1:
        return [row[:] for row in matrix]
    half = matrix_power(matrix, k // 2)
    result = matrix_multiply_square(half, half)
    if k % 2 == 1:
        result = matrix_multiply_square(result, matrix)
    return result
```

## Sparse Matrix Multiplication

Multiply two sparse matrices. For each (i, k) in A and (k, j) in B, add A[i][k]*B[k][j] to C[i][j]. Use hash map or pre-index non-zeros.

```python
def sparse_multiply(A, B):
    def get_nonzeros(M):
        d = {}
        for i in range(len(M)):
            for j in range(len(M[0])):
                if M[i][j] != 0:
                    d[(i, j)] = M[i][j]
        return d
    a_nz = get_nonzeros(A)
    b_nz = get_nonzeros(B)
    m, n, p = len(A), len(A[0]), len(B[0])
    result = [[0] * p for _ in range(m)]
    for (i, k), v1 in a_nz.items():
        for j in range(p):
            if (k, j) in b_nz:
                result[i][j] += v1 * b_nz[(k, j)]
    return result
```

## Toeplitz Matrix Check

A matrix is Toeplitz if every diagonal from top-left to bottom-right has the same elements. M[i][j] == M[i-1][j-1] for i,j >= 1.

```python
def is_toeplitz(matrix):
    if not matrix or not matrix[0]:
        return True
    m, n = len(matrix), len(matrix[0])
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] != matrix[i - 1][j - 1]:
                return False
    return True
```

## Magic Square Check

Square matrix where sum of each row, each column, and both diagonals are equal.

```python
def is_magic_square(matrix):
    if not matrix or len(matrix) != len(matrix[0]):
        return False
    n = len(matrix)
    target = sum(matrix[0])
    for i in range(1, n):
        if sum(matrix[i]) != target:
            return False
    for j in range(n):
        if sum(matrix[i][j] for i in range(n)) != target:
            return False
    if sum(matrix[i][i] for i in range(n)) != target:
        return False
    if sum(matrix[i][n - 1 - i] for i in range(n)) != target:
        return False
    return True
```

## Saddle Point

Element that is minimum in its row and maximum in its column (or vice versa).

```python
def find_saddle_points(matrix):
    if not matrix or not matrix[0]:
        return []
    m, n = len(matrix), len(matrix[0])
    row_mins = [min(row) for row in matrix]
    col_maxs = [max(matrix[i][j] for i in range(m)) for j in range(n)]
    result = []
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == row_mins[i] and matrix[i][j] == col_maxs[j]:
                result.append((i, j))
    return result
```

## Snake Pattern Print

Print row 0 left-to-right, row 1 right-to-left, row 2 left-to-right, etc.

```python
def snake_pattern(matrix):
    result = []
    for i in range(len(matrix)):
        if i % 2 == 0:
            result.extend(matrix[i])
        else:
            result.extend(matrix[i][::-1])
    return result
```

## Convert 1D to 2D and Back

```python
def flatten_2d_to_1d(matrix):
    return [val for row in matrix for val in row]

def reshape_1d_to_2d(arr, m, n):
    if len(arr) != m * n:
        raise ValueError("Length mismatch")
    return [arr[i * n:(i + 1) * n] for i in range(m)]

def index_1d_to_2d(idx, n):
    return idx // n, idx % n

def index_2d_to_1d(i, j, n):
    return i * n + j
```

## Reshape Matrix

Reshape matrix from r x c to new_r x new_c, filling row by row. Must have r*c == new_r*new_c.

```python
def reshape_matrix(matrix, new_r, new_c):
    flat = [val for row in matrix for val in row]
    if len(flat) != new_r * new_c:
        return matrix
    return [flat[i * new_c:(i + 1) * new_c] for i in range(new_r)]
```
