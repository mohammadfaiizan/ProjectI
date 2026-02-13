# Basic Matrix Operations

## Create Matrix m x n

```python
def create_matrix(m, n, default=0):
    return [[default] * n for _ in range(m)]

def create_matrix_from_list(flat, m, n):
    matrix = []
    for i in range(m):
        matrix.append(flat[i * n:(i + 1) * n])
    return matrix
```

## Access Element

```python
def access(matrix, i, j):
    return matrix[i][j]

def safe_access(matrix, i, j, default=None):
    if 0 <= i < len(matrix) and 0 <= j < len(matrix[0]):
        return matrix[i][j]
    return default
```

## Traverse Row-Wise

```python
def traverse_row_wise(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            yield matrix[i][j]

def traverse_row_wise_indexed(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            yield (i, j, matrix[i][j])
```

## Traverse Column-Wise

```python
def traverse_column_wise(matrix):
    n = len(matrix[0])
    for j in range(n):
        for i in range(len(matrix)):
            yield matrix[i][j]

def traverse_column_wise_indexed(matrix):
    n = len(matrix[0])
    for j in range(n):
        for i in range(len(matrix)):
            yield (i, j, matrix[i][j])
```

## Transpose

```python
def transpose(matrix):
    if not matrix or not matrix[0]:
        return []
    m, n = len(matrix), len(matrix[0])
    result = [[0] * m for _ in range(n)]
    for i in range(m):
        for j in range(n):
            result[j][i] = matrix[i][j]
    return result
```

## Matrix Addition

```python
def matrix_add(A, B):
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Dimensions must match")
    m, n = len(A), len(A[0])
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(m)]
```

## Matrix Subtraction

```python
def matrix_subtract(A, B):
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Dimensions must match")
    m, n = len(A), len(A[0])
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(m)]
```

## Scalar Multiplication

```python
def scalar_multiply(matrix, scalar):
    return [[matrix[i][j] * scalar for j in range(len(matrix[0]))] for i in range(len(matrix))]
```

## Matrix Multiplication (Naive O(n^3))

```python
def matrix_multiply(A, B):
    m, n, p = len(A), len(A[0]), len(B[0])
    if len(B) != n:
        raise ValueError("A cols must equal B rows")
    result = [[0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result
```

## Check Symmetric

```python
def is_symmetric(matrix):
    if not matrix or len(matrix) != len(matrix[0]):
        return False
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True
```

## Check Identity

```python
def is_identity(matrix):
    if not matrix or len(matrix) != len(matrix[0]):
        return False
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            expected = 1 if i == j else 0
            if matrix[i][j] != expected:
                return False
    return True
```

## Find Row with Max 1s (Binary Matrix)

```python
def row_with_max_ones(matrix):
    if not matrix or not matrix[0]:
        return -1
    max_count = -1
    result_row = -1
    for i in range(len(matrix)):
        count = sum(matrix[i])
        if count > max_count:
            max_count = count
            result_row = i
    return result_row

def row_with_max_ones_binary_search(matrix):
    if not matrix or not matrix[0]:
        return -1
    n = len(matrix[0])
    max_ones = -1
    result_row = -1
    for i in range(len(matrix)):
        lo, hi = 0, n - 1
        first_one = n
        while lo <= hi:
            mid = (lo + hi) // 2
            if matrix[i][mid] == 1:
                first_one = mid
                hi = mid - 1
            else:
                lo = mid + 1
        ones = n - first_one
        if ones > max_ones:
            max_ones = ones
            result_row = i
    return result_row
```

## Row-Wise Sum

```python
def row_wise_sum(matrix):
    return [sum(row) for row in matrix]
```

## Column-Wise Sum

```python
def column_wise_sum(matrix):
    if not matrix or not matrix[0]:
        return []
    n = len(matrix[0])
    result = [0] * n
    for row in matrix:
        for j in range(n):
            result[j] += row[j]
    return result
```

## Print Boundary

```python
def print_boundary(matrix):
    if not matrix or not matrix[0]:
        return []
    m, n = len(matrix), len(matrix[0])
    if m == 1:
        return matrix[0][:]
    if n == 1:
        return [matrix[i][0] for i in range(m)]
    result = []
    result.extend(matrix[0])
    for i in range(1, m - 1):
        result.append(matrix[i][n - 1])
    result.extend(matrix[m - 1][::-1])
    for i in range(m - 2, 0, -1):
        result.append(matrix[i][0])
    return result

def boundary_elements(matrix):
    if not matrix or not matrix[0]:
        return []
    m, n = len(matrix), len(matrix[0])
    if m <= 2 or n <= 2:
        return [matrix[i][j] for i in range(m) for j in range(n)]
    result = []
    for j in range(n):
        result.append(matrix[0][j])
    for i in range(1, m - 1):
        result.append(matrix[i][n - 1])
    for j in range(n - 1, -1, -1):
        result.append(matrix[m - 1][j])
    for i in range(m - 2, 0, -1):
        result.append(matrix[i][0])
    return result
```

## Count Zeros and Ones in Binary Matrix

```python
def count_zeros_ones(matrix):
    zeros = ones = 0
    for row in matrix:
        for val in row:
            if val == 0:
                zeros += 1
            else:
                ones += 1
    return zeros, ones

def count_zeros_ones_binary(matrix):
    total = len(matrix) * len(matrix[0])
    ones = sum(sum(row) for row in matrix)
    return total - ones, ones
```
