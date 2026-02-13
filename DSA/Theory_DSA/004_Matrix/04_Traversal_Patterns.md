# Matrix Traversal Patterns

## Spiral Order Traversal (Layer Approach)

Traverse matrix in spiral order: top row left-to-right, right column top-to-bottom, bottom row right-to-left, left column bottom-to-top. Repeat for inner layers.

```
Matrix:              Spiral order: 1 2 3 4 8 12 16 15 14 13 9 5 6 7 11 10

  1  2  3  4
  5  6  7  8
  9 10 11 12
 13 14 15 16

Layer 0: top(1,2,3,4) -> right(8,12,16) -> bottom(15,14,13) -> left(9,5)
Layer 1: top(6,7) -> right(11) -> bottom(10)
```

```python
def spiral_order(matrix):
    if not matrix or not matrix[0]:
        return []
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        top += 1
        if top > bottom:
            break
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1
        if left > right:
            break
        for j in range(right, left - 1, -1):
            result.append(matrix[bottom][j])
        bottom -= 1
        if top > bottom:
            break
        for i in range(bottom, top - 1, -1):
            result.append(matrix[i][left])
        left += 1
    return result
```

## Spiral Matrix II (Generate Filled)

Generate n x n matrix filled with 1 to n^2 in spiral order.

```
n=3:
  1 2 3
  8 9 4
  7 6 5
```

```python
def generate_spiral_matrix(n):
    matrix = [[0] * n for _ in range(n)]
    num = 1
    top, bottom = 0, n - 1
    left, right = 0, n - 1
    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            matrix[top][j] = num
            num += 1
        top += 1
        if top > bottom:
            break
        for i in range(top, bottom + 1):
            matrix[i][right] = num
            num += 1
        right -= 1
        if left > right:
            break
        for j in range(right, left - 1, -1):
            matrix[bottom][j] = num
            num += 1
        bottom -= 1
        if top > bottom:
            break
        for i in range(bottom, top - 1, -1):
            matrix[i][left] = num
            num += 1
        left += 1
    return matrix
```

## Zigzag/Snake Traversal

Row 0 left-to-right, row 1 right-to-left, row 2 left-to-right, alternating.

```
Matrix:              Zigzag: 1 2 3 4 8 7 6 5 9 10 11 12

  1  2  3  4
  5  6  7  8
  9 10 11 12
```

```python
def zigzag_traversal(matrix):
    result = []
    for i in range(len(matrix)):
        if i % 2 == 0:
            result.extend(matrix[i])
        else:
            result.extend(matrix[i][::-1])
    return result
```

## Diagonal Traversal (Top-Right to Bottom-Left)

Traverse diagonals from top-right to bottom-left. Each diagonal has constant (i + j).

```
Matrix:              Diagonals (i+j): 0: (0,0)
  1  2  3  4                  1: (0,1),(1,0)
  5  6  7  8                  2: (0,2),(1,1),(2,0)
  9 10 11 12                  3: (0,3),(1,2),(2,1)
                              4: (1,3),(2,2)
Order: 1, 2, 5, 3, 6, 9, 4, 7, 10, 8, 11, 12  5: (2,3)
```

```python
def diagonal_traversal(matrix):
    if not matrix or not matrix[0]:
        return []
    m, n = len(matrix), len(matrix[0])
    result = []
    for d in range(m + n - 1):
        if d < n:
            i, j = 0, d
        else:
            i, j = d - n + 1, n - 1
        while i < m and j >= 0:
            result.append(matrix[i][j])
            i += 1
            j -= 1
    return result
```

## Anti-Diagonal Traversal

Traverse diagonals from top-left to bottom-right. Each anti-diagonal has constant (i - j) or (j - i).

```
Matrix:              Anti-diagonals (i-j): -2: (0,2),(1,3)
  1  2  3  4                -1: (0,1),(1,2),(2,3)
  5  6  7  8                 0: (0,0),(1,1),(2,2)
  9 10 11 12                 1: (1,0),(2,1)
                              2: (2,0)
Order: 1, 2, 5, 3, 6, 9, 4, 7, 10, 8, 11, 12
```

```python
def anti_diagonal_traversal(matrix):
    if not matrix or not matrix[0]:
        return []
    m, n = len(matrix), len(matrix[0])
    result = []
    for d in range(-(n - 1), m):
        if d < 0:
            i, j = 0, -d
        else:
            i, j = d, 0
        while i < m and j < n:
            result.append(matrix[i][j])
            i += 1
            j += 1
    return result
```

## Boundary Traversal

Traverse outer boundary: top row, right column (excluding corners), bottom row (reversed), left column (excluding corners).

```
Matrix:              Boundary: 1 2 3 4 8 12 16 15 14 13 9 5

  1  2  3  4
  5  6  7  8
  9 10 11 12
 13 14 15 16
```

```python
def boundary_traversal(matrix):
    if not matrix or not matrix[0]:
        return []
    m, n = len(matrix), len(matrix[0])
    if m == 1:
        return matrix[0][:]
    if n == 1:
        return [matrix[i][0] for i in range(m)]
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

## Wave Traversal (Alternating Direction)

Column-wise traversal: col 0 top-to-bottom, col 1 bottom-to-top, col 2 top-to-bottom, etc.

```
Matrix:              Wave: 1 5 9 13 14 10 6 2 3 7 11 15 16 12 8 4

  1  2  3  4
  5  6  7  8
  9 10 11 12
 13 14 15 16
```

```python
def wave_traversal(matrix):
    if not matrix or not matrix[0]:
        return []
    result = []
    n = len(matrix[0])
    for j in range(n):
        if j % 2 == 0:
            for i in range(len(matrix)):
                result.append(matrix[i][j])
        else:
            for i in range(len(matrix) - 1, -1, -1):
                result.append(matrix[i][j])
    return result
```

## Print All Diagonals

Print each diagonal as a separate list. Diagonals from top-left going down-right.

```
Matrix:              Diagonals:
  1  2  3           [1], [2, 5], [3, 6, 9], [4, 7, 10], [8, 11], [12]
  4  5  6
  7  8  9
 10 11 12
```

```python
def all_diagonals(matrix):
    if not matrix or not matrix[0]:
        return []
    m, n = len(matrix), len(matrix[0])
    result = []
    for j in range(n):
        diag = []
        i, c = 0, j
        while i < m and c < n:
            diag.append(matrix[i][c])
            i += 1
            c += 1
        result.append(diag)
    for i in range(1, m):
        diag = []
        r, j = i, 0
        while r < m and j < n:
            diag.append(matrix[r][j])
            r += 1
            j += 1
        result.append(diag)
    return result
```

## Concentric Rectangular Traversal

Traverse layer by layer from outside to inside, each layer being a rectangle.

```
Matrix:              Layer 0: 1,2,3,4,8,12,16,15,14,13,9,5
  1  2  3  4         Layer 1: 6,7,11,10
  5  6  7  8
  9 10 11 12
 13 14 15 16
```

```python
def concentric_traversal(matrix):
    if not matrix or not matrix[0]:
        return []
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        for i in range(top + 1, bottom + 1):
            result.append(matrix[i][right])
        if top < bottom:
            for j in range(right - 1, left - 1, -1):
                result.append(matrix[bottom][j])
        if left < right:
            for i in range(bottom - 1, top, -1):
                result.append(matrix[i][left])
        top += 1
        bottom -= 1
        left += 1
        right -= 1
    return result
```
