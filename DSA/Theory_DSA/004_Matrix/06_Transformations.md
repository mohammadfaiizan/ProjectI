# Matrix Transformations

## Rotate Image 90 Clockwise (Transpose + Reverse)

Transpose the matrix, then reverse each row. Equivalent to rotating 90 degrees clockwise.

```
Before:              After (90 CW):
a b c                 g d a
d e f                 h e b
g h i                 i f c
```

```python
def rotate_90_clockwise(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for i in range(n):
        matrix[i].reverse()
```

## Rotate 90 Counter-Clockwise (Transpose + Reverse Cols)

Transpose then reverse each column. Or reverse each row then transpose.

```
Before:              After (90 CCW):
a b c                 c f i
d e f                 b e h
g h i                 a d g
```

```python
def rotate_90_counter_clockwise(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for j in range(n):
        for i in range(n // 2):
            matrix[i][j], matrix[n - 1 - i][j] = matrix[n - 1 - i][j], matrix[i][j]
```

## Rotate 180

Reverse each row, then reverse the order of rows. Or swap (i,j) with (n-1-i, n-1-j) for all i,j in upper half.

```
Before:              After (180):
a b c                 i h g
d e f                 f e d
g h i                 c b a
```

```python
def rotate_180(matrix):
    n = len(matrix)
    for i in range(n // 2):
        for j in range(n):
            matrix[i][j], matrix[n - 1 - i][n - 1 - j] = matrix[n - 1 - i][n - 1 - j], matrix[i][j]
    if n % 2 == 1:
        mid = n // 2
        for j in range(n // 2):
            matrix[mid][j], matrix[mid][n - 1 - j] = matrix[mid][n - 1 - j], matrix[mid][j]
```

## Reflect Over Main Diagonal (Transpose)

Swap elements across main diagonal. M[i][j] <-> M[j][i].

```
Before:              After (transpose):
a b c                 a d g
d e f                 b e h
g h i                 c f i
```

```python
def reflect_main_diagonal(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
```

## Reflect Over Anti-Diagonal

Swap (i,j) with (n-1-j, n-1-i). Or transpose then rotate 180.

```
Before:              After (anti-diagonal):
a b c                 i f c
d e f                 h e b
g h i                 g d a
```

```python
def reflect_anti_diagonal(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(n - 1 - i):
            ni, nj = n - 1 - j, n - 1 - i
            if (i, j) < (ni, nj):
                matrix[i][j], matrix[ni][nj] = matrix[ni][nj], matrix[i][j]
```

## Reflect Horizontally

Reverse each row. Mirror over vertical axis.

```
Before:              After:
a b c                 c b a
d e f                 f e d
g h i                 i h g
```

```python
def reflect_horizontal(matrix):
    for row in matrix:
        row.reverse()
```

## Reflect Vertically

Reverse row order. Mirror over horizontal axis.

```
Before:              After:
a b c                 g h i
d e f                 d e f
g h i                 a b c
```

```python
def reflect_vertical(matrix):
    matrix.reverse()
```

## Game of Life (Simultaneous Update with Bit Encoding)

Each cell: 1=live, 0=dead. Rules: live with 2-3 neighbors stays live; dead with 3 neighbors becomes live; else dead. Update all cells simultaneously. Use bit encoding: store next state in second bit, then shift.

```
Encoding: 0b00=dead->dead, 0b01=live->dead, 0b10=dead->live, 0b11=live->live
Current state: LSB. Next state: second bit.
After processing: right shift to get next state.
```

```python
def game_of_life(matrix):
    if not matrix or not matrix[0]:
        return
    m, n = len(matrix), len(matrix[0])
    for i in range(m):
        for j in range(n):
            live = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m and 0 <= nj < n and (matrix[ni][nj] & 1):
                        live += 1
            if matrix[i][j] & 1:
                if 2 <= live <= 3:
                    matrix[i][j] |= 2
            else:
                if live == 3:
                    matrix[i][j] |= 2
    for i in range(m):
        for j in range(n):
            matrix[i][j] >>= 1
```

## Image Smoother

Replace each pixel with average of itself and 8 neighbors (floor of average). Use new matrix to avoid overwriting.

```python
def image_smoother(matrix):
    if not matrix or not matrix[0]:
        return matrix
    m, n = len(matrix), len(matrix[0])
    result = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            total, count = 0, 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m and 0 <= nj < n:
                        total += matrix[ni][nj]
                        count += 1
            result[i][j] = total // count
    return result
```

## Flood Fill

Replace all connected cells of same color from (sr, sc) with new color. DFS or BFS.

```python
def flood_fill(matrix, sr, sc, new_color):
    if not matrix or not matrix[0]:
        return matrix
    old_color = matrix[sr][sc]
    if old_color == new_color:
        return matrix
    m, n = len(matrix), len(matrix[0])

    def dfs(r, c):
        if r < 0 or r >= m or c < 0 or c >= n or matrix[r][c] != old_color:
            return
        matrix[r][c] = new_color
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    dfs(sr, sc)
    return matrix
```
