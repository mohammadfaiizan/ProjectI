# Easy Matrix Problems

## 1. Transpose Matrix

**Description**: Return transpose of matrix. Swap rows and columns.
**Approach**: Create new matrix of size n x m. result[j][i] = matrix[i][j].

```python
def transpose(matrix):
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]
```

Time: O(m * n) | Space: O(m * n)

---

## 2. Reshape the Matrix

**Description**: Reshape matrix from r x c to new_r x new_c. Fill row by row. Return original if impossible.
**Approach**: Flatten to 1D, then fill new matrix. Check r*c == new_r*new_c.

```python
def matrix_reshape(mat, r, c):
    m, n = len(mat), len(mat[0])
    if m * n != r * c:
        return mat
    flat = [x for row in mat for x in row]
    return [flat[i:i + c] for i in range(0, len(flat), c)]
```

Time: O(m * n) | Space: O(m * n)

---

## 3. Flipping an Image

**Description**: Flip image horizontally, then invert (0 to 1, 1 to 0).
**Approach**: For each row, reverse then XOR each element with 1.

```python
def flip_and_invert_image(image):
    return [[1 - x for x in row[::-1]] for row in image]
```

Time: O(m * n) | Space: O(1)

---

## 4. Toeplitz Matrix

**Description**: Check if matrix is Toeplitz (each diagonal has same elements).
**Approach**: For each (i,j) with i>0 and j>0, check matrix[i][j] == matrix[i-1][j-1].

```python
def is_toeplitz_matrix(matrix):
    for i in range(1, len(matrix)):
        for j in range(1, len(matrix[0])):
            if matrix[i][j] != matrix[i - 1][j - 1]:
                return False
    return True
```

Time: O(m * n) | Space: O(1)

---

## 5. Image Smoother

**Description**: Replace each pixel with average of 3x3 neighborhood (floor).
**Approach**: Create new matrix. For each cell, sum 9 neighbors (or fewer at edges), divide by count.

```python
def image_smoother(img):
    m, n = len(img), len(img[0])
    out = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            total, cnt = 0, 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m and 0 <= nj < n:
                        total += img[ni][nj]
                        cnt += 1
            out[i][j] = total // cnt
    return out
```

Time: O(m * n) | Space: O(m * n)

---

## 6. Flood Fill

**Description**: Replace connected region of same color from (sr,sc) with new color.
**Approach**: DFS or BFS from start. Only recurse when neighbor has same color as original.

```python
def flood_fill(image, sr, sc, new_color):
    old = image[sr][sc]
    if old == new_color:
        return image
    m, n = len(image), len(image[0])
    def dfs(i, j):
        if 0 <= i < m and 0 <= j < n and image[i][j] == old:
            image[i][j] = new_color
            for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
                dfs(i + di, j + dj)
    dfs(sr, sc)
    return image
```

Time: O(m * n) | Space: O(m * n)

---

## 7. Find the Town Judge

**Description**: In trust array [a,b] meaning a trusts b, find person trusted by all except themselves who trusts nobody.
**Approach**: Count in-degree and out-degree. Judge has in-degree n-1, out-degree 0.

```python
def find_judge(n, trust):
    in_deg = [0] * (n + 1)
    out_deg = [0] * (n + 1)
    for a, b in trust:
        out_deg[a] += 1
        in_deg[b] += 1
    for i in range(1, n + 1):
        if in_deg[i] == n - 1 and out_deg[i] == 0:
            return i
    return -1
```

Time: O(n + t) | Space: O(n)

---

## 8. Matrix Diagonal Sum

**Description**: Sum elements on both diagonals of square matrix. Do not double-count center.
**Approach**: Sum matrix[i][i] and matrix[i][n-1-i]. If n odd, subtract center once.

```python
def diagonal_sum(mat):
    n = len(mat)
    total = sum(mat[i][i] + mat[i][n - 1 - i] for i in range(n))
    return total - (mat[n // 2][n // 2] if n % 2 else 0)
```

Time: O(n) | Space: O(1)

---

## 9. Cells with Odd Values in a Matrix

**Description**: Start with zeros. Apply operations: increment all cells in row i, or in col j. Count cells with odd values.
**Approach**: Track which rows and cols are incremented odd times. Cell (i,j) odd iff (row_i XOR col_j) is 1.

```python
def odd_cells(m, n, indices):
    rows = [0] * m
    cols = [0] * n
    for r, c in indices:
        rows[r] ^= 1
        cols[c] ^= 1
    return sum(rows[i] ^ cols[j] for i in range(m) for j in range(n))
```

Time: O(m * n + len(indices)) | Space: O(m + n)

---

## 10. Special Positions in a Binary Matrix

**Description**: Position (i,j) is special if matrix[i][j]==1 and all other elements in row i and col j are 0.
**Approach**: Precompute row sums and col sums. Special if matrix[i][j]==1 and row_sum[i]==1 and col_sum[j]==1.

```python
def num_special(mat):
    rows = [sum(row) for row in mat]
    cols = [sum(mat[i][j] for i in range(len(mat))) for j in range(len(mat[0]))]
    return sum(1 for i in range(len(mat)) for j in range(len(mat[0]))
               if mat[i][j] == 1 and rows[i] == 1 and cols[j] == 1)
```

Time: O(m * n) | Space: O(m + n)

---

## 11. Maximum Population Year

**Description**: Logs [birth, death]. Find year with maximum population.
**Approach**: Create array of size 101 (1950-2050). For each log, increment birth year, decrement death year. Prefix sum, find max.

```python
def maximum_population(logs):
    delta = [0] * 101
    for b, d in logs:
        delta[b - 1950] += 1
        delta[d - 1950] -= 1
    cur = best = year = 0
    for i in range(101):
        cur += delta[i]
        if cur > best:
            best, year = cur, 1950 + i
    return year
```

Time: O(n) | Space: O(1)

---

## 12. Row With Maximum Ones

**Description**: Binary matrix. Find row index with maximum number of 1s.
**Approach**: Linear scan each row, count 1s. Or binary search for first 1 in each row (if sorted).

```python
def row_with_max_ones(mat):
    return max(range(len(mat)), key=lambda i: sum(mat[i]))
```

Time: O(m * n) | Space: O(1)

---

## 13. Richest Customer Wealth

**Description**: accounts[i][j] is money in bank j of customer i. Find max sum over all banks for any customer.
**Approach**: For each row, sum elements. Return max row sum.

```python
def maximum_wealth(accounts):
    return max(sum(row) for row in accounts)
```

Time: O(m * n) | Space: O(1)

---

## 14. Check if Every Row and Column Contains All Numbers

**Description**: n x n matrix. Check if each row and each column contains exactly 1 to n.
**Approach**: For each row and col, use set to check distinct values in range [1,n].

```python
def check_valid(matrix):
    n = len(matrix)
    target = set(range(1, n + 1))
    for row in matrix:
        if set(row) != target:
            return False
    for j in range(n):
        if set(matrix[i][j] for i in range(n)) != target:
            return False
    return True
```

Time: O(n^2) | Space: O(n)

---

## 15. Minimum Time Visiting All Points

**Description**: Array of points. Find minimum time to visit all in order. Can move 1 unit horizontally, vertically, or diagonally per second.
**Approach**: For consecutive points (x1,y1) and (x2,y2), time = max(|x2-x1|, |y2-y1|). Sum over pairs.

```python
def min_time_to_visit_all_points(points):
    return sum(max(abs(points[i][0] - points[i-1][0]), abs(points[i][1] - points[i-1][1]))
              for i in range(1, len(points)))
```

Time: O(n) | Space: O(1)

---

## 16. Lucky Numbers in a Matrix

**Description**: Lucky number is minimum in its row and maximum in its column.
**Approach**: Precompute row mins and col maxs. Check each cell.

```python
def lucky_numbers(matrix):
    row_mins = [min(row) for row in matrix]
    col_maxs = [max(matrix[i][j] for i in range(len(matrix))) for j in range(len(matrix[0]))]
    return [matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix[0]))
            if matrix[i][j] == row_mins[i] == col_maxs[j]]
```

Time: O(m * n) | Space: O(m + n)

---

## 17. Count Negative Numbers in Sorted Matrix

**Description**: Each row and column sorted non-increasing. Count negatives.
**Approach**: Start top-right. For each row, find first negative. All to the right are negative. Move left as we go down.

```python
def count_negatives(grid):
    m, n = len(grid), len(grid[0])
    r, c, cnt = 0, n - 1, 0
    while r < m and c >= 0:
        if grid[r][c] < 0:
            cnt += m - r
            c -= 1
        else:
            r += 1
    return cnt
```

Time: O(m + n) | Space: O(1)

---

## 18. Sort the Matrix Diagonally

**Description**: Sort each diagonal (top-left to bottom-right) of matrix.
**Approach**: Group elements by (i-j). Sort each group. Place back.

```python
def diagonal_sort(mat):
    from collections import defaultdict
    d = defaultdict(list)
    m, n = len(mat), len(mat[0])
    for i in range(m):
        for j in range(n):
            d[i - j].append(mat[i][j])
    for k in d:
        d[k].sort(reverse=True)
    for i in range(m):
        for j in range(n):
            mat[i][j] = d[i - j].pop()
    return mat
```

Time: O(m * n log(min(m,n))) | Space: O(m * n)

---

## 19. Find Winner on a Tic Tac Toe Game

**Description**: Moves array. Determine winner of 3x3 game (A, B, or pending/draw).
**Approach**: Build 3x3 board. Check rows, cols, diagonals for 3 in a row.

```python
def tictactoe(moves):
    board = [[0] * 3 for _ in range(3)]
    for i, (r, c) in enumerate(moves):
        board[r][c] = 1 if i % 2 == 0 else -1
    for row in board:
        if sum(row) == 3:
            return "A"
        if sum(row) == -3:
            return "B"
    for j in range(3):
        col = sum(board[i][j] for i in range(3))
        if col == 3:
            return "A"
        if col == -3:
            return "B"
    d1 = board[0][0] + board[1][1] + board[2][2]
    d2 = board[0][2] + board[1][1] + board[2][0]
    if d1 == 3 or d2 == 3:
        return "A"
    if d1 == -3 or d2 == -3:
        return "B"
    return "Draw" if len(moves) == 9 else "Pending"
```

Time: O(1) | Space: O(1)

---

## 20. Available Captures for Rook

**Description**: Chess board. Find number of pawns rook can capture (first piece in each direction).
**Approach**: Find rook position. Scan 4 directions until piece or edge. Count pawns ('p').

```python
def num_rook_captures(board):
    for i in range(8):
        for j in range(8):
            if board[i][j] == 'R':
                cnt = 0
                for di, dj in [(0,1),(0,-1),(1,0),(-1,0)]:
                    ni, nj = i + di, j + dj
                    while 0 <= ni < 8 and 0 <= nj < 8:
                        if board[ni][nj] == 'p':
                            cnt += 1
                            break
                        if board[ni][nj] != '.':
                            break
                        ni += di
                        nj += dj
                return cnt
    return 0
```

Time: O(1) | Space: O(1)

---

## 21. Projection Area of 3D Shapes

**Description**: grid[i][j] = height of tower. Find total projection area (top + front + side).
**Approach**: Top: count non-zero cells. Front: max per column. Side: max per row. Sum all.

```python
def projection_area(grid):
    top = sum(1 for row in grid for x in row if x)
    front = sum(max(row) for row in grid)
    side = sum(max(grid[i][j] for i in range(len(grid))) for j in range(len(grid[0])))
    return top + front + side
```

Time: O(m * n) | Space: O(1)

---

## 22. Shift 2D Grid

**Description**: Shift grid right by k positions (circular).
**Approach**: Flatten to 1D, rotate by k (reverse trick or modulo), reshape.

```python
def shift_grid(grid, k):
    m, n = len(grid), len(grid[0])
    flat = [x for row in grid for x in row]
    k %= len(flat)
    flat = flat[-k:] + flat[:-k]
    return [flat[i:i + n] for i in range(0, len(flat), n)]
```

Time: O(m * n) | Space: O(m * n)

---

## 23. Delete Greatest Value in Each Row

**Description**: Repeatedly pick max from each row (delete after), add to score. Find max total score.
**Approach**: Sort each row. For each column, add max of that column across rows.

```python
def delete_greatest_value(grid):
    for row in grid:
        row.sort()
    return sum(max(grid[i][j] for i in range(len(grid))) for j in range(len(grid[0])))
```

Time: O(m * n log n) | Space: O(1)

---

## 24. Equal Row and Column Pairs

**Description**: Count pairs (i,j) where row i equals column j.
**Approach**: Hash row tuples. For each column, count how many rows match.

```python
def equal_pairs(grid):
    from collections import Counter
    rows = Counter(tuple(row) for row in grid)
    return sum(rows[tuple(grid[i][j] for i in range(len(grid)))] for j in range(len(grid[0])))
```

Time: O(n^2) | Space: O(n^2)

---

## 25. Sum of Matrix After Queries

**Description**: n x n matrix of zeros. Queries: set row i to val, or set col j to val. Sum final matrix.
**Approach**: Process queries in reverse. Track which rows/cols are overwritten. Later queries override earlier. Sum = sum of (val * cells not overwritten by later ops).

```python
def matrix_sum_queries(n, queries):
    row_seen = col_seen = set()
    total = 0
    for t, i, v in reversed(queries):
        if t == 0 and i not in row_seen:
            row_seen.add(i)
            total += v * (n - len(col_seen))
        elif t == 1 and i not in col_seen:
            col_seen.add(i)
            total += v * (n - len(row_seen))
    return total
```Time: O(q) | Space: O(n)
