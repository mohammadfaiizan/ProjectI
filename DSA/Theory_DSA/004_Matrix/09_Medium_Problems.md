# Medium Matrix Problems

## 1. Set Matrix Zeroes

**Description**: If element is 0, set entire row and column to 0. O(1) space.
**Approach**: Use first row and first column as markers. Handle (0,0) overlap with separate flags for first row and first column.

```python
def set_zeroes(matrix):
    m, n = len(matrix), len(matrix[0])
    row0 = col0 = False
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 0:
                if i == 0:
                    row0 = True
                if j == 0:
                    col0 = True
                matrix[i][0] = matrix[0][j] = 0
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    if row0:
        for j in range(n):
            matrix[0][j] = 0
    if col0:
        for i in range(m):
            matrix[i][0] = 0
```

Time: O(m * n) | Space: O(1)

---

## 2. Spiral Matrix

**Description**: Return elements in spiral order (top, right, bottom, left, repeat).
**Approach**: Layer-by-layer with four boundaries. Shrink boundaries after each side.

```python
def spiral_order(matrix):
    if not matrix:
        return []
    r1, r2, c1, c2 = 0, len(matrix) - 1, 0, len(matrix[0]) - 1
    out = []
    while r1 <= r2 and c1 <= c2:
        for c in range(c1, c2 + 1):
            out.append(matrix[r1][c])
        for r in range(r1 + 1, r2 + 1):
            out.append(matrix[r][c2])
        if r1 < r2 and c1 < c2:
            for c in range(c2 - 1, c1 - 1, -1):
                out.append(matrix[r2][c])
            for r in range(r2 - 1, r1, -1):
                out.append(matrix[r][c1])
        r1, r2, c1, c2 = r1 + 1, r2 - 1, c1 + 1, c2 - 1
    return out
```

Time: O(m * n) | Space: O(1)

---

## 3. Spiral Matrix II

**Description**: Generate n x n matrix filled with 1 to n^2 in spiral order.
**Approach**: Same layer approach. Fill while moving boundaries.

```python
def generate_spiral(n):
    mat = [[0] * n for _ in range(n)]
    r1, r2, c1, c2, v = 0, n - 1, 0, n - 1, 1
    while r1 <= r2 and c1 <= c2:
        for c in range(c1, c2 + 1):
            mat[r1][c] = v
            v += 1
        for r in range(r1 + 1, r2 + 1):
            mat[r][c2] = v
            v += 1
        if r1 < r2 and c1 < c2:
            for c in range(c2 - 1, c1 - 1, -1):
                mat[r2][c] = v
                v += 1
            for r in range(r2 - 1, r1, -1):
                mat[r][c1] = v
                v += 1
        r1, r2, c1, c2 = r1 + 1, r2 - 1, c1 + 1, c2 - 1
    return mat
```

Time: O(n^2) | Space: O(1)

---

## 4. Rotate Image

**Description**: Rotate n x n matrix 90 degrees clockwise in-place.
**Approach**: Transpose then reverse each row. Or rotate in 4-way swaps for each element in top-left quadrant.

```python
def rotate_image(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()
```

Time: O(n^2) | Space: O(1)

---

## 5. Search a 2D Matrix

**Description**: Matrix sorted row-wise (each row's last <= next row's first). Search target.
**Approach**: Treat as 1D sorted array. Binary search with index mapping mid//n, mid%n.

```python
def search_matrix(matrix, target):
    if not matrix:
        return False
    m, n = len(matrix), len(matrix[0])
    lo, hi = 0, m * n - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        r, c = mid // n, mid % n
        if matrix[r][c] == target:
            return True
        if matrix[r][c] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return False
```

Time: O(log(m * n)) | Space: O(1)

---

## 6. Search a 2D Matrix II

**Description**: Each row sorted, each column sorted. Search target.
**Approach**: Staircase from top-right. If current > target, move left; if current < target, move down. O(m+n).

```python
def search_matrix_ii(matrix, target):
    if not matrix:
        return False
    r, c = 0, len(matrix[0]) - 1
    while r < len(matrix) and c >= 0:
        if matrix[r][c] == target:
            return True
        if matrix[r][c] > target:
            c -= 1
        else:
            r += 1
    return False
```

Time: O(m + n) | Space: O(1)

---

## 7. Game of Life

**Description**: Apply Conway's rules simultaneously. In-place with O(1) extra space.
**Approach**: Encode next state in second bit. 0b01=live->dead, 0b10=dead->live. Right shift after.

```python
def game_of_life(board):
    m, n = len(board), len(board[0])
    for i in range(m):
        for j in range(n):
            live = sum(1 for di in (-1,0,1) for dj in (-1,0,1) if (di or dj) and
                     0 <= i+di < m and 0 <= j+dj < n and board[i+di][j+dj] & 1)
            if board[i][j] == 1 and 2 <= live <= 3:
                board[i][j] = 3
            elif board[i][j] == 0 and live == 3:
                board[i][j] = 2
    for i in range(m):
        for j in range(n):
            board[i][j] >>= 1
```

Time: O(m * n) | Space: O(1)

---

## 8. Unique Paths

**Description**: Robot at (0,0), move right or down to (m-1,n-1). Count paths.
**Approach**: DP. dp[i][j] = dp[i-1][j] + dp[i][j-1]. Space optimize to single row.

```python
def unique_paths(m, n):
    dp = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    return dp[-1]
```

Time: O(m * n) | Space: O(n)

---

## 9. Unique Paths II

**Description**: Same with obstacles. 1 blocks, 0 allows.
**Approach**: DP with obstacle check. dp[i][j]=0 if obstacle else dp[i-1][j]+dp[i][j-1].

```python
def unique_paths_with_obstacles(grid):
    m, n = len(grid), len(grid[0])
    dp = [0] * n
    dp[0] = 1
    for i in range(m):
        for j in range(n):
            if grid[i][j]:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j - 1]
    return dp[-1]
```

Time: O(m * n) | Space: O(n)

---

## 10. Minimum Path Sum

**Description**: Path from top-left to bottom-right minimizing sum.
**Approach**: DP. dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1]).

```python
def min_path_sum(grid):
    m, n = len(grid), len(grid[0])
    for i in range(1, m):
        grid[i][0] += grid[i - 1][0]
    for j in range(1, n):
        grid[0][j] += grid[0][j - 1]
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
    return grid[-1][-1]
```

Time: O(m * n) | Space: O(1)

---

## 11. Triangle (Minimum Path Sum)

**Description**: Triangular grid. Move to adjacent in next row. Minimize path sum.
**Approach**: DP from bottom. dp[j] = triangle[i][j] + min(dp[j], dp[j+1]).

```python
def minimum_total(triangle):
    dp = triangle[-1][:]
    for i in range(len(triangle) - 2, -1, -1):
        for j in range(len(triangle[i])):
            dp[j] = triangle[i][j] + min(dp[j], dp[j + 1])
    return dp[0]
```

Time: O(n^2) | Space: O(n)

---

## 12. Maximal Square

**Description**: Binary matrix. Find largest square of 1s.
**Approach**: DP. dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) if matrix[i][j]==1.

```python
def maximal_square(matrix):
    if not matrix:
        return 0
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    best = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if matrix[i - 1][j - 1] == '1':
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
                best = max(best, dp[i][j])
    return best * best
```

Time: O(m * n) | Space: O(m * n)

---

## 13. Maximal Rectangle

**Description**: Binary matrix. Find largest rectangle of 1s.
**Approach**: For each row, build histogram of consecutive 1s from above. Max area in histogram (stack) for each row.

```python
def maximal_rectangle(matrix):
    if not matrix:
        return 0
    n = len(matrix[0])
    h = [0] * (n + 1)
    best = 0
    for row in matrix:
        for j in range(n):
            h[j] = h[j] + 1 if row[j] == '1' else 0
        st = [-1]
        for j in range(n + 1):
            while st and h[st[-1]] > h[j]:
                idx = st.pop()
                best = max(best, h[idx] * (j - st[-1] - 1))
            st.append(j)
    return best
```

Time: O(m * n) | Space: O(n)

---

## 14. Number of Islands

**Description**: Count connected components of 1s (4-direction).
**Approach**: DFS/BFS. Mark visited by flipping to 0. Count DFS calls.

```python
def num_islands(grid):
    m, n = len(grid), len(grid[0])
    def dfs(i, j):
        if 0 <= i < m and 0 <= j < n and grid[i][j] == '1':
            grid[i][j] = '0'
            for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
                dfs(i + di, j + dj)
    cnt = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                dfs(i, j)
                cnt += 1
    return cnt
```

Time: O(m * n) | Space: O(m * n)

---

## 15. Surrounded Regions

**Description**: Flip O to X if surrounded. Border O's and connected stay.
**Approach**: DFS from all border O's, mark as temporary. Flip remaining O to X, restore temporary.

```python
def solve(board):
    if not board:
        return
    m, n = len(board), len(board[0])
    def dfs(i, j):
        if 0 <= i < m and 0 <= j < n and board[i][j] == 'O':
            board[i][j] = 'T'
            for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
                dfs(i + di, j + dj)
    for i in range(m):
        dfs(i, 0)
        dfs(i, n - 1)
    for j in range(n):
        dfs(0, j)
        dfs(m - 1, j)
    for i in range(m):
        for j in range(n):
            board[i][j] = 'X' if board[i][j] == 'O' else ('O' if board[i][j] == 'T' else board[i][j])
```

Time: O(m * n) | Space: O(m * n)

---

## 16. Pacific Atlantic Water Flow

**Description**: Cells that can flow to both Pacific (top/left) and Atlantic (bottom/right).
**Approach**: DFS from Pacific border and Atlantic border. Intersection of reachable sets.

```python
def pacific_atlantic(heights):
    if not heights:
        return []
    m, n = len(heights), len(heights[0])
    def dfs(i, j, seen):
        seen.add((i, j))
        for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and (ni, nj) not in seen and heights[ni][nj] >= heights[i][j]:
                dfs(ni, nj, seen)
    pac, atl = set(), set()
    for i in range(m):
        dfs(i, 0, pac)
        dfs(i, n - 1, atl)
    for j in range(n):
        dfs(0, j, pac)
        dfs(m - 1, j, atl)
    return list(pac & atl)
```

Time: O(m * n) | Space: O(m * n)

---

## 17. Longest Increasing Path in a Matrix

**Description**: Longest strictly increasing path (any direction).
**Approach**: DFS with memoization. For each cell, try 4 directions, memo longest path from that cell.

```python
def longest_increasing_path(matrix):
    if not matrix:
        return 0
    m, n = len(matrix), len(matrix[0])
    memo = {}
    def dfs(i, j):
        if (i, j) in memo:
            return memo[(i, j)]
        best = 1
        for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and matrix[ni][nj] > matrix[i][j]:
                best = max(best, 1 + dfs(ni, nj))
        memo[(i, j)] = best
        return best
    return max(dfs(i, j) for i in range(m) for j in range(n))
```

Time: O(m * n) | Space: O(m * n)

---

## 18. Word Search

**Description**: Find word by moving adjacent. No cell reuse.
**Approach**: Backtracking. For each starting cell, DFS with visited set (or mark in-place).

```python
def exist(board, word):
    m, n = len(board), len(board[0])
    def dfs(i, j, k):
        if k == len(word):
            return True
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[k]:
            return False
        tmp, board[i][j] = board[i][j], '#'
        for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
            if dfs(i + di, j + dj, k + 1):
                return True
        board[i][j] = tmp
        return False
    return any(dfs(i, j, 0) for i in range(m) for j in range(n))
```

Time: O(m * n * 4^L) | Space: O(L)

---

## 19. Word Search II

**Description**: Find all words from dictionary on board.
**Approach**: Build Trie of words. For each cell, DFS with Trie. When reaching word end, add to result.

```python
def find_words(board, words):
    from collections import defaultdict
    trie = lambda: defaultdict(trie)
    root = trie()
    for w in words:
        node = root
        for c in w:
            node = node[c]
        node['$'] = w
    m, n = len(board), len(board[0])
    out = []
    def dfs(i, j, node):
        c = board[i][j]
        if c not in node:
            return
        node = node[c]
        if '$' in node:
            out.append(node['$'])
            del node['$']
        board[i][j] = '#'
        for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n:
                dfs(ni, nj, node)
        board[i][j] = c
    for i in range(m):
        for j in range(n):
            dfs(i, j, root)
    return out
```

Time: O(m * n * 4^L) | Space: O(W * L)

---

## 20. Kth Smallest Element in a Sorted Matrix

**Description**: Each row and column sorted. Find kth smallest.
**Approach**: Min-heap of first element per row. Pop k times, push next in row. Or binary search on value range.

```python
def kth_smallest(matrix, k):
    import heapq
    n = len(matrix)
    h = [(matrix[i][0], i, 0) for i in range(min(k, n))]
    heapq.heapify(h)
    for _ in range(k - 1):
        _, i, j = heapq.heappop(h)
        if j + 1 < n:
            heapq.heappush(h, (matrix[i][j + 1], i, j + 1))
    return h[0][0]
```

Time: O(k log n) | Space: O(n)

---

## 21. 01 Matrix (Distance to Nearest Zero)

**Description**: Binary matrix. For each cell, find distance to nearest 0.
**Approach**: Multi-source BFS from all 0s. Or two passes (from top-left and bottom-right).

```python
def update_matrix(mat):
    from collections import deque
    m, n = len(mat), len(mat[0])
    q = deque((i, j) for i in range(m) for j in range(n) if mat[i][j] == 0)
    seen = set(q)
    while q:
        i, j = q.popleft()
        for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and (ni, nj) not in seen:
                mat[ni][nj] = mat[i][j] + 1
                seen.add((ni, nj))
                q.append((ni, nj))
    return mat
```

Time: O(m * n) | Space: O(m * n)

---

## 22. Shortest Path in Binary Matrix

**Description**: 0s passable, 1s blocked. 8-direction. Shortest path from (0,0) to (n-1,n-1).
**Approach**: BFS. Queue (r, c, dist). Mark visited. Return dist when reaching bottom-right.

```python
def shortest_path_binary_matrix(grid):
    from collections import deque
    if grid[0][0]:
        return -1
    n = len(grid)
    q = deque([(0, 0, 1)])
    grid[0][0] = 1
    while q:
        r, c, d = q.popleft()
        if r == c == n - 1:
            return d
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    nr, nc = r + di, c + dj
                    if 0 <= nr < n and 0 <= nc < n and not grid[nr][nc]:
                        grid[nr][nc] = 1
                        q.append((nr, nc, d + 1))
    return -1
```

Time: O(n^2) | Space: O(n^2)

---

## 23. Rotting Oranges

**Description**: 0=empty, 1=fresh, 2=rotten. Each minute rotten oranges rot adjacent. Minutes to rot all.
**Approach**: Multi-source BFS from rotten. Track minutes per level. Check if any fresh remains.

```python
def oranges_rotting(grid):
    from collections import deque
    m, n = len(grid), len(grid[0])
    q = deque((i, j, 0) for i in range(m) for j in range(n) if grid[i][j] == 2)
    mins = 0
    while q:
        i, j, mins = q.popleft()
        for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1:
                grid[ni][nj] = 2
                q.append((ni, nj, mins + 1))
    return -1 if any(grid[i][j] == 1 for i in range(m) for j in range(n)) else mins
```

Time: O(m * n) | Space: O(m * n)

---

## 24. As Far from Land as Possible

**Description**: 0=water, 1=land. Find water cell with maximum distance to nearest land.
**Approach**: Multi-source BFS from all land. Max distance in BFS is answer.

```python
def max_distance(grid):
    from collections import deque
    m, n = len(grid), len(grid[0])
    q = deque((i, j, 0) for i in range(m) for j in range(n) if grid[i][j])
    if len(q) == 0 or len(q) == m * n:
        return -1
    best = 0
    while q:
        i, j, d = q.popleft()
        best = d
        for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 0:
                grid[ni][nj] = 1
                q.append((ni, nj, d + 1))
    return best
```

Time: O(m * n) | Space: O(m * n)

---

## 25. Number of Closed Islands

**Description**: 0=land, 1=water. Count islands not touching border.
**Approach**: DFS from border 0s to mark as "not closed". Count remaining connected 0 components.

```python
def closed_island(grid):
    m, n = len(grid), len(grid[0])
    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n:
            return False
        if grid[i][j]:
            return True
        grid[i][j] = 1
        a = dfs(i + 1, j)
        b = dfs(i - 1, j)
        c = dfs(i, j + 1)
        d = dfs(i, j - 1)
        return a and b and c and d
    return sum(dfs(i, j) for i in range(m) for j in range(n) if grid[i][j] == 0)
```

Time: O(m * n) | Space: O(m * n)

---

## 26. Count Sub Islands

**Description**: grid1 and grid2. Count islands in grid2 fully covered by 1s in grid1.
**Approach**: For each island in grid2, DFS and check all cells have grid1[i][j]==1. If yes, count.

```python
def count_sub_islands(grid1, grid2):
    m, n = len(grid1), len(grid1[0])
    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or grid2[i][j] == 0:
            return True
        grid2[i][j] = 0
        ok = grid1[i][j] == 1
        for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
            ok &= dfs(i + di, j + dj)
        return ok
    return sum(dfs(i, j) for i in range(m) for j in range(n) if grid2[i][j])
```

Time: O(m * n) | Space: O(m * n)

---

## 27. Number of Enclaves

**Description**: Count 1s that cannot reach boundary.
**Approach**: DFS from all border 1s to mark reachable. Count unmarked 1s.

```python
def num_enclaves(grid):
    m, n = len(grid), len(grid[0])
    def dfs(i, j):
        if 0 <= i < m and 0 <= j < n and grid[i][j]:
            grid[i][j] = 0
            for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
                dfs(i + di, j + dj)
    for i in range(m):
        dfs(i, 0)
        dfs(i, n - 1)
    for j in range(n):
        dfs(0, j)
        dfs(m - 1, j)
    return sum(grid[i][j] for i in range(m) for j in range(n))
```

Time: O(m * n) | Space: O(m * n)

---

## 28. Shortest Bridge

**Description**: Two islands of 1s. Minimum 0s to flip to connect.
**Approach**: DFS to find and mark first island. BFS from first island to reach second. Return BFS distance.

```python
def shortest_bridge(grid):
    from collections import deque
    m, n = len(grid), len(grid[0])
    def dfs(i, j, island):
        if 0 <= i < m and 0 <= j < n and grid[i][j] == 1:
            grid[i][j] = 2
            island.append((i, j))
            for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
                dfs(i + di, j + dj, island)
    island = []
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                dfs(i, j, island)
                break
        if island:
            break
    q = deque((r, c, 0) for r, c in island)
    while q:
        r, c, d = q.popleft()
        for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r + di, c + dj
            if 0 <= nr < m and 0 <= nc < n:
                if grid[nr][nc] == 1:
                    return d
                if grid[nr][nc] == 0:
                    grid[nr][nc] = 2
                    q.append((nr, nc, d + 1))
    return 0
```

Time: O(m * n) | Space: O(m * n)

---

## 29. Walls and Gates

**Description**: -1=wall, 0=gate, INF=room. Fill each room with distance to nearest gate.
**Approach**: Multi-source BFS from all gates.

```python
def walls_and_gates(rooms):
    from collections import deque
    if not rooms:
        return
    m, n = len(rooms), len(rooms[0])
    q = deque((i, j) for i in range(m) for j in range(n) if rooms[i][j] == 0)
    while q:
        i, j = q.popleft()
        for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and rooms[ni][nj] == 2**31 - 1:
                rooms[ni][nj] = rooms[i][j] + 1
                q.append((ni, nj))
```

Time: O(m * n) | Space: O(m * n)

---

## 30. Valid Sudoku

**Description**: Check if 9x9 partially filled Sudoku is valid (no duplicates in row, col, 3x3 box).
**Approach**: Use sets for rows, cols, boxes. box_id = (r//3)*3 + c//3.

```python
def is_valid_sudoku(board):
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    for i in range(9):
        for j in range(9):
            c = board[i][j]
            if c == '.':
                continue
            if c in rows[i] or c in cols[j] or c in boxes[(i//3)*3 + j//3]:
                return False
            rows[i].add(c)
            cols[j].add(c)
            boxes[(i//3)*3 + j//3].add(c)
    return True
```

Time: O(1) | Space: O(1)
