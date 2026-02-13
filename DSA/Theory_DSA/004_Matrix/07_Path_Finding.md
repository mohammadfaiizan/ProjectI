# Matrix Path Finding

## Unique Paths (DP)

Robot at top-left, can move only right or down. Count paths to bottom-right. dp[i][j] = dp[i-1][j] + dp[i][j-1].

```python
def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[m - 1][n - 1]

def unique_paths_optimized(m, n):
    row = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            row[j] += row[j - 1]
    return row[n - 1]
```

## Unique Paths with Obstacles

Grid has 1 for obstacle, 0 for empty. Same movement rules.

```python
def unique_paths_with_obstacles(grid):
    if not grid or not grid[0] or grid[0][0] == 1:
        return 0
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] if grid[0][j] == 0 else 0
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] if grid[i][0] == 0 else 0
    for i in range(1, m):
        for j in range(1, n):
            if grid[i][j] == 1:
                dp[i][j] = 0
            else:
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[m - 1][n - 1]
```

## Minimum Path Sum

Find path from top-left to bottom-right that minimizes sum of numbers on path.

```python
def min_path_sum(grid):
    if not grid or not grid[0]:
        return 0
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i - 1][j], dp[i][j - 1])
    return dp[m - 1][n - 1]
```

## Triangle Min Path Sum

Triangular grid: row i has i+1 elements. Move to adjacent numbers in row below.

```python
def minimum_total(triangle):
    if not triangle:
        return 0
    n = len(triangle)
    dp = triangle[n - 1][:]
    for i in range(n - 2, -1, -1):
        for j in range(len(triangle[i])):
            dp[j] = triangle[i][j] + min(dp[j], dp[j + 1])
    return dp[0]
```

## Maximal Square of 1s (DP)

Find largest square of 1s. dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) if matrix[i][j]==1.

```python
def maximal_square(matrix):
    if not matrix or not matrix[0]:
        return 0
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_side = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if matrix[i - 1][j - 1] == '1':
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
                max_side = max(max_side, dp[i][j])
    return max_side * max_side
```

## Maximal Rectangle of 1s (Histogram)

For each row, build histogram of consecutive 1s from above, then find max area in histogram.

```python
def maximal_rectangle(matrix):
    if not matrix or not matrix[0]:
        return 0
    n = len(matrix[0])
    heights = [0] * (n + 1)
    max_area = 0
    for row in matrix:
        for j in range(n):
            heights[j] = heights[j] + 1 if row[j] == '1' else 0
        stack = []
        for j in range(n + 1):
            while stack and heights[stack[-1]] > heights[j]:
                h = heights[stack.pop()]
                w = j - stack[-1] - 1 if stack else j
                max_area = max(max_area, h * w)
            stack.append(j)
    return max_area
```

## Dungeon Game

Princess in bottom-right. Knight starts top-left. Each cell has health change. Knight must have min 1 health at any time. Find minimum initial health.

```python
def calculate_minimum_hp(dungeon):
    if not dungeon or not dungeon[0]:
        return 1
    m, n = len(dungeon), len(dungeon[0])
    dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
    dp[m][n - 1] = dp[m - 1][n] = 1
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            need = min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j]
            dp[i][j] = max(1, need)
    return dp[0][0]
```

## Cherry Pickup

Two paths from (0,0) to (n-1,n-1). Collect cherries. Same cell counted once. Maximize total.

```python
def cherry_pickup(grid):
    n = len(grid)
    dp = [[[float('-inf')] * n for _ in range(n)] for _ in range(2 * n - 1)]
    dp[0][0][0] = grid[0][0]
    for s in range(1, 2 * n - 1):
        for r1 in range(max(0, s - n + 1), min(s + 1, n)):
            for r2 in range(max(0, s - n + 1), min(s + 1, n)):
                c1, c2 = s - r1, s - r2
                if grid[r1][c1] == -1 or grid[r2][c2] == -1:
                    continue
                cherries = grid[r1][c1] if r1 == r2 else grid[r1][c1] + grid[r2][c2]
                best = float('-inf')
                for dr1, dc1 in [(0, -1), (-1, 0)]:
                    for dr2, dc2 in [(0, -1), (-1, 0)]:
                        nr1, nc1 = r1 + dr1, c1 + dc1
                        nr2, nc2 = r2 + dr2, c2 + dc2
                        if 0 <= nr1 < n and 0 <= nc1 < n and 0 <= nr2 < n and 0 <= nc2 < n:
                            best = max(best, dp[s - 1][nr1][nr2])
                if best != float('-inf'):
                    dp[s][r1][r2] = best + cherries
    return max(0, dp[2 * n - 2][n - 1][n - 1])
```

## Cherry Pickup II (Two Robots)

Two robots start at (0,0) and (0,n-1). Move down (same row each step). Can move to adjacent columns. Maximize cherries collected.

```python
def cherry_pickup_ii(grid):
    m, n = len(grid), len(grid[0])
    dp = [[[float('-inf')] * n for _ in range(n)] for _ in range(m)]
    dp[0][0][n - 1] = grid[0][0] + (grid[0][n - 1] if n > 1 else 0)
    for i in range(1, m):
        for j1 in range(n):
            for j2 in range(n):
                best = float('-inf')
                for d1 in [-1, 0, 1]:
                    for d2 in [-1, 0, 1]:
                        pj1, pj2 = j1 + d1, j2 + d2
                        if 0 <= pj1 < n and 0 <= pj2 < n and dp[i - 1][pj1][pj2] != float('-inf'):
                            cherries = grid[i][j1] + (grid[i][j2] if j1 != j2 else 0)
                            best = max(best, dp[i - 1][pj1][pj2] + cherries)
                if best != float('-inf'):
                    dp[i][j1][j2] = best
    return max(max(row) for row in dp[m - 1])
```

## Longest Increasing Path (DFS + Memo)

Find longest strictly increasing path (up, down, left, right).

```python
def longest_increasing_path(matrix):
    if not matrix or not matrix[0]:
        return 0
    m, n = len(matrix), len(matrix[0])
    memo = {}

    def dfs(i, j):
        if (i, j) in memo:
            return memo[(i, j)]
        best = 1
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and matrix[ni][nj] > matrix[i][j]:
                best = max(best, 1 + dfs(ni, nj))
        memo[(i, j)] = best
        return best

    return max(dfs(i, j) for i in range(m) for j in range(n))
```

## Shortest Path in Binary Matrix (BFS)

0s are passable, 1s blocked. 8-direction movement. Find shortest path from (0,0) to (n-1,n-1).

```python
from collections import deque

def shortest_path_binary_matrix(grid):
    if not grid or grid[0][0] == 1:
        return -1
    n = len(grid)
    if n == 1:
        return 1
    q = deque([(0, 0, 1)])
    grid[0][0] = 1
    while q:
        r, c, dist = q.popleft()
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if nr == n - 1 and nc == n - 1:
                    return dist + 1
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                    grid[nr][nc] = 1
                    q.append((nr, nc, dist + 1))
    return -1
```

## Shortest Bridge (BFS)

Two islands of 1s in sea of 0s. Find minimum 0s to flip to connect them.

```python
from collections import deque

def shortest_bridge(grid):
    n = len(grid)
    found = False
    queue = deque()
    for i in range(n):
        if found:
            break
        for j in range(n):
            if grid[i][j] == 1:
                stack = [(i, j)]
                grid[i][j] = 2
                while stack:
                    r, c = stack.pop()
                    queue.append((r, c, 0))
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 1:
                            grid[nr][nc] = 2
                            stack.append((nr, nc))
                found = True
                break
    while queue:
        r, c, d = queue.popleft()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                if grid[nr][nc] == 1:
                    return d
                if grid[nr][nc] == 0:
                    grid[nr][nc] = 2
                    queue.append((nr, nc, d + 1))
    return 0
```

## Pacific Atlantic Water Flow

Cells flow to Pacific (top/left) or Atlantic (bottom/right). Find cells that can flow to both.

```python
def pacific_atlantic(heights):
    if not heights or not heights[0]:
        return []
    m, n = len(heights), len(heights[0])

    def dfs(r, c, visited):
        visited.add((r, c))
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in visited and heights[nr][nc] >= heights[r][c]:
                dfs(nr, nc, visited)

    pacific, atlantic = set(), set()
    for j in range(n):
        dfs(0, j, pacific)
        dfs(m - 1, j, atlantic)
    for i in range(m):
        dfs(i, 0, pacific)
        dfs(i, n - 1, atlantic)
    return list(pacific & atlantic)
```

## Surrounded Regions

Flip O's to X's if surrounded by X's. O's on border or connected to border stay.

```python
def solve_surrounded(board):
    if not board or not board[0]:
        return
    m, n = len(board), len(board[0])

    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != 'O':
            return
        board[i][j] = 'T'
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            dfs(i + di, j + dj)

    for i in range(m):
        dfs(i, 0)
        dfs(i, n - 1)
    for j in range(n):
        dfs(0, j)
        dfs(m - 1, j)
    for i in range(m):
        for j in range(n):
            if board[i][j] == 'T':
                board[i][j] = 'O'
            elif board[i][j] == 'O':
                board[i][j] = 'X'
```

## Walls and Gates

Fill each empty room with distance to nearest gate. -1 is wall, 0 is gate, INF is empty room.

```python
from collections import deque

def walls_and_gates(rooms):
    if not rooms or not rooms[0]:
        return
    m, n = len(rooms), len(rooms[0])
    q = deque()
    for i in range(m):
        for j in range(n):
            if rooms[i][j] == 0:
                q.append((i, j, 0))
    while q:
        r, c, d = q.popleft()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and rooms[nr][nc] == 2147483647:
                rooms[nr][nc] = d + 1
                q.append((nr, nc, d + 1))
```

## Number of Islands

Count connected components of 1s (4-direction).

```python
def num_islands(grid):
    if not grid or not grid[0]:
        return 0
    m, n = len(grid), len(grid[0])
    count = 0

    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != '1':
            return
        grid[i][j] = '0'
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            dfs(i + di, j + dj)

    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                count += 1
                dfs(i, j)
    return count
```

## Number of Distinct Islands

Count distinct island shapes. Encode shape as path string (directions taken).

```python
def num_distinct_islands(grid):
    if not grid or not grid[0]:
        return 0
    m, n = len(grid), len(grid[0])
    shapes = set()

    def dfs(i, j, path, dir_char):
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != 1:
            return
        grid[i][j] = 0
        path.append(dir_char)
        dfs(i + 1, j, path, 'd')
        dfs(i - 1, j, path, 'u')
        dfs(i, j + 1, path, 'r')
        dfs(i, j - 1, path, 'l')
        path.append('b')

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                path = []
                dfs(i, j, path, 's')
                shapes.add(''.join(path))
    return len(shapes)
```

## Number of Enclaves

Count 1s that cannot reach the boundary (walking on 1s).

```python
def num_enclaves(grid):
    if not grid or not grid[0]:
        return 0
    m, n = len(grid), len(grid[0])

    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != 1:
            return 0
        grid[i][j] = 0
        return 1 + dfs(i + 1, j) + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i, j - 1)

    for i in range(m):
        dfs(i, 0)
        dfs(i, n - 1)
    for j in range(n):
        dfs(0, j)
        dfs(m - 1, j)
    return sum(dfs(i, j) for i in range(m) for j in range(n) if grid[i][j] == 1)
```

## Count Sub-Islands

Grid1 and grid2. Count islands in grid2 that are fully covered by 1s in grid1.

```python
def count_sub_islands(grid1, grid2):
    if not grid2 or not grid2[0]:
        return 0
    m, n = len(grid2), len(grid2[0])
    count = 0

    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or grid2[i][j] != 1:
            return True
        grid2[i][j] = 0
        valid = grid1[i][j] == 1
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            valid = dfs(i + di, j + dj) and valid
        return valid

    for i in range(m):
        for j in range(n):
            if grid2[i][j] == 1 and dfs(i, j):
                count += 1
    return count
```

## Word Search (Backtracking)

Find if word exists by moving adjacent (no reuse).

```python
def exist(board, word):
    if not board or not board[0] or not word:
        return False
    m, n = len(board), len(board[0])

    def dfs(i, j, k):
        if k == len(word):
            return True
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[k]:
            return False
        tmp = board[i][j]
        board[i][j] = '#'
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if dfs(i + di, j + dj, k + 1):
                board[i][j] = tmp
                return True
        board[i][j] = tmp
        return False

    for i in range(m):
        for j in range(n):
            if dfs(i, j, 0):
                return True
    return False
```

## Word Search II (Trie + Backtracking)

Find all words from dictionary that exist on board.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None

def build_trie(words):
    root = TrieNode()
    for w in words:
        node = root
        for c in w:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.word = w
    return root

def find_words(board, words):
    if not board or not board[0] or not words:
        return []
    root = build_trie(words)
    m, n = len(board), len(board[0])
    result = []

    def dfs(i, j, node):
        c = board[i][j]
        if c not in node.children:
            return
        node = node.children[c]
        if node.word:
            result.append(node.word)
            node.word = None
        tmp = board[i][j]
        board[i][j] = '#'
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and board[ni][nj] != '#':
                dfs(ni, nj, node)
        board[i][j] = tmp

    for i in range(m):
        for j in range(n):
            dfs(i, j, root)
    return result
```
