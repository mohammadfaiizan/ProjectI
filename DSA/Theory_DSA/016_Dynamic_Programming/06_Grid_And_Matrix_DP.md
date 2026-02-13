# Grid and Matrix DP

## Unique Paths I

```python
def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[m - 1][n - 1]
```

## Unique Paths II (Obstacles)

```python
def unique_paths_with_obstacles(obstacle_grid):
    m, n = len(obstacle_grid), len(obstacle_grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1 if obstacle_grid[0][0] == 0 else 0
    for i in range(m):
        for j in range(n):
            if obstacle_grid[i][j] == 1:
                continue
            if i > 0:
                dp[i][j] += dp[i - 1][j]
            if j > 0:
                dp[i][j] += dp[i][j - 1]
    return dp[m - 1][n - 1]
```

## Minimum Path Sum

```python
def min_path_sum(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
    return dp[m - 1][n - 1]
```

## Triangle Min Total

```python
def minimum_total(triangle):
    n = len(triangle)
    dp = triangle[-1][:]
    for i in range(n - 2, -1, -1):
        for j in range(len(triangle[i])):
            dp[j] = min(dp[j], dp[j + 1]) + triangle[i][j]
    return dp[0]
```

## Maximal Square of 1s

```python
def maximal_square(matrix):
    if not matrix:
        return 0
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_side = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if matrix[i - 1][j - 1] == '1':
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                max_side = max(max_side, dp[i][j])
    return max_side * max_side
```

## Maximal Rectangle of 1s (Histogram)

```python
def maximal_rectangle(matrix):
    if not matrix:
        return 0
    m, n = len(matrix), len(matrix[0])
    heights = [0] * (n + 1)
    max_area = 0
    for i in range(m):
        for j in range(n):
            heights[j] = heights[j] + 1 if matrix[i][j] == '1' else 0
        stack = []
        for j in range(n + 1):
            while stack and heights[stack[-1]] > heights[j]:
                h = heights[stack.pop()]
                w = j if not stack else j - stack[-1] - 1
                max_area = max(max_area, h * w)
            stack.append(j)
    return max_area
```

## Count Square Submatrices

```python
def count_squares(matrix):
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    total = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if matrix[i - 1][j - 1] == 1:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                total += dp[i][j]
    return total
```

## Dungeon Game

```python
def calculate_minimum_hp(dungeon):
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

```python
def cherry_pickup(grid):
    n = len(grid)
    dp = [[float('-inf')] * n for _ in range(n)]
    dp[0][0] = grid[0][0]
    for t in range(1, 2 * n - 1):
        dp2 = [[float('-inf')] * n for _ in range(n)]
        for i in range(max(0, t - n + 1), min(n, t + 1)):
            for j in range(max(0, t - n + 1), min(n, t + 1)):
                if grid[i][t - i] == -1 or grid[j][t - j] == -1:
                    continue
                val = grid[i][t - i]
                if i != j:
                    val += grid[j][t - j]
                for pi in (i - 1, i):
                    for pj in (j - 1, j):
                        if pi >= 0 and pj >= 0:
                            dp2[i][j] = max(dp2[i][j], dp[pi][pj] + val)
        dp = dp2
    return max(0, dp[n - 1][n - 1])
```

## Cherry Pickup II (Two Robots)

```python
def cherry_pickup_ii(grid):
    m, n = len(grid), len(grid[0])
    dp = [[[float('-inf')] * n for _ in range(n)] for _ in range(m)]
    dp[0][0][n - 1] = grid[0][0] + grid[0][n - 1] if n > 1 else grid[0][0]
    for r in range(1, m):
        for c1 in range(n):
            for c2 in range(n):
                best = float('-inf')
                for d1 in (-1, 0, 1):
                    for d2 in (-1, 0, 1):
                        nc1, nc2 = c1 + d1, c2 + d2
                        if 0 <= nc1 < n and 0 <= nc2 < n:
                            best = max(best, dp[r - 1][nc1][nc2])
                if best == float('-inf'):
                    continue
                val = grid[r][c1]
                if c1 != c2:
                    val += grid[r][c2]
                dp[r][c1][c2] = best + val
    return max(max(row) for row in dp[m - 1])
```

## Longest Increasing Path (DFS + Memo)

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
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and matrix[ni][nj] > matrix[i][j]:
                best = max(best, 1 + dfs(ni, nj))
        memo[(i, j)] = best
        return best
    
    return max(dfs(i, j) for i in range(m) for j in range(n))
```

## Minimum Falling Path Sum

```python
def min_falling_path_sum(matrix):
    n = len(matrix)
    dp = matrix[0][:]
    for i in range(1, n):
        new_dp = [0] * n
        for j in range(n):
            new_dp[j] = matrix[i][j] + min(
                dp[j - 1] if j > 0 else float('inf'),
                dp[j],
                dp[j + 1] if j < n - 1 else float('inf')
            )
        dp = new_dp
    return min(dp)
```

## Minimum Falling Path Sum II

```python
def min_falling_path_sum_ii(grid):
    n = len(grid)
    dp = grid[0][:]
    for i in range(1, n):
        min1 = min(dp)
        idx = dp.index(min1)
        min2 = min(v for j, v in enumerate(dp) if j != idx)
        new_dp = []
        for j in range(n):
            val = grid[i][j] + (min1 if j != idx else min2)
            new_dp.append(val)
        dp = new_dp
    return min(dp)
```

## Knight Probability

```python
def knight_probability(n, k, row, column):
    moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    dp = [[0] * n for _ in range(n)]
    dp[row][column] = 1
    for _ in range(k):
        dp2 = [[0] * n for _ in range(n)]
        for r in range(n):
            for c in range(n):
                if dp[r][c] > 0:
                    for dr, dc in moves:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n and 0 <= nc < n:
                            dp2[nr][nc] += dp[r][c] / 8
        dp = dp2
    return sum(sum(row) for row in dp)
```

## Out of Boundary Paths

```python
def find_paths(m, n, max_move, start_row, start_column):
    MOD = 10**9 + 7
    dp = [[0] * n for _ in range(m)]
    dp[start_row][start_column] = 1
    result = 0
    for _ in range(max_move):
        dp2 = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if dp[i][j] > 0:
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < m and 0 <= nj < n:
                            dp2[ni][nj] = (dp2[ni][nj] + dp[i][j]) % MOD
                        else:
                            result = (result + dp[i][j]) % MOD
        dp = dp2
    return result
```

## Paint House

```python
def min_cost_paint_house(costs):
    if not costs:
        return 0
    r, g, b = costs[0]
    for i in range(1, len(costs)):
        r, g, b = (
            costs[i][0] + min(g, b),
            costs[i][1] + min(r, b),
            costs[i][2] + min(r, g)
        )
    return min(r, g, b)
```

## Paint House II

```python
def min_cost_ii(costs):
    if not costs:
        return 0
    n, k = len(costs), len(costs[0])
    dp = costs[0][:]
    for i in range(1, n):
        min1 = min(dp)
        idx = dp.index(min1)
        min2 = min(v for j, v in enumerate(dp) if j != idx)
        new_dp = []
        for j in range(k):
            val = costs[i][j] + (min1 if j != idx else min2)
            new_dp.append(val)
        dp = new_dp
    return min(dp)
```
