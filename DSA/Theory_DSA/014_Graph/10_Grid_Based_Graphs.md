# Graph - Grid Based Graphs

## Theory

A 2D grid can be treated as an implicit graph: each cell is a vertex, adjacent cells (4 or 8 directions) are edges. No explicit adjacency list needed; compute neighbors on the fly. Common for BFS shortest path, DFS flood fill, multi-source BFS.

## Grid as Implicit Graph

Each (r, c) is a vertex. Edges: (r,c) to (r+1,c), (r-1,c), (r,c+1), (r,c-1) for 4-directional; add diagonals for 8-directional.

## BFS on Grid Template

```python
def bfs_grid(grid, start_r, start_c):
    from collections import deque
    rows, cols = len(grid), len(grid[0])
    q = deque([(start_r, start_c)])
    visited = {(start_r, start_c)}
    while q:
        r, c = q.popleft()
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
```

## DFS on Grid Template

```python
def dfs_grid(grid, r, c, visited):
    rows, cols = len(grid), len(grid[0])
    visited.add((r, c))
    for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
            dfs_grid(grid, nr, nc, visited)
```

## Number of Islands

```python
def num_islands(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    count = 0
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return
        grid[r][c] = '0'
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)
    return count
```

## Number of Distinct Islands

```python
def num_distinct_islands(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    shapes = set()
    def dfs(r, c, r0, c0, shape):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == 0:
            return
        grid[r][c] = 0
        shape.append((r - r0, c - c0))
        dfs(r+1, c, r0, c0, shape)
        dfs(r-1, c, r0, c0, shape)
        dfs(r, c+1, r0, c0, shape)
        dfs(r, c-1, r0, c0, shape)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                shape = []
                dfs(r, c, r, c, shape)
                shapes.add(tuple(shape))
    return len(shapes)
```

## Max Area of Island

```python
def max_area_of_island(grid):
    rows, cols = len(grid), len(grid[0])
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == 0:
            return 0
        grid[r][c] = 0
        return 1 + dfs(r+1,c) + dfs(r-1,c) + dfs(r,c+1) + dfs(r,c-1)
    return max(dfs(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1) if any(grid[r][c] for r in range(rows) for c in range(cols)) else 0
```

## Number of Enclaves

```python
def num_enclaves(grid):
    rows, cols = len(grid), len(grid[0])
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == 0:
            return
        grid[r][c] = 0
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)
    for r in range(rows):
        if grid[r][0] == 1:
            dfs(r, 0)
        if grid[r][cols-1] == 1:
            dfs(r, cols-1)
    for c in range(cols):
        if grid[0][c] == 1:
            dfs(0, c)
        if grid[rows-1][c] == 1:
            dfs(rows-1, c)
    return sum(grid[r][c] for r in range(rows) for c in range(cols))
```

## Count Sub-Islands

```python
def count_sub_islands(grid1, grid2):
    rows, cols = len(grid1), len(grid1[0])
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid2[r][c] == 0:
            return True
        grid2[r][c] = 0
        valid = grid1[r][c] == 1
        valid &= dfs(r+1, c)
        valid &= dfs(r-1, c)
        valid &= dfs(r, c+1)
        valid &= dfs(r, c-1)
        return valid
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid2[r][c] == 1 and dfs(r, c):
                count += 1
    return count
```

## Making a Large Island

```python
def largest_island(grid):
    n = len(grid)
    island_id = 2
    size = {}
    def dfs(r, c, idx):
        if r < 0 or r >= n or c < 0 or c >= n or grid[r][c] != 1:
            return 0
        grid[r][c] = idx
        return 1 + dfs(r+1,c,idx) + dfs(r-1,c,idx) + dfs(r,c+1,idx) + dfs(r,c-1,idx)
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 1:
                size[island_id] = dfs(r, c, island_id)
                island_id += 1
    result = max(size.values()) if size else 0
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 0:
                seen = set()
                total = 1
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] and grid[nr][nc] not in seen:
                        seen.add(grid[nr][nc])
                        total += size[grid[nr][nc]]
                result = max(result, total)
    return result
```

## Shortest Path Binary Matrix

```python
def shortest_path_binary_matrix(grid):
    from collections import deque
    n = len(grid)
    if grid[0][0] or grid[n-1][n-1]:
        return -1
    q = deque([(0, 0, 1)])
    grid[0][0] = 1
    while q:
        r, c, d = q.popleft()
        if r == n-1 and c == n-1:
            return d
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                    grid[nr][nc] = 1
                    q.append((nr, nc, d + 1))
    return -1
```

## Rotting Oranges (Multi-Source BFS)

```python
def oranges_rotting(grid):
    from collections import deque
    rows, cols = len(grid), len(grid[0])
    q = deque()
    fresh = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                q.append((r, c, 0))
            elif grid[r][c] == 1:
                fresh += 1
    minutes = 0
    while q:
        r, c, t = q.popleft()
        minutes = t
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh -= 1
                q.append((nr, nc, t + 1))
    return minutes if fresh == 0 else -1
```

## Walls and Gates

```python
def walls_and_gates(rooms):
    from collections import deque
    if not rooms:
        return
    rows, cols = len(rooms), len(rooms[0])
    q = deque((r, c) for r in range(rows) for c in range(cols) if rooms[r][c] == 0)
    while q:
        r, c = q.popleft()
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and rooms[nr][nc] == 2**31 - 1:
                rooms[nr][nc] = rooms[r][c] + 1
                q.append((nr, nc))
```

## 01 Matrix (Nearest 0)

```python
def update_matrix(mat):
    from collections import deque
    rows, cols = len(mat), len(mat[0])
    q = deque((r, c) for r in range(rows) for c in range(cols) if mat[r][c] == 0)
    visited = set(q)
    while q:
        r, c = q.popleft()
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                visited.add((nr, nc))
                mat[nr][nc] = mat[r][c] + 1
                q.append((nr, nc))
    return mat
```

## Pacific Atlantic Water Flow

```python
def pacific_atlantic(heights):
    if not heights:
        return []
    rows, cols = len(heights), len(heights[0])
    pacific = set()
    atlantic = set()
    def dfs(r, c, ocean):
        ocean.add((r, c))
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in ocean and heights[nr][nc] >= heights[r][c]:
                dfs(nr, nc, ocean)
    for r in range(rows):
        dfs(r, 0, pacific)
        dfs(r, cols-1, atlantic)
    for c in range(cols):
        dfs(0, c, pacific)
        dfs(rows-1, c, atlantic)
    return list(pacific & atlantic)
```

## Surrounded Regions

```python
def solve(board):
    if not board:
        return
    rows, cols = len(board), len(board[0])
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != 'O':
            return
        board[r][c] = 'T'
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)
    for r in range(rows):
        if board[r][0] == 'O':
            dfs(r, 0)
        if board[r][cols-1] == 'O':
            dfs(r, cols-1)
    for c in range(cols):
        if board[0][c] == 'O':
            dfs(0, c)
        if board[rows-1][c] == 'O':
            dfs(rows-1, c)
    for r in range(rows):
        for c in range(cols):
            board[r][c] = 'X' if board[r][c] != 'T' else 'O'
```

## As Far From Land as Possible

```python
def max_distance(grid):
    from collections import deque
    n = len(grid)
    q = deque((r, c) for r in range(n) for c in range(n) if grid[r][c] == 1)
    if len(q) == 0 or len(q) == n * n:
        return -1
    dist = 0
    while q:
        for _ in range(len(q)):
            r, c = q.popleft()
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                    grid[nr][nc] = 1
                    q.append((nr, nc))
        dist += 1
    return dist - 1
```

## Map of Highest Peak

```python
def highest_peak(is_water):
    from collections import deque
    rows, cols = len(is_water), len(is_water[0])
    result = [[-1] * cols for _ in range(rows)]
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if is_water[r][c] == 1:
                result[r][c] = 0
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] == -1:
                result[nr][nc] = result[r][c] + 1
                q.append((nr, nc))
    return result
```

## Shortest Bridge

```python
def shortest_bridge(grid):
    from collections import deque
    n = len(grid)
    def dfs(r, c, island):
        if r < 0 or r >= n or c < 0 or c >= n or grid[r][c] != 1:
            return
        grid[r][c] = 2
        island.append((r, c))
        dfs(r+1, c, island)
        dfs(r-1, c, island)
        dfs(r, c+1, island)
        dfs(r, c-1, island)
    first = []
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 1:
                dfs(r, c, first)
                break
        if first:
            break
    q = deque((r, c, 0) for r, c in first)
    visited = set(first)
    while q:
        r, c, d = q.popleft()
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in visited:
                if grid[nr][nc] == 1:
                    return d
                visited.add((nr, nc))
                q.append((nr, nc, d + 1))
    return 0
```

## Swim in Rising Water

```python
def swim_in_water(grid):
    import heapq
    n = len(grid)
    pq = [(grid[0][0], 0, 0)]
    visited = {(0, 0)}
    while pq:
        t, r, c = heapq.heappop(pq)
        if r == n-1 and c == n-1:
            return t
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in visited:
                visited.add((nr, nc))
                heapq.heappush(pq, (max(t, grid[nr][nc]), nr, nc))
    return -1
```

## Path With Min Effort

```python
def minimum_effort_path(heights):
    import heapq
    rows, cols = len(heights), len(heights[0])
    dist = [[float('inf')] * cols for _ in range(rows)]
    dist[0][0] = 0
    pq = [(0, 0, 0)]
    while pq:
        effort, r, c = heapq.heappop(pq)
        if r == rows-1 and c == cols-1:
            return effort
        if effort > dist[r][c]:
            continue
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols:
                new_effort = max(effort, abs(heights[nr][nc] - heights[r][c]))
                if new_effort < dist[nr][nc]:
                    dist[nr][nc] = new_effort
                    heapq.heappush(pq, (new_effort, nr, nc))
    return -1
```

## Word Search (DFS Backtracking)

```python
def exist(board, word):
    rows, cols = len(board), len(board[0])
    def dfs(r, c, i):
        if i == len(word):
            return True
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != word[i]:
            return False
        tmp = board[r][c]
        board[r][c] = '#'
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            if dfs(r+dr, c+dc, i+1):
                return True
        board[r][c] = tmp
        return False
    return any(dfs(r, c, 0) for r in range(rows) for c in range(cols) if board[r][c] == word[0])
```

## Unique Paths III

```python
def unique_paths_iii(grid):
    rows, cols = len(grid), len(grid[0])
    empty = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                empty += 1
            elif grid[r][c] == 1:
                start = (r, c)
            elif grid[r][c] == 2:
                end = (r, c)
    def dfs(r, c, count):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == -1:
            return 0
        if (r, c) == end:
            return 1 if count == empty else 0
        grid[r][c] = -1
        total = 0
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            total += dfs(r+dr, c+dc, count + 1)
        grid[r][c] = 0
        return total
    return dfs(start[0], start[1], 0)
```
