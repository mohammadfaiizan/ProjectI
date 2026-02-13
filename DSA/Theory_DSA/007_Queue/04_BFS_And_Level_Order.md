# BFS and Level-Order Traversal

## BFS Template (Single Source)

Breadth-first search explores nodes level by level. Use a queue to process nodes in FIFO order. Mark visited to avoid reprocessing.

```python
from collections import deque

def bfs_template(graph, start):
    visited = {start}
    q = deque([start])
    while q:
        node = q.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
```

## BFS Shortest Path in Unweighted Graph

In an unweighted graph, BFS from source gives shortest path (by number of edges). Track distance or parent for each node.

```python
from collections import deque

def bfs_shortest_path(graph, start, end):
    if start == end:
        return 0
    visited = {start}
    q = deque([(start, 0)])
    while q:
        node, dist = q.popleft()
        for neighbor in graph[node]:
            if neighbor == end:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                q.append((neighbor, dist + 1))
    return -1
```

## Level-Order Traversal of Binary Tree

Process each level left to right. Use queue; for each level, process all nodes in queue (current level), enqueue their children.

```python
from collections import deque

def level_order(root):
    if not root:
        return []
    result = []
    q = deque([root])
    while q:
        node = q.popleft()
        result.append(node.val)
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return result
```

Level-order as list of levels:

```python
def level_order_levels(root):
    if not root:
        return []
    result = []
    q = deque([root])
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        result.append(level)
    return result
```

## Level-Order Bottom-Up

Same as level-order but reverse the result list of levels.

```python
def level_order_bottom_up(root):
    levels = level_order_levels(root)
    return levels[::-1]
```

## Zigzag Level-Order

Alternate left-to-right and right-to-left per level. Use a flag; reverse level when flag is True.

```python
def zigzag_level_order(root):
    if not root:
        return []
    result = []
    q = deque([root])
    left_to_right = True
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        if not left_to_right:
            level.reverse()
        result.append(level)
        left_to_right = not left_to_right
    return result
```

## Right Side View

Return the rightmost node of each level. BFS and take last element of each level.

```python
def right_side_view(root):
    if not root:
        return []
    result = []
    q = deque([root])
    while q:
        level_size = len(q)
        for i in range(level_size):
            node = q.popleft()
            if i == level_size - 1:
                result.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
    return result
```

## Left Side View

Return the leftmost node of each level.

```python
def left_side_view(root):
    if not root:
        return []
    result = []
    q = deque([root])
    while q:
        result.append(q[0].val)
        for _ in range(len(q)):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
    return result
```

## Average of Levels

Return the average value of nodes at each level.

```python
def average_of_levels(root):
    if not root:
        return []
    result = []
    q = deque([root])
    while q:
        total = 0
        count = len(q)
        for _ in range(count):
            node = q.popleft()
            total += node.val
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        result.append(total / count)
    return result
```

## Minimum Depth of Binary Tree

Shortest path from root to any leaf. BFS and return depth when we first see a leaf.

```python
def min_depth(root):
    if not root:
        return 0
    q = deque([(root, 1)])
    while q:
        node, depth = q.popleft()
        if not node.left and not node.right:
            return depth
        if node.left:
            q.append((node.left, depth + 1))
        if node.right:
            q.append((node.right, depth + 1))
    return 0
```

## Number of Islands (BFS)

Grid of '1' and '0'. Count connected components of '1'. BFS from each unvisited '1'.

```python
def num_islands(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                q = deque([(r, c)])
                grid[r][c] = '0'
                while q:
                    i, j = q.popleft()
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] == '1':
                            grid[ni][nj] = '0'
                            q.append((ni, nj))
    return count
```

## Rotting Oranges

Grid with 0 (empty), 1 (fresh), 2 (rotten). Each minute, rotten oranges rot adjacent fresh. Return minutes to rot all, or -1 if impossible.

```python
def oranges_rotting(grid):
    rows, cols = len(grid), len(grid[0])
    q = deque()
    fresh = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                q.append((r, c))
            elif grid[r][c] == 1:
                fresh += 1
    if fresh == 0:
        return 0
    minutes = 0
    while q:
        for _ in range(len(q)):
            i, j = q.popleft()
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] == 1:
                    grid[ni][nj] = 2
                    fresh -= 1
                    q.append((ni, nj))
        minutes += 1
    return minutes - 1 if fresh == 0 else -1
```

## Walls and Gates

Grid with -1 (wall), 0 (gate), INF (empty). Fill each empty room with distance to nearest gate.

```python
def walls_and_gates(rooms):
    if not rooms:
        return
    rows, cols = len(rooms), len(rooms[0])
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if rooms[r][c] == 0:
                q.append((r, c))
    while q:
        i, j = q.popleft()
        dist = rooms[i][j]
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols and rooms[ni][nj] == 2**31 - 1:
                rooms[ni][nj] = dist + 1
                q.append((ni, nj))
```

## Shortest Path in Binary Matrix

NxN grid of 0s and 1s. Find shortest path from (0,0) to (n-1,n-1) moving through 0s only. 8-direction movement.

```python
def shortest_path_binary_matrix(grid):
    n = len(grid)
    if grid[0][0] or grid[n-1][n-1]:
        return -1
    if n == 1:
        return 1
    q = deque([(0, 0, 1)])
    grid[0][0] = 1
    dirs = [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    while q:
        i, j, dist = q.popleft()
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if ni == n - 1 and nj == n - 1:
                return dist + 1
            if 0 <= ni < n and 0 <= nj < n and grid[ni][nj] == 0:
                grid[ni][nj] = 1
                q.append((ni, nj, dist + 1))
    return -1
```

## Word Ladder

Transform beginWord to endWord by changing one letter at a time; each intermediate must be in wordList. Return shortest transformation length.

```python
def ladder_length(begin_word, end_word, word_list):
    word_set = set(word_list)
    if end_word not in word_set:
        return 0
    q = deque([(begin_word, 1)])
    word_set.discard(begin_word)
    while q:
        word, length = q.popleft()
        if word == end_word:
            return length
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]
                if next_word in word_set:
                    word_set.discard(next_word)
                    q.append((next_word, length + 1))
    return 0
```

## Word Ladder II

Return all shortest transformation sequences from beginWord to endWord.

```python
def find_ladders(begin_word, end_word, word_list):
    word_set = set(word_list)
    if end_word not in word_set:
        return []
    layer = {begin_word: [[begin_word]]}
    while layer:
        new_layer = {}
        for word in layer:
            if word == end_word:
                return layer[word]
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    next_word = word[:i] + c + word[i+1:]
                    if next_word in word_set:
                        for path in layer[word]:
                            new_layer.setdefault(next_word, []).append(path + [next_word])
        word_set -= set(new_layer.keys())
        layer = new_layer
    return []
```

## Open the Lock

4-digit lock; each wheel 0-9. Start at "0000", target is end. Deadends are forbidden. Each move: rotate one wheel by 1. Return minimum number of moves.

```python
def open_lock(deadends, target):
    dead = set(deadends)
    if "0000" in dead:
        return -1
    q = deque([("0000", 0)])
    seen = {"0000"}
    while q:
        state, moves = q.popleft()
        if state == target:
            return moves
        for i in range(4):
            for d in (-1, 1):
                digit = (int(state[i]) + d) % 10
                next_state = state[:i] + str(digit) + state[i+1:]
                if next_state not in seen and next_state not in dead:
                    seen.add(next_state)
                    q.append((next_state, moves + 1))
    return -1
```

## Snakes and Ladders

NxN board, cells numbered 1 to N*N in Boustrophedon order. Some cells have snakes or ladders. Return minimum dice rolls from 1 to N*N.

```python
def snakes_and_ladders(board):
    n = len(board)
    target = n * n

    def get_pos(square):
        row = (square - 1) // n
        col = (square - 1) % n
        if row % 2 == 1:
            col = n - 1 - col
        row = n - 1 - row
        return row, col

    q = deque([(1, 0)])
    seen = {1}
    while q:
        square, moves = q.popleft()
        if square == target:
            return moves
        for next_sq in range(square + 1, min(square + 7, target + 1)):
            r, c = get_pos(next_sq)
            dest = board[r][c] if board[r][c] != -1 else next_sq
            if dest not in seen:
                seen.add(dest)
                q.append((dest, moves + 1))
    return -1
```

## Jump Game III (BFS)

Array arr and start index. From index i you can jump to i+arr[i] or i-arr[i]. Return true if you can reach any index with value 0.

```python
def can_reach(arr, start):
    n = len(arr)
    q = deque([start])
    seen = {start}
    while q:
        i = q.popleft()
        if arr[i] == 0:
            return True
        for j in [i + arr[i], i - arr[i]]:
            if 0 <= j < n and j not in seen:
                seen.add(j)
                q.append(j)
    return False
```

## Minimum Knight Moves

Minimum moves for knight from (0,0) to (x,y) on infinite chessboard. Knight moves in L-shape.

```python
def min_knight_moves(x, y):
    x, y = abs(x), abs(y)
    dirs = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]
    q = deque([(0, 0, 0)])
    seen = {(0, 0)}
    while q:
        i, j, moves = q.popleft()
        if i == x and j == y:
            return moves
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if (ni, nj) not in seen and ni >= -2 and nj >= -2:
                seen.add((ni, nj))
                q.append((ni, nj, moves + 1))
    return -1
```

## Shortest Path with Alternating Colors

Directed graph with red and blue edges. Find shortest path from 0 to n-1 where edge colors alternate. BFS with state (node, last_color); from red we take blue edges, from blue we take red edges.

```python
def shortest_alternating_paths(n, red_edges, blue_edges):
    red_adj = [[] for _ in range(n)]
    blue_adj = [[] for _ in range(n)]
    for u, v in red_edges:
        red_adj[u].append(v)
    for u, v in blue_edges:
        blue_adj[u].append(v)
    result = [-1] * n
    result[0] = 0
    q = deque([(0, 0, 0), (0, 0, 1)])
    seen = {(0, 0), (0, 1)}
    while q:
        node, dist, last_color = q.popleft()
        adj = blue_adj if last_color == 0 else red_adj
        next_color = 1 - last_color
        for neighbor in adj[node]:
            if (neighbor, next_color) not in seen:
                seen.add((neighbor, next_color))
                if result[neighbor] == -1:
                    result[neighbor] = dist + 1
                q.append((neighbor, dist + 1, next_color))
    return result
```

## Nearest Exit from Entrance

Maze with '.' and '+'. Find shortest path from entrance to any border cell (exit). BFS.

```python
def nearest_exit(maze, entrance):
    rows, cols = len(maze), len(maze[0])
    er, ec = entrance
    q = deque([(er, ec, 0)])
    maze[er][ec] = '+'
    while q:
        r, c, dist = q.popleft()
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                if dist > 0:
                    return dist
                continue
            if maze[nr][nc] == '.':
                maze[nr][nc] = '+'
                q.append((nr, nc, dist + 1))
    return -1
```

## Map of Highest Peak

Matrix of land (0) and water (1). Assign each cell a height so adjacent cells differ by at most 1 and water is 0. Maximize heights. Multi-source BFS from water cells.

```python
def highest_peak(is_water):
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
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] == -1:
                result[nr][nc] = result[r][c] + 1
                q.append((nr, nc))
    return result
```

## 01 Matrix (Nearest 0)

Matrix of 0s and 1s. For each cell, return distance to nearest 0. Multi-source BFS from all 0s.

```python
def update_matrix(mat):
    rows, cols = len(mat), len(mat[0])
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if mat[r][c] == 0:
                q.append((r, c))
            else:
                mat[r][c] = -1
    while q:
        r, c = q.popleft()
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and mat[nr][nc] == -1:
                mat[nr][nc] = mat[r][c] + 1
                q.append((nr, nc))
    return mat
```

## As Far from Land as Possible

NxN grid of 0 (water) and 1 (land). Find a water cell with maximum distance to nearest land. Multi-source BFS from land; return max distance.

```python
def max_distance(grid):
    n = len(grid)
    q = deque()
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 1:
                q.append((r, c))
    if len(q) == 0 or len(q) == n * n:
        return -1
    dist = 0
    while q:
        for _ in range(len(q)):
            r, c = q.popleft()
            for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                    grid[nr][nc] = 1
                    q.append((nr, nc))
        dist += 1
    return dist - 1
```

## Bus Routes

Array of bus routes; each route is list of stops. Find minimum number of buses to take from source to target. BFS on bus indices; from a bus, try all buses that share a stop.

```python
def num_buses_to_destination(routes, source, target):
    if source == target:
        return 0
    stop_to_buses = {}
    for i, route in enumerate(routes):
        for stop in route:
            stop_to_buses.setdefault(stop, []).append(i)
    q = deque([(source, 0)])
    seen_buses = set()
    while q:
        stop, buses = q.popleft()
        if stop == target:
            return buses
        for bus in stop_to_buses.get(stop, []):
            if bus not in seen_buses:
                seen_buses.add(bus)
                for next_stop in routes[bus]:
                    q.append((next_stop, buses + 1))
    return -1
```

## Sliding Puzzle

2x3 board with tiles 1-5 and empty (0). Find minimum moves to reach [[1,2,3],[4,5,0]]. BFS over board states.

```python
def sliding_puzzle(board):
    target = "123450"
    start = "".join(str(c) for row in board for c in row)
    if start == target:
        return 0
    neighbors = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4], 4: [1, 3, 5], 5: [2, 4]}
    q = deque([(start, 0)])
    seen = {start}
    while q:
        state, moves = q.popleft()
        idx = state.index('0')
        for n in neighbors[idx]:
            arr = list(state)
            arr[idx], arr[n] = arr[n], arr[idx]
            new_state = "".join(arr)
            if new_state == target:
                return moves + 1
            if new_state not in seen:
                seen.add(new_state)
                q.append((new_state, moves + 1))
    return -1
```
