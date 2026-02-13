# Graph - DFS and BFS

## Theory

DFS (Depth-First Search) explores as far as possible along each branch before backtracking. Uses stack (explicit or implicit via recursion). BFS (Breadth-First Search) explores level by level. Uses queue. Both visit each vertex once, time O(V + E) with adjacency list.

## DFS Recursive Template

```python
def dfs_recursive(graph, start):
    visited = set()
    result = []
    def dfs(v):
        visited.add(v)
        result.append(v)
        for u in graph.get(v, []):
            if u not in visited:
                dfs(u)
    dfs(start)
    return result
```

## DFS Iterative (Stack)

```python
def dfs_iterative(graph, start):
    visited = set()
    result = []
    stack = [start]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        result.append(v)
        for u in graph.get(v, []):
            if u not in visited:
                stack.append(u)
    return result
```

## BFS Iterative (Queue) Template

```python
def bfs_template(graph, start):
    from collections import deque
    visited = {start}
    q = deque([start])
    while q:
        v = q.popleft()
        for u in graph.get(v, []):
            if u not in visited:
                visited.add(u)
                q.append(u)
```

## DFS vs BFS Comparison Table

| Aspect | DFS | BFS |
|--------|-----|-----|
| Data structure | Stack | Queue |
| Order | Depth-first | Level-order |
| Shortest path (unweighted) | No | Yes |
| Memory | O(h) for recursion | O(w) for queue |
| Cycle detection | Yes | Yes |
| Topological sort | Yes (post-order) | Kahn's |

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

## Flood Fill

```python
def flood_fill(image, sr, sc, color):
    orig = image[sr][sc]
    if orig == color:
        return image
    rows, cols = len(image), len(image[0])
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or image[r][c] != orig:
            return
        image[r][c] = color
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)
    dfs(sr, sc)
    return image
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
            if board[r][c] == 'O':
                board[r][c] = 'X'
            elif board[r][c] == 'T':
                board[r][c] = 'O'
```

## Number of Provinces

```python
def find_circle_num(is_connected):
    n = len(is_connected)
    visited = [False] * n
    def dfs(i):
        visited[i] = True
        for j in range(n):
            if is_connected[i][j] and not visited[j]:
                dfs(j)
    count = 0
    for i in range(n):
        if not visited[i]:
            count += 1
            dfs(i)
    return count
```

## Keys and Rooms

```python
def can_visit_all_rooms(rooms):
    visited = {0}
    stack = [0]
    while stack:
        v = stack.pop()
        for key in rooms[v]:
            if key not in visited:
                visited.add(key)
                stack.append(key)
    return len(visited) == len(rooms)
```

## All Paths Source to Target

```python
def all_paths_source_target(graph):
    n = len(graph)
    result = []
    def dfs(v, path):
        if v == n - 1:
            result.append(path[:])
            return
        for u in graph[v]:
            path.append(u)
            dfs(u, path)
            path.pop()
    dfs(0, [0])
    return result
```

## Find If Path Exists

```python
def valid_path(n, edges, source, destination):
    from collections import defaultdict, deque
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    visited = {source}
    q = deque([source])
    while q:
        v = q.popleft()
        if v == destination:
            return True
        for u in adj[v]:
            if u not in visited:
                visited.add(u)
                q.append(u)
    return False
```

## Is Graph Bipartite

```python
def is_bipartite(graph):
    from collections import deque
    color = {}
    for i in range(len(graph)):
        if i in color:
            continue
        color[i] = 0
        q = deque([i])
        while q:
            v = q.popleft()
            for u in graph[v]:
                if u not in color:
                    color[u] = 1 - color[v]
                    q.append(u)
                elif color[u] == color[v]:
                    return False
    return True
```

## Possible Bipartition

```python
def possible_bipartition(n, dislikes):
    from collections import defaultdict, deque
    adj = defaultdict(list)
    for a, b in dislikes:
        adj[a].append(b)
        adj[b].append(a)
    color = {}
    for i in range(1, n + 1):
        if i in color:
            continue
        color[i] = 0
        q = deque([i])
        while q:
            v = q.popleft()
            for u in adj[v]:
                if u not in color:
                    color[u] = 1 - color[v]
                    q.append(u)
                elif color[u] == color[v]:
                    return False
    return True
```

## Course Schedule (Can Finish)

```python
def can_finish(num_courses, prerequisites):
    from collections import defaultdict, deque
    adj = defaultdict(list)
    indegree = [0] * num_courses
    for a, b in prerequisites:
        adj[b].append(a)
        indegree[a] += 1
    q = deque(i for i in range(num_courses) if indegree[i] == 0)
    count = 0
    while q:
        v = q.popleft()
        count += 1
        for u in adj[v]:
            indegree[u] -= 1
            if indegree[u] == 0:
                q.append(u)
    return count == num_courses
```

## Clone Graph

```python
def clone_graph(node):
    if not node:
        return None
    mapping = {}
    def dfs(original):
        if original in mapping:
            return mapping[original]
        copy = Node(original.val)
        mapping[original] = copy
        for neighbor in original.neighbors:
            copy.neighbors.append(dfs(neighbor))
        return copy
    return dfs(node)
```

## Reconstruct Itinerary

```python
def find_itinerary(tickets):
    from collections import defaultdict
    graph = defaultdict(list)
    for a, b in tickets:
        graph[a].append(b)
    for key in graph:
        graph[key].sort(reverse=True)
    result = []
    def dfs(airport):
        while graph[airport]:
            dfs(graph[airport].pop())
        result.append(airport)
    dfs("JFK")
    return result[::-1]
```

## Evaluate Division

```python
def calc_equation(equations, values, queries):
    from collections import defaultdict
    graph = defaultdict(dict)
    for (a, b), v in zip(equations, values):
        graph[a][b] = v
        graph[b][a] = 1 / v
    def dfs(start, end, visited):
        if start not in graph or end not in graph:
            return -1.0
        if start == end:
            return 1.0
        visited.add(start)
        for neighbor, val in graph[start].items():
            if neighbor not in visited:
                result = dfs(neighbor, end, visited)
                if result != -1.0:
                    return val * result
        return -1.0
    return [dfs(a, b, set()) for a, b in queries]
```

## Word Ladder

```python
def ladder_length(begin_word, end_word, word_list):
    from collections import deque
    word_set = set(word_list)
    if end_word not in word_set:
        return 0
    q = deque([(begin_word, 1)])
    visited = {begin_word}
    while q:
        word, dist = q.popleft()
        if word == end_word:
            return dist
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]
                if next_word in word_set and next_word not in visited:
                    visited.add(next_word)
                    q.append((next_word, dist + 1))
    return 0
```

## Open the Lock

```python
def open_lock(deadends, target):
    from collections import deque
    dead = set(deadends)
    if "0000" in dead:
        return -1
    q = deque([("0000", 0)])
    visited = {"0000"}
    while q:
        state, moves = q.popleft()
        if state == target:
            return moves
        for i in range(4):
            for d in (-1, 1):
                digit = (int(state[i]) + d) % 10
                next_state = state[:i] + str(digit) + state[i+1:]
                if next_state not in dead and next_state not in visited:
                    visited.add(next_state)
                    q.append((next_state, moves + 1))
    return -1
```

## Minimum Genetic Mutation

```python
def min_mutation(start, end, bank):
    from collections import deque
    bank_set = set(bank)
    if end not in bank_set:
        return -1
    q = deque([(start, 0)])
    visited = {start}
    genes = "ACGT"
    while q:
        gene, steps = q.popleft()
        if gene == end:
            return steps
        for i in range(8):
            for c in genes:
                if c != gene[i]:
                    next_gene = gene[:i] + c + gene[i+1:]
                    if next_gene in bank_set and next_gene not in visited:
                        visited.add(next_gene)
                        q.append((next_gene, steps + 1))
    return -1
```

## Accounts Merge

```python
def accounts_merge(accounts):
    from collections import defaultdict
    parent = {}
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        parent[find(x)] = find(y)
    email_to_name = {}
    for acc in accounts:
        name = acc[0]
        for email in acc[1:]:
            if email not in parent:
                parent[email] = email
            email_to_name[email] = name
            union(acc[1], email)
    groups = defaultdict(list)
    for email in parent:
        groups[find(email)].append(email)
    return [[email_to_name[root]] + sorted(emails) for root, emails in groups.items()]
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

## Reorder Routes

```python
def min_reorder(n, connections):
    from collections import defaultdict, deque
    graph = defaultdict(list)
    for a, b in connections:
        graph[a].append((b, 1))
        graph[b].append((a, 0))
    q = deque([0])
    visited = {0}
    count = 0
    while q:
        v = q.popleft()
        for u, direction in graph[v]:
            if u not in visited:
                visited.add(u)
                count += direction
                q.append(u)
    return count
```

## Find Eventual Safe States

```python
def eventual_safe_nodes(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {}
    def dfs(v):
        if v in color:
            return color[v] == BLACK
        color[v] = GRAY
        for u in graph[v]:
            if not dfs(u):
                return False
        color[v] = BLACK
        return True
    return [i for i in range(len(graph)) if dfs(i)]
```

## Detect Cycles in 2D Grid

```python
def contains_cycle(grid):
    rows, cols = len(grid), len(grid[0])
    visited = set()
    def dfs(r, c, pr, pc):
        visited.add((r, c))
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == grid[r][c]:
                if (nr, nc) not in visited:
                    if dfs(nr, nc, r, c):
                        return True
                elif (nr, nc) != (pr, pc):
                    return True
        return False
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and dfs(r, c, -1, -1):
                return True
    return False
```
