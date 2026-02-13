# Graph - Medium Problems

## 1. Course Schedule

**Description**: numCourses and prerequisites. Can you finish all courses?

**Approach**: Build directed graph, detect cycle. Kahn's algorithm or DFS 3-color. If topological order exists, can finish.

```python
def canFinish(numCourses, prerequisites):
    from collections import defaultdict, deque
    adj = defaultdict(list)
    in_deg = [0] * numCourses
    for a, b in prerequisites:
        adj[b].append(a)
        in_deg[a] += 1
    q = deque(i for i in range(numCourses) if in_deg[i] == 0)
    count = 0
    while q:
        node = q.popleft()
        count += 1
        for nei in adj[node]:
            in_deg[nei] -= 1
            if in_deg[nei] == 0:
                q.append(nei)
    return count == numCourses
```

Time: O(n + e) | Space: O(n)

---

## 2. Course Schedule II

**Description**: Return valid order to take all courses, or empty if impossible.

**Approach**: Kahn's algorithm. Return topological order.

```python
def findOrder(numCourses, prerequisites):
    from collections import defaultdict, deque
    adj = defaultdict(list)
    in_deg = [0] * numCourses
    for a, b in prerequisites:
        adj[b].append(a)
        in_deg[a] += 1
    q = deque(i for i in range(numCourses) if in_deg[i] == 0)
    order = []
    while q:
        node = q.popleft()
        order.append(node)
        for nei in adj[node]:
            in_deg[nei] -= 1
            if in_deg[nei] == 0:
                q.append(nei)
    return order if len(order) == numCourses else []
```

Time: O(n + e) | Space: O(n)

---

## 3. Number of Connected Components

**Description**: Given n and edges, count connected components.

**Approach**: Union-Find or DFS/BFS. Count number of times we start from unvisited.

```python
def countComponents(n, edges):
    parent = list(range(n))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        parent[find(a)] = find(b)
    for a, b in edges:
        union(a, b)
    return len(set(find(i) for i in range(n)))
```

Time: O(n + e) | Space: O(n)

---

## 4. Redundant Connection

**Description**: Tree plus one extra edge. Find edge that creates cycle.

**Approach**: Union-Find. First edge that connects already-connected vertices is answer.

```python
def findRedundantConnection(edges):
    parent = list(range(len(edges) + 1))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for a, b in edges:
        if find(a) == find(b):
            return [a, b]
        parent[find(a)] = find(b)
    return []
```

Time: O(n) | Space: O(n)

---

## 5. Redundant Connection II

**Description**: Rooted tree plus one edge. Find edge to remove to get valid rooted tree.

**Approach**: Two cases: node with two parents, or cycle. Union-Find with parent tracking.

```python
def findRedundantDirectedConnection(edges):
    parent = list(range(len(edges) + 1))
    cand1, cand2 = None, None
    for a, b in edges:
        if parent[b] != b:
            cand1, cand2 = [parent[b], b], [a, b]
        else:
            parent[b] = a
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    parent = list(range(len(edges) + 1))
    for a, b in edges:
        if [a, b] == cand2:
            continue
        if find(a) == find(b):
            return cand1 if cand1 else [a, b]
        parent[find(a)] = find(b)
    return cand2
```

Time: O(n) | Space: O(n)

---

## 6. Accounts Merge

**Description**: Merge accounts that share email. Return merged account lists.

**Approach**: Union-Find on emails. Group by root, sort emails per group.

```python
def accountsMerge(accounts):
    parent = {}
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        parent[find(a)] = find(b)
    name = {}
    for acc in accounts:
        for i in range(1, len(acc)):
            parent[acc[i]] = acc[i]
            name[acc[i]] = acc[0]
    for acc in accounts:
        for i in range(2, len(acc)):
            union(acc[1], acc[i])
    from collections import defaultdict
    groups = defaultdict(list)
    for email in parent:
        groups[find(email)].append(email)
    return [[name[em]] + sorted(emails) for em, emails in groups.items()]
```

Time: O(n * k log k) | Space: O(n * k)

---

## 7. Evaluate Division

**Description**: Equations a/b = value. Answer queries c/d.

**Approach**: Build weighted graph. DFS/BFS to find path and multiply weights.

```python
def calcEquation(equations, values, queries):
    from collections import defaultdict
    g = defaultdict(dict)
    for (a, b), v in zip(equations, values):
        g[a][b], g[b][a] = v, 1.0 / v
    def query(c, d):
        if c not in g or d not in g:
            return -1.0
        if c == d:
            return 1.0
        visited, q = set(), [(c, 1.0)]
        while q:
            node, prod = q.pop()
            if node == d:
                return prod
            visited.add(node)
            for nei, w in g[node].items():
                if nei not in visited:
                    q.append((nei, prod * w))
        return -1.0
    return [query(c, d) for c, d in queries]
```

Time: O(q * (v + e)) | Space: O(v + e)

---

## 8. Word Ladder

**Description**: Transform beginWord to endWord changing one letter at a time. Words from list.

**Approach**: BFS. Each state is a word. Neighbors are words differing by one letter.

```python
def ladderLength(beginWord, endWord, wordList):
    words = set(wordList)
    if endWord not in words:
        return 0
    q, dist = [beginWord], 1
    while q:
        nq = []
        for w in q:
            if w == endWord:
                return dist
            for i in range(len(w)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    nw = w[:i] + c + w[i+1:]
                    if nw in words:
                        words.discard(nw)
                        nq.append(nw)
        q, dist = nq, dist + 1
    return 0
```

Time: O(n * m * 26) | Space: O(n)

---

## 9. Word Ladder II

**Description**: Find all shortest transformation sequences from begin to end.

**Approach**: BFS to find shortest distance. DFS to reconstruct all paths of that length.

```python
def findLadders(beginWord, endWord, wordList):
    words = set(wordList)
    if endWord not in words:
        return []
    layer = {beginWord: [[beginWord]]}
    while layer:
        nlayer = {}
        for w in layer:
            if w == endWord:
                return layer[w]
            for i in range(len(w)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    nw = w[:i] + c + w[i+1:]
                    if nw in words:
                        nlayer.setdefault(nw, []).extend(path + [nw] for path in layer[w])
        words -= set(nlayer.keys())
        layer = nlayer
    return []
```

Time: O(n * m * 26) | Space: O(paths)

---

## 10. Surrounded Regions

**Description**: Flip 'O' to 'X' if not connected to border.

**Approach**: DFS from border 'O's, mark as temporary. Flip remaining 'O' to 'X'.

```python
def solve(board):
    if not board:
        return
    m, n = len(board), len(board[0])
    def dfs(r, c):
        if 0 <= r < m and 0 <= c < n and board[r][c] == 'O':
            board[r][c] = 'T'
            for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                dfs(r+dr, c+dc)
    for i in range(m):
        dfs(i, 0)
        dfs(i, n-1)
    for j in range(n):
        dfs(0, j)
        dfs(m-1, j)
    for i in range(m):
        for j in range(n):
            board[i][j] = 'X' if board[i][j] != 'T' else 'O'
```

Time: O(m * n) | Space: O(m * n)

---

## 11. Clone Graph

**Description**: Deep copy graph with same structure.

**Approach**: DFS with node-to-copy mapping.

```python
def cloneGraph(node):
    if not node:
        return None
    seen = {}
    def dfs(n):
        if n.val in seen:
            return seen[n.val]
        copy = Node(n.val)
        seen[n.val] = copy
        copy.neighbors = [dfs(nei) for nei in n.neighbors]
        return copy
    return dfs(node)
```

Time: O(n) | Space: O(n)

---

## 12. Pacific Atlantic Water Flow

**Description**: Grid heights. Which cells can flow to both oceans (top/left and bottom/right)?

**Approach**: DFS from Pacific border, DFS from Atlantic border. Intersection of reachable sets.

```python
def pacificAtlantic(heights):
    if not heights:
        return []
    m, n = len(heights), len(heights[0])
    pac, atl = set(), set()
    def dfs(r, c, visited):
        visited.add((r, c))
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < m and 0 <= nc < n and (nr,nc) not in visited and heights[nr][nc] >= heights[r][c]:
                dfs(nr, nc, visited)
    for i in range(m):
        dfs(i, 0, pac)
        dfs(i, n-1, atl)
    for j in range(n):
        dfs(0, j, pac)
        dfs(m-1, j, atl)
    return list(pac & atl)
```

Time: O(m * n) | Space: O(m * n)

---

## 13. Number of Islands

**Description**: Count connected '1' regions in grid.

**Approach**: DFS/BFS for each unvisited '1'.

```python
def numIslands(grid):
    m, n, count = len(grid), len(grid[0]), 0
    def dfs(r, c):
        if 0 <= r < m and 0 <= c < n and grid[r][c] == '1':
            grid[r][c] = '0'
            for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                dfs(r+dr, c+dc)
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                dfs(i, j)
                count += 1
    return count
```

Time: O(m * n) | Space: O(m * n)

---

## 14. Max Area of Island

**Description**: Find largest connected component of 1s.

**Approach**: DFS return area, track max.

```python
def maxAreaOfIsland(grid):
    m, n, res = len(grid), len(grid[0]), 0
    def dfs(r, c):
        if 0 <= r < m and 0 <= c < n and grid[r][c] == 1:
            grid[r][c] = 0
            return 1 + sum(dfs(r+dr, c+dc) for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)])
        return 0
    for i in range(m):
        for j in range(n):
            res = max(res, dfs(i, j))
    return res
```

Time: O(m * n) | Space: O(m * n)

---

## 15. Rotting Oranges

**Description**: Grid with fresh (1) and rotten (2) oranges. Minutes until all rotten?

**Approach**: Multi-source BFS from all rotten. Expand each minute.

```python
def orangesRotting(grid):
    m, n = len(grid), len(grid[0])
    q = [(i, j) for i in range(m) for j in range(n) if grid[i][j] == 2]
    fresh = sum(1 for i in range(m) for j in range(n) if grid[i][j] == 1)
    mins = 0
    while q and fresh:
        nq = []
        for r, c in q:
            for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1:
                    grid[nr][nc] = 2
                    fresh -= 1
                    nq.append((nr, nc))
        q, mins = nq, mins + 1
    return mins if fresh == 0 else -1
```

Time: O(m * n) | Space: O(m * n)

---

## 16. 01 Matrix

**Description**: For each cell, distance to nearest 0.

**Approach**: Multi-source BFS from all 0s. Propagate distance.

```python
def updateMatrix(mat):
    m, n = len(mat), len(mat[0])
    q = [(i, j) for i in range(m) for j in range(n) if mat[i][j] == 0]
    visited = set(q)
    dist = 0
    while q:
        nq = []
        for r, c in q:
            mat[r][c] = dist
            for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < m and 0 <= nc < n and (nr,nc) not in visited:
                    visited.add((nr, nc))
                    nq.append((nr, nc))
        q, dist = nq, dist + 1
    return mat
```

Time: O(m * n) | Space: O(m * n)

---

## 17. Shortest Path in Binary Matrix

**Description**: 0-1 grid, shortest path from (0,0) to (n-1,n-1) using 8 directions, only 0 cells.

**Approach**: BFS with 8-directional moves.

```python
def shortestPathBinaryMatrix(grid):
    if grid[0][0] or grid[-1][-1]:
        return -1
    n, q, grid[0][0] = len(grid), [(0, 0)], 1
    steps = 1
    while q:
        nq = []
        for r, c in q:
            if r == n-1 and c == n-1:
                return steps
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                        grid[nr][nc] = 1
                        nq.append((nr, nc))
        q, steps = nq, steps + 1
    return -1
```

Time: O(n^2) | Space: O(n^2)

---

## 18. Network Delay Time

**Description**: Times (u, v, w). Signal from k. Time for all nodes to receive?

**Approach**: Dijkstra from k. Return max distance or -1 if unreachable.

```python
def networkDelayTime(times, n, k):
    import heapq
    from collections import defaultdict
    g = defaultdict(list)
    for u, v, w in times:
        g[u].append((v, w))
    dist = {}
    pq = [(0, k)]
    while pq:
        d, node = heapq.heappop(pq)
        if node in dist:
            continue
        dist[node] = d
        for nei, w in g[node]:
            if nei not in dist:
                heapq.heappush(pq, (d + w, nei))
    return max(dist.values()) if len(dist) == n else -1
```

Time: O((v+e) log v) | Space: O(v)

---

## 19. Cheapest Flights Within K Stops

**Description**: Find cheapest path from src to dst with at most k stops.

**Approach**: BFS/Dijkstra variant. Track (node, cost, stops). Relax with stop limit.

```python
def findCheapestPrice(n, flights, src, dst, k):
    import heapq
    from collections import defaultdict
    g = defaultdict(list)
    for u, v, w in flights:
        g[u].append((v, w))
    pq = [(0, src, 0)]
    while pq:
        cost, node, stops = heapq.heappop(pq)
        if node == dst:
            return cost
        if stops <= k:
            for nei, w in g[node]:
                heapq.heappush(pq, (cost + w, nei, stops + 1))
    return -1
```

Time: O((v+e) * k) | Space: O(v)

---

## 20. Path With Maximum Probability

**Description**: Undirected graph, edge success probabilities. Max probability path from start to end.

**Approach**: Dijkstra with max-heap (or negate for min-heap). Multiply probabilities.

```python
def maxProbability(n, edges, succProb, start, end):
    import heapq
    from collections import defaultdict
    g = defaultdict(list)
    for (u, v), p in zip(edges, succProb):
        g[u].append((v, p))
        g[v].append((u, p))
    pq = [(-1, start)]
    best = {}
    while pq:
        prob, node = heapq.heappop(pq)
        prob = -prob
        if node == end:
            return prob
        if node in best:
            continue
        best[node] = prob
        for nei, p in g[node]:
            if nei not in best:
                heapq.heappush(pq, (-prob * p, nei))
    return 0
```

Time: O((v+e) log v) | Space: O(v)

---

## 21. Reorder Routes to Make All Paths Lead to City Zero

**Description**: Directed edges. Some point to 0, some away. Min edges to flip so all lead to 0?

**Approach**: BFS from 0. Count edges that point toward 0 (direction 1 in connections).

```python
def minReorder(n, connections):
    from collections import defaultdict
    g = defaultdict(list)
    for a, b in connections:
        g[a].append((b, 1))
        g[b].append((a, 0))
    res, q, visited = 0, [0], {0}
    while q:
        node = q.pop()
        for nei, d in g[node]:
            if nei not in visited:
                visited.add(nei)
                res += d
                q.append(nei)
    return res
```

Time: O(n) | Space: O(n)

---

## 22. Find Eventual Safe States

**Description**: Directed graph. Node is safe if all paths lead to terminal (no outgoing). Find all safe nodes.

**Approach**: Reverse graph, start from terminals. Or DFS cycle detection; nodes not in cycle are safe.

```python
def eventualSafeNodes(graph):
    n = len(graph)
    out_deg = [len(adj) for adj in graph]
    rev = [[] for _ in range(n)]
    for i, adj in enumerate(graph):
        for j in adj:
            rev[j].append(i)
    q = [i for i in range(n) if out_deg[i] == 0]
    safe = [False] * n
    while q:
        node = q.pop()
        safe[node] = True
        for nei in rev[node]:
            out_deg[nei] -= 1
            if out_deg[nei] == 0:
                q.append(nei)
    return [i for i in range(n) if safe[i]]
```

Time: O(n + e) | Space: O(n)

---

## 23. Making a Large Island

**Description**: Grid of 0s and 1s. Can change one 0 to 1. Max island size?

**Approach**: DFS to label islands and get sizes. For each 0, sum sizes of adjacent distinct islands + 1.

```python
def largestIsland(grid):
    m, n = len(grid), len(grid[0])
    sizes, label = {}, 2
    def dfs(r, c, lb):
        if 0 <= r < m and 0 <= c < n and grid[r][c] == 1:
            grid[r][c] = lb
            return 1 + sum(dfs(r+dr, c+dc, lb) for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)])
        return 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                sizes[label] = dfs(i, j, label)
                label += 1
    res = max(sizes.values()) if sizes else 1
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 0:
                seen = set()
                for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                    ni, nj = i+dr, j+dc
                    if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] > 1:
                        seen.add(grid[ni][nj])
                res = max(res, 1 + sum(sizes[lb] for lb in seen))
    return res
```

Time: O(m * n) | Space: O(m * n)

---

## 24. Shortest Bridge

**Description**: Two islands of 1s in sea of 0s. Min 0s to flip to connect islands?

**Approach**: DFS to find first island. Multi-source BFS from first island until hitting second.

```python
def shortestBridge(grid):
    m, n = len(grid), len(grid[0])
    def dfs(r, c, island):
        if 0 <= r < m and 0 <= c < n and grid[r][c] == 1:
            grid[r][c] = 2
            island.append((r, c))
            for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                dfs(r+dr, c+dc, island)
    island = []
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                dfs(i, j, island)
                break
        if island:
            break
    q, steps = island[:], 0
    while q:
        nq = []
        for r, c in q:
            for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < m and 0 <= nc < n:
                    if grid[nr][nc] == 1:
                        return steps
                    if grid[nr][nc] == 0:
                        grid[nr][nc] = 2
                        nq.append((nr, nc))
        q, steps = nq, steps + 1
    return steps
```

Time: O(m * n) | Space: O(m * n)

---

## 25. Number of Enclaves

**Description**: Count 1s that cannot reach border.

**Approach**: DFS from border 1s to mark reachable. Count remaining 1s.

```python
def numEnclaves(grid):
    m, n = len(grid), len(grid[0])
    def dfs(r, c):
        if 0 <= r < m and 0 <= c < n and grid[r][c] == 1:
            grid[r][c] = 0
            for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                dfs(r+dr, c+dc)
    for i in range(m):
        dfs(i, 0)
        dfs(i, n-1)
    for j in range(n):
        dfs(0, j)
        dfs(m-1, j)
    return sum(grid[i][j] for i in range(m) for j in range(n))
```

Time: O(m * n) | Space: O(m * n)

---

## 26. Count Sub-Islands

**Description**: grid1 and grid2. Count islands in grid2 that are fully covered by grid1.

**Approach**: For each grid2 island, DFS and check all cells are 1 in grid1.

```python
def countSubIslands(grid1, grid2):
    m, n = len(grid1), len(grid1[0])
    def dfs(r, c):
        if 0 <= r < m and 0 <= c < n and grid2[r][c] == 1:
            grid2[r][c] = 0
            valid = grid1[r][c] == 1
            for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                valid = dfs(r+dr, c+dc) and valid
            return valid
        return True
    count = 0
    for i in range(m):
        for j in range(n):
            if grid2[i][j] == 1:
                count += dfs(i, j)
    return count
```

Time: O(m * n) | Space: O(m * n)

---

## 27. Minimum Genetic Mutation

**Description**: Transform start gene to end via valid mutations (one char change, must be in bank).

**Approach**: BFS like word ladder. States are genes.

```python
def minMutation(start, end, bank):
    bank = set(bank)
    if end not in bank:
        return -1
    q, steps = [start], 0
    while q:
        nq = []
        for g in q:
            if g == end:
                return steps
            for i in range(len(g)):
                for c in 'ACGT':
                    ng = g[:i] + c + g[i+1:]
                    if ng in bank:
                        bank.discard(ng)
                        nq.append(ng)
        q, steps = nq, steps + 1
    return -1
```

Time: O(n * 8) | Space: O(n)

---

## 28. Open the Lock

**Description**: 4-digit lock. Deadends and target. Min moves to reach target?

**Approach**: BFS. States are 4-digit strings. Neighbors: increment/decrement each digit.

```python
def openLock(deadends, target):
    dead = set(deadends)
    if '0000' in dead:
        return -1
    q, visited = ['0000'], {'0000'}
    steps = 0
    while q:
        nq = []
        for s in q:
            if s == target:
                return steps
            for i in range(4):
                for d in (-1, 1):
                    ns = s[:i] + str((int(s[i]) + d) % 10) + s[i+1:]
                    if ns not in dead and ns not in visited:
                        visited.add(ns)
                        nq.append(ns)
        q, steps = nq, steps + 1
    return -1
```

Time: O(10^4) | Space: O(10^4)

---

## 29. Satisfiability of Equality Equations

**Description**: Equations like a==b or a!=b. Are all satisfiable?

**Approach**: Union-Find for ==. Check != pairs are in different sets.

```python
def equationsPossible(equations):
    parent = {c: c for c in 'abcdefghijklmnopqrstuvwxyz'}
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        parent[find(a)] = find(b)
    for eq in equations:
        if eq[1] == '=':
            union(eq[0], eq[3])
    for eq in equations:
        if eq[1] == '!' and find(eq[0]) == find(eq[3]):
            return False
    return True
```

Time: O(n) | Space: O(1)

---

## 30. Smallest String With Swaps

**Description**: String and pairs of indices. Can swap any pair unlimited times. Lexicographically smallest?

**Approach**: Union-Find to group swappable indices. Sort chars in each group, place at sorted indices.

```python
def smallestStringWithSwaps(s, pairs):
    n = len(s)
    parent = list(range(n))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        parent[find(a)] = find(b)
    for a, b in pairs:
        union(a, b)
    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    res = [''] * n
    for indices in groups.values():
        chars = sorted(s[i] for i in indices)
        for i, idx in enumerate(sorted(indices)):
            res[idx] = chars[i]
    return ''.join(res)
```

Time: O(n log n) | Space: O(n)

---

# Hard Problems

## 1. Critical Connections in a Network

**Description**: Find all bridges (edges whose removal increases connected components).

**Approach**: Tarjan's bridge-finding algorithm. Low-link values.

```python
def criticalConnections(n, connections):
    from collections import defaultdict
    g = defaultdict(list)
    for a, b in connections:
        g[a].append(b)
        g[b].append(a)
    low, disc, res = [0]*n, [0]*n, []
    def dfs(node, parent, time):
        disc[node] = low[node] = time
        for nei in g[node]:
            if disc[nei] == 0:
                dfs(nei, node, time + 1)
                low[node] = min(low[node], low[nei])
                if low[nei] > disc[node]:
                    res.append([node, nei])
            elif nei != parent:
                low[node] = min(low[node], disc[nei])
    dfs(0, -1, 1)
    return res
```

Time: O(n + e) | Space: O(n)

---

## 2. Word Ladder II

**Description**: All shortest transformation sequences from begin to end word.

**Approach**: BFS to get distances. DFS to build paths. Or BFS with path tracking.

```python
def findLadders(beginWord, endWord, wordList):
    words = set(wordList)
    if endWord not in words:
        return []
    layer = {beginWord: [[beginWord]]}
    while layer:
        nlayer = {}
        for w in layer:
            if w == endWord:
                return layer[w]
            for i in range(len(w)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    nw = w[:i] + c + w[i+1:]
                    if nw in words:
                        nlayer.setdefault(nw, []).extend(path + [nw] for path in layer[w])
        words -= set(nlayer.keys())
        layer = nlayer
    return []
```

Time: O(n * m * 26) | Space: O(paths)

---

## 3. Minimum Cost to Connect All Points

**Description**: Connect all points with minimum total Manhattan distance.

**Approach**: Kruskal's MST. Edges are all pairs with Manhattan weight.

```python
def minCostConnectPoints(points):
    n = len(points)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            d = abs(points[i][0]-points[j][0]) + abs(points[i][1]-points[j][1])
            edges.append((d, i, j))
    edges.sort()
    parent = list(range(n))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    res, count = 0, 0
    for d, a, b in edges:
        if find(a) != find(b):
            parent[find(a)] = find(b)
            res += d
            count += 1
            if count == n - 1:
                break
    return res
```

Time: O(n^2 log n) | Space: O(n^2)

---

## 4. Swim in Rising Water

**Description**: Grid with heights. Water rises. Earliest time to swim from (0,0) to (n-1,n-1)?

**Approach**: Dijkstra. Edge weight is max of current time and cell height.

```python
def swimInWater(grid):
    import heapq
    n = len(grid)
    pq = [(grid[0][0], 0, 0)]
    seen = {(0, 0)}
    while pq:
        t, r, c = heapq.heappop(pq)
        if r == n-1 and c == n-1:
            return t
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < n and 0 <= nc < n and (nr,nc) not in seen:
                seen.add((nr, nc))
                heapq.heappush(pq, (max(t, grid[nr][nc]), nr, nc))
    return 0
```

Time: O(n^2 log n) | Space: O(n^2)

---

## 5. Path With Minimum Effort

**Description**: Grid heights. Path effort = max absolute difference along path. Min effort from top-left to bottom-right?

**Approach**: Dijkstra with effort as cost. Relax: new_effort = max(current, abs(diff)).

```python
def minimumEffortPath(heights):
    import heapq
    m, n = len(heights), len(heights[0])
    pq = [(0, 0, 0)]
    seen = {}
    while pq:
        eff, r, c = heapq.heappop(pq)
        if r == m-1 and c == n-1:
            return eff
        if (r, c) in seen and seen[(r, c)] <= eff:
            continue
        seen[(r, c)] = eff
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < m and 0 <= nc < n:
                ne = max(eff, abs(heights[nr][nc] - heights[r][c]))
                if (nr, nc) not in seen or seen[(nr, nc)] > ne:
                    heapq.heappush(pq, (ne, nr, nc))
    return 0
```

Time: O(m*n*log(m*n)) | Space: O(m*n)

---

## 6. Shortest Path Visiting All Nodes

**Description**: Undirected graph. Shortest path that visits every node at least once.

**Approach**: BFS with state (node, bitmask of visited). Target: all bits set.

```python
def shortestPathLength(graph):
    n = len(graph)
    target = (1 << n) - 1
    q = [(i, 1 << i) for i in range(n)]
    seen = set(q)
    steps = 0
    while q:
        nq = []
        for node, mask in q:
            if mask == target:
                return steps
            for nei in graph[node]:
                nm = mask | (1 << nei)
                if (nei, nm) not in seen:
                    seen.add((nei, nm))
                    nq.append((nei, nm))
        q, steps = nq, steps + 1
    return 0
```

Time: O(n * 2^n) | Space: O(n * 2^n)

---

## 7. Number of Restricted Paths

**Description**: Weighted graph. Restricted path: each step must go to node with strictly greater distance to n. Count restricted paths from 1 to n.

**Approach**: Dijkstra from n to get distances. DP: paths[v] = sum paths[u] for u with dist[u] > dist[v].

```python
def countRestrictedPaths(n, edges):
    from collections import defaultdict
    import heapq
    g = defaultdict(list)
    for u, v, w in edges:
        g[u].append((v, w))
        g[v].append((u, w))
    dist = [float('inf')] * (n + 1)
    dist[n] = 0
    pq = [(0, n)]
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist[node]:
            continue
        for nei, w in g[node]:
            if dist[nei] > d + w:
                dist[nei] = d + w
                heapq.heappush(pq, (dist[nei], nei))
    dp = [0] * (n + 1)
    dp[n] = 1
    for node in sorted(range(1, n+1), key=lambda x: dist[x], reverse=True):
        for nei, _ in g[node]:
            if dist[nei] < dist[node]:
                dp[node] = (dp[node] + dp[nei]) % (10**9+7)
    return dp[1]
```

Time: O((v+e) log v + v log v) | Space: O(v)

---

## 8. Parallel Courses III

**Description**: n courses, relations, time per course. Min time to finish all (parallel allowed with dependencies).

**Approach**: Topological sort. dist[u] = time[u] + max(dist[v]) for prerequisites v.

```python
def minimumTime(n, relations, time):
    from collections import defaultdict, deque
    adj = defaultdict(list)
    in_deg = [0] * (n + 1)
    for a, b in relations:
        adj[a].append(b)
        in_deg[b] += 1
    dist = [0] * (n + 1)
    for i in range(1, n + 1):
        dist[i] = time[i - 1]
    q = deque(i for i in range(1, n + 1) if in_deg[i] == 0)
    while q:
        node = q.popleft()
        for nei in adj[node]:
            dist[nei] = max(dist[nei], dist[node] + time[nei - 1])
            in_deg[nei] -= 1
            if in_deg[nei] == 0:
                q.append(nei)
    return max(dist)
```

Time: O(n + e) | Space: O(n)

---

## 9. Sequence Reconstruction

**Description**: Check if nums is the unique shortest supersequence of all sequences.

**Approach**: Build graph from sequences. Topological sort. Check unique order and matches nums.

```python
def sequenceReconstruction(nums, sequences):
    n = len(nums)
    adj = {i: set() for i in range(1, n + 1)}
    in_deg = [0] * (n + 1)
    for seq in sequences:
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if b not in adj[a]:
                adj[a].add(b)
                in_deg[b] += 1
    q = [i for i in range(1, n + 1) if in_deg[i] == 0]
    order = []
    while q:
        if len(q) > 1:
            return False
        node = q.pop()
        order.append(node)
        for nei in adj[node]:
            in_deg[nei] -= 1
            if in_deg[nei] == 0:
                q.append(nei)
    return order == nums
```

Time: O(n + sum(seq)) | Space: O(n)

---

## 10. Alien Dictionary

**Description**: Sorted dictionary of alien language. Derive character order.

**Approach**: Build graph from adjacent word comparisons. Topological sort.

```python
def alienOrder(words):
    from collections import defaultdict, deque
    adj = defaultdict(set)
    in_deg = {c: 0 for w in words for c in w}
    for i in range(len(words) - 1):
        a, b = words[i], words[i + 1]
        for j in range(min(len(a), len(b))):
            if a[j] != b[j]:
                if b[j] not in adj[a[j]]:
                    adj[a[j]].add(b[j])
                    in_deg[b[j]] += 1
                break
        else:
            if len(a) > len(b):
                return ""
    q = deque(c for c in in_deg if in_deg[c] == 0)
    res = []
    while q:
        c = q.popleft()
        res.append(c)
        for nei in adj[c]:
            in_deg[nei] -= 1
            if in_deg[nei] == 0:
                q.append(nei)
    return ''.join(res) if len(res) == len(in_deg) else ""
```

Time: O(n * L) | Space: O(1)

---

## 11. Minimum Height Trees

**Description**: Tree of n nodes. Which nodes as root give minimum height?

**Approach**: Repeatedly remove leaves. Last 1 or 2 nodes are centers.

```python
def findMinHeightTrees(n, edges):
    if n == 1:
        return [0]
    from collections import defaultdict
    g = defaultdict(set)
    for a, b in edges:
        g[a].add(b)
        g[b].add(a)
    leaves = [i for i in range(n) if len(g[i]) == 1]
    while n > 2:
        n -= len(leaves)
        nleaves = []
        for node in leaves:
            nei = g[node].pop()
            g[nei].discard(node)
            if len(g[nei]) == 1:
                nleaves.append(nei)
        leaves = nleaves
    return leaves
```

Time: O(n) | Space: O(n)

---

## 12. Longest Increasing Path in a Matrix

**Description**: Grid. Longest strictly increasing path (any direction).

**Approach**: DFS with memoization. Path from (r,c) = 1 + max(neighbors with smaller value).

```python
def longestIncreasingPath(matrix):
    if not matrix:
        return 0
    m, n = len(matrix), len(matrix[0])
    memo = {}
    def dfs(r, c):
        if (r, c) in memo:
            return memo[(r, c)]
        best = 1
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < m and 0 <= nc < n and matrix[nr][nc] > matrix[r][c]:
                best = max(best, 1 + dfs(nr, nc))
        memo[(r, c)] = best
        return best
    return max(dfs(i, j) for i in range(m) for j in range(n))
```

Time: O(m * n) | Space: O(m * n)

---

## 13. Count Subtrees With Max Distance Between Cities

**Description**: Tree of n nodes. For each d from 1 to n-1, count subtrees with diameter d.

**Approach**: For each node as root, DFS to compute subtree diameters. Complex tree DP.

```python
def countSubgraphsForEachDiameter(n, edges):
    from collections import defaultdict
    g = defaultdict(list)
    for a, b in edges:
        g[a-1].append(b-1)
        g[b-1].append(a-1)
    res = [0] * (n - 1)
    for mask in range(1, 1 << n):
        nodes = [i for i in range(n) if mask & (1 << i)]
        if len(nodes) < 2:
            continue
        def bfs(start):
            dist, q = {start: 0}, [start]
            while q:
                node = q.pop()
                for nei in g[node]:
                    if nei in nodes and nei not in dist:
                        dist[nei] = dist[node] + 1
                        q.append(nei)
            return dist
        d1 = bfs(nodes[0])
        if len(d1) != len(nodes):
            continue
        end = max(d1, key=d1.get)
        d2 = bfs(end)
        diam = max(d2.values())
        if diam > 0:
            res[diam - 1] += 1
    return res
```

Time: O(2^n * n) | Space: O(n)

---

## 14. Graph Connectivity With Threshold

**Description**: n nodes. Nodes i and j connected if gcd(i,j) > threshold. Queries: are a and b connected?

**Approach**: Union-Find. For each pair (i,j) with gcd > threshold, union. Answer queries with find.

```python
def areConnected(n, threshold, queries):
    import math
    parent = list(range(n + 1))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        parent[find(a)] = find(b)
    for i in range(threshold + 1, n + 1):
        for j in range(2 * i, n + 1, i):
            union(i, j)
    return [find(a) == find(b) for a, b in queries]
```

Time: O(n log n + q) | Space: O(n)

---

## 15. Minimum Cost to Reach Destination in Time

**Description**: Graph with edge times and node fees. Max time limit. Min cost to reach n-1 within time?

**Approach**: Dijkstra-like. State (node, time). Minimize cost. Relax with time constraint.

```python
def minCost(maxTime, edges, passingFees):
    from collections import defaultdict
    import heapq
    g = defaultdict(list)
    for a, b, t in edges:
        g[a].append((b, t))
        g[b].append((a, t))
    n = len(passingFees)
    best = {}
    pq = [(passingFees[0], 0, 0)]
    while pq:
        cost, node, time = heapq.heappop(pq)
        if node == n - 1:
            return cost
        if time > maxTime:
            continue
        if (node, time) in best and best[(node, time)] <= cost:
            continue
        best[(node, time)] = cost
        for nei, t in g[node]:
            nt = time + t
            if nt <= maxTime:
                heapq.heappush(pq, (cost + passingFees[nei], nei, nt))
    return -1
```

Time: O((v+e) * maxTime * log) | Space: O(v * maxTime)
