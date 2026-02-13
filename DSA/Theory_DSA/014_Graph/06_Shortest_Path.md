# Graph - Shortest Path

## Theory

Shortest path algorithms find the minimum-cost path between vertices. Choice depends on graph type: unweighted (BFS), non-negative weights (Dijkstra), negative weights (Bellman-Ford), all pairs (Floyd-Warshall).

## BFS Shortest Path (Unweighted O(V+E))

```python
def bfs_shortest_path(graph, start, end):
    from collections import deque
    if start == end:
        return 0
    visited = {start}
    q = deque([(start, 0)])
    while q:
        v, dist = q.popleft()
        for u in graph.get(v, []):
            if u == end:
                return dist + 1
            if u not in visited:
                visited.add(u)
                q.append((u, dist + 1))
    return -1
```

## Dijkstra's (Non-Negative Min-Heap O((V+E)log V))

```python
def dijkstra(graph, start, n):
    import heapq
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, v = heapq.heappop(pq)
        if d > dist[v]:
            continue
        for u, w in graph.get(v, []):
            if dist[v] + w < dist[u]:
                dist[u] = dist[v] + w
                heapq.heappush(pq, (dist[u], u))
    return dist
```

## Bellman-Ford (Negative Weights O(V*E))

```python
def bellman_ford(edges, n, start):
    dist = [float('inf')] * n
    dist[start] = 0
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    return dist
```

## Floyd-Warshall (All Pairs O(V^3))

```python
def floyd_warshall(n, edges):
    dist = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = w
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist
```

## 0-1 BFS (Deque)

```python
def zero_one_bfs(graph, start, n):
    from collections import deque
    dist = [float('inf')] * n
    dist[start] = 0
    dq = deque([start])
    while dq:
        v = dq.popleft()
        for u, w in graph.get(v, []):
            if dist[v] + w < dist[u]:
                dist[u] = dist[v] + w
                if w == 0:
                    dq.appendleft(u)
                else:
                    dq.append(u)
    return dist
```

## SPFA Overview

Shortest Path Faster Algorithm: Bellman-Ford variant using queue. Relax edges only when distance improves. Average O(E) but worst O(VE). Handles negative weights.

## A* Overview

Informed search using heuristic h(n). f(n) = g(n) + h(n). Optimal if heuristic is admissible. Used for grid pathfinding, games.

## Network Delay Time

```python
def network_delay_time(times, n, k):
    import heapq
    from collections import defaultdict
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))
    dist = [float('inf')] * (n + 1)
    dist[k] = 0
    pq = [(0, k)]
    while pq:
        d, v = heapq.heappop(pq)
        if d > dist[v]:
            continue
        for u, w in graph[v]:
            if dist[v] + w < dist[u]:
                dist[u] = dist[v] + w
                heapq.heappush(pq, (dist[u], u))
    result = max(dist[1:])
    return result if result != float('inf') else -1
```

## Cheapest Flights Within K Stops

```python
def find_cheapest_price(n, flights, src, dst, k):
    from collections import defaultdict
    graph = defaultdict(list)
    for u, v, w in flights:
        graph[u].append((v, w))
    dist = [float('inf')] * n
    dist[src] = 0
    for _ in range(k + 1):
        new_dist = dist[:]
        for u, v, w in flights:
            if dist[u] != float('inf') and dist[u] + w < new_dist[v]:
                new_dist[v] = dist[u] + w
        dist = new_dist
    return dist[dst] if dist[dst] != float('inf') else -1
```

## Path With Minimum Effort

```python
def minimum_effort_path(heights):
    import heapq
    rows, cols = len(heights), len(heights[0])
    dist = [[float('inf')] * cols for _ in range(rows)]
    dist[0][0] = 0
    pq = [(0, 0, 0)]
    while pq:
        effort, r, c = heapq.heappop(pq)
        if r == rows - 1 and c == cols - 1:
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

## Path With Maximum Probability

```python
def max_probability(n, edges, succ_prob, start, end):
    import heapq
    from collections import defaultdict
    graph = defaultdict(list)
    for (u, v), p in zip(edges, succ_prob):
        graph[u].append((v, p))
        graph[v].append((u, p))
    prob = [0.0] * n
    prob[start] = 1.0
    pq = [(-1.0, start)]
    while pq:
        p, v = heapq.heappop(pq)
        p = -p
        if v == end:
            return p
        if p < prob[v]:
            continue
        for u, edge_p in graph[v]:
            new_p = p * edge_p
            if new_p > prob[u]:
                prob[u] = new_p
                heapq.heappush(pq, (-new_p, u))
    return 0.0
```

## Shortest Path in Binary Matrix

```python
def shortest_path_binary_matrix(grid):
    from collections import deque
    n = len(grid)
    if grid[0][0] or grid[n-1][n-1]:
        return -1
    q = deque([(0, 0, 1)])
    grid[0][0] = 1
    while q:
        r, c, dist = q.popleft()
        if r == n-1 and c == n-1:
            return dist
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                    grid[nr][nc] = 1
                    q.append((nr, nc, dist + 1))
    return -1
```

## Shortest Path Alternating Colors

```python
def shortest_alternating_paths(n, red_edges, blue_edges):
    from collections import defaultdict, deque
    red = defaultdict(list)
    blue = defaultdict(list)
    for u, v in red_edges:
        red[u].append(v)
    for u, v in blue_edges:
        blue[u].append(v)
    result = [-1] * n
    result[0] = 0
    q = deque([(0, 0, None)])
    visited = set()
    visited.add((0, None))
    while q:
        v, dist, prev_color = q.popleft()
        if result[v] == -1:
            result[v] = dist
        if prev_color != 0:
            for u in red[v]:
                if (u, 0) not in visited:
                    visited.add((u, 0))
                    q.append((u, dist + 1, 0))
        if prev_color != 1:
            for u in blue[v]:
                if (u, 1) not in visited:
                    visited.add((u, 1))
                    q.append((u, dist + 1, 1))
    return result
```

## Minimum Cost to Reach Destination in Time

```python
def min_cost(max_time, edges, passing_fees):
    n = len(passing_fees)
    from collections import defaultdict
    graph = defaultdict(list)
    for u, v, t in edges:
        graph[u].append((v, t))
        graph[v].append((u, t))
    import heapq
    dist = [[float('inf')] * (max_time + 1) for _ in range(n)]
    dist[0][0] = passing_fees[0]
    pq = [(passing_fees[0], 0, 0)]
    while pq:
        cost, v, t = heapq.heappop(pq)
        if v == n - 1:
            return cost
        if cost > dist[v][t]:
            continue
        for u, time in graph[v]:
            nt = t + time
            if nt <= max_time:
                nc = cost + passing_fees[u]
                if nc < dist[u][nt]:
                    dist[u][nt] = nc
                    heapq.heappush(pq, (nc, u, nt))
    return -1
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

## City With Smallest Neighbors at Threshold

```python
def find_the_city(n, edges, distance_threshold):
    dist = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = dist[v][u] = w
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    min_count = float('inf')
    result = -1
    for i in range(n):
        count = sum(1 for j in range(n) if i != j and dist[i][j] <= distance_threshold)
        if count <= min_count:
            min_count = count
            result = i
    return result
```

## Shortest Path Visiting All Nodes (Bitmask BFS)

```python
def shortest_path_length(graph):
    from collections import deque
    n = len(graph)
    target = (1 << n) - 1
    q = deque((i, 1 << i, 0) for i in range(n))
    visited = {(i, 1 << i) for i in range(n)}
    while q:
        v, mask, dist = q.popleft()
        if mask == target:
            return dist
        for u in graph[v]:
            new_mask = mask | (1 << u)
            if (u, new_mask) not in visited:
                visited.add((u, new_mask))
                q.append((u, new_mask, dist + 1))
    return 0
```

## Number of Restricted Paths

```python
def count_restricted_paths(n, edges):
    from collections import defaultdict
    import heapq
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    dist = [float('inf')] * (n + 1)
    dist[n] = 0
    pq = [(0, n)]
    while pq:
        d, v = heapq.heappop(pq)
        if d > dist[v]:
            continue
        for u, w in graph[v]:
            if dist[v] + w < dist[u]:
                dist[u] = dist[v] + w
                heapq.heappush(pq, (dist[u], u))
    MOD = 10**9 + 7
    order = sorted(range(1, n + 1), key=lambda x: dist[x])
    dp = [0] * (n + 1)
    dp[n] = 1
    for v in order:
        for u, _ in graph[v]:
            if dist[u] < dist[v]:
                dp[v] = (dp[v] + dp[u]) % MOD
    return dp[1]
```
