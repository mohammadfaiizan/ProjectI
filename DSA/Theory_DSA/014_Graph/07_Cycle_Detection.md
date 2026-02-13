# Graph - Cycle Detection

## Theory

A cycle exists when there is a path from a vertex back to itself. Undirected: back edge to non-parent. Directed: back edge to ancestor in DFS tree (gray node in 3-color scheme). Union-Find can detect cycles in undirected graphs by checking if both endpoints of an edge share the same root.

## Cycle in Undirected (DFS Back Edge to Non-Parent)

```python
def has_cycle_undirected_dfs(graph):
    visited = set()
    def dfs(v, parent):
        visited.add(v)
        for u in graph.get(v, []):
            if u not in visited:
                if dfs(u, v):
                    return True
            elif u != parent:
                return True
        return False
    for v in graph:
        if v not in visited and dfs(v, -1):
            return True
    return False
```

## Cycle in Undirected (BFS)

```python
def has_cycle_undirected_bfs(graph):
    from collections import deque
    visited = set()
    for start in graph:
        if start in visited:
            continue
        q = deque([(start, -1)])
        visited.add(start)
        while q:
            v, parent = q.popleft()
            for u in graph.get(v, []):
                if u not in visited:
                    visited.add(u)
                    q.append((u, v))
                elif u != parent:
                    return True
    return False
```

## Cycle in Undirected (Union-Find)

```python
def has_cycle_undirected_uf(edges):
    parent = {}
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return True
        parent[px] = py
        return False
    for u, v in edges:
        if union(u, v):
            return True
    return False
```

## Cycle in Directed (DFS 3-Color White/Gray/Black)

```python
def has_cycle_directed(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {}
    def dfs(v):
        color[v] = GRAY
        for u in graph.get(v, []):
            if u not in color:
                if dfs(u):
                    return True
            elif color[u] == GRAY:
                return True
        color[v] = BLACK
        return False
    for v in graph:
        if v not in color and dfs(v):
            return True
    return False
```

## Cycle in Directed (Kahn's - Incomplete Processing)

```python
def has_cycle_kahn(graph):
    from collections import deque
    indegree = {v: 0 for v in graph}
    for v in graph:
        for u in graph[v]:
            indegree[u] = indegree.get(u, 0) + 1
    q = deque(v for v in indegree if indegree[v] == 0)
    count = 0
    while q:
        v = q.popleft()
        count += 1
        for u in graph.get(v, []):
            indegree[u] -= 1
            if indegree[u] == 0:
                q.append(u)
    return count != len(graph)
```

## Redundant Connection (Union-Find)

```python
def find_redundant_connection(edges):
    parent = list(range(len(edges) + 1))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return True
        parent[px] = py
        return False
    for u, v in edges:
        if union(u, v):
            return [u, v]
    return []
```

## Redundant Connection II (Directed)

```python
def find_redundant_directed_connection(edges):
    n = len(edges)
    parent = list(range(n + 1))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    candidate1 = candidate2 = None
    for u, v in edges:
        if parent[v] != v:
            candidate1 = [parent[v], v]
            candidate2 = [u, v]
        else:
            parent[v] = u
    if candidate1 is None:
        parent = list(range(n + 1))
        for u, v in edges:
            if find(u) == find(v):
                return [u, v]
            parent[find(u)] = find(v)
    else:
        parent = list(range(n + 1))
        for u, v in edges:
            if [u, v] == candidate2:
                continue
            if find(u) == find(v):
                return candidate1
            parent[find(u)] = find(v)
        return candidate2
    return []
```

## Detect Negative Cycle (Bellman-Ford)

```python
def has_negative_cycle(edges, n):
    dist = [0] * n
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return True
    return False
```

## Course Schedule (Cycle Check)

```python
def can_finish(num_courses, prerequisites):
    from collections import defaultdict
    adj = defaultdict(list)
    for a, b in prerequisites:
        adj[b].append(a)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {}
    def dfs(v):
        color[v] = GRAY
        for u in adj[v]:
            if u not in color:
                if dfs(u):
                    return True
            elif color[u] == GRAY:
                return True
        color[v] = BLACK
        return False
    for i in range(num_courses):
        if i not in color and dfs(i):
            return False
    return True
```
