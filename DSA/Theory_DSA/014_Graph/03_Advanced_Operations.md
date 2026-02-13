# Graph - Advanced Operations

## Check Bipartite (2-Coloring BFS)

```python
def is_bipartite_bfs(graph):
    from collections import deque
    color = {}
    for start in graph["adj"]:
        if start in color:
            continue
        color[start] = 0
        q = deque([start])
        while q:
            v = q.popleft()
            for u in graph["adj"].get(v, []):
                if u not in color:
                    color[u] = 1 - color[v]
                    q.append(u)
                elif color[u] == color[v]:
                    return False
    return True
```

## Check Bipartite (2-Coloring DFS)

```python
def is_bipartite_dfs(graph):
    color = {}
    def dfs(v, c):
        color[v] = c
        for u in graph["adj"].get(v, []):
            if u not in color:
                if not dfs(u, 1 - c):
                    return False
            elif color[u] == c:
                return False
        return True
    for start in graph["adj"]:
        if start not in color and not dfs(start, 0):
            return False
    return True
```

## Find All Connected Components

```python
def find_connected_components(graph):
    visited = set()
    components = []
    def dfs(v, comp):
        visited.add(v)
        comp.append(v)
        for u in graph["adj"].get(v, []):
            if u not in visited:
                dfs(u, comp)
    for v in graph["adj"]:
        if v not in visited:
            comp = []
            dfs(v, comp)
            components.append(comp)
    return components
```

## Find Strongly Connected Components (Kosaraju's)

```python
def kosaraju_scc(graph):
    visited = set()
    stack = []
    def dfs1(v):
        visited.add(v)
        for u in graph["adj"].get(v, []):
            if u not in visited:
                dfs1(u)
        stack.append(v)
    for v in graph["adj"]:
        if v not in visited:
            dfs1(v)
    transpose = {}
    for v in graph["adj"]:
        transpose[v] = []
    for v in graph["adj"]:
        for u in graph["adj"][v]:
            transpose[u].append(v)
    visited.clear()
    sccs = []
    def dfs2(v, comp):
        visited.add(v)
        comp.append(v)
        for u in transpose.get(v, []):
            if u not in visited:
                dfs2(u, comp)
    while stack:
        v = stack.pop()
        if v not in visited:
            comp = []
            dfs2(v, comp)
            sccs.append(comp)
    return sccs
```

## Find Strongly Connected Components (Tarjan's)

```python
def tarjan_scc(graph):
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    on_stack = {}
    sccs = []
    def strongconnect(v):
        index[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for u in graph["adj"].get(v, []):
            if u not in index:
                strongconnect(u)
                lowlinks[v] = min(lowlinks[v], lowlinks[u])
            elif on_stack.get(u):
                lowlinks[v] = min(lowlinks[v], index[u])
        if lowlinks[v] == index[v]:
            comp = []
            while True:
                u = stack.pop()
                on_stack[u] = False
                comp.append(u)
                if u == v:
                    break
            sccs.append(comp)
    for v in graph["adj"]:
        if v not in index:
            strongconnect(v)
    return sccs
```

## Find Bridges (Tarjan's)

```python
def find_bridges(graph):
    index_counter = [0]
    index = {}
    low = {}
    bridges = []
    def dfs(v, parent):
        index[v] = index_counter[0]
        low[v] = index_counter[0]
        index_counter[0] += 1
        for u in graph["adj"].get(v, []):
            if u not in index:
                dfs(u, v)
                low[v] = min(low[v], low[u])
                if low[u] > index[v]:
                    bridges.append((v, u))
            elif u != parent:
                low[v] = min(low[v], index[u])
    for v in graph["adj"]:
        if v not in index:
            dfs(v, -1)
    return bridges
```

## Find Articulation Points

```python
def find_articulation_points(graph):
    index_counter = [0]
    index = {}
    low = {}
    ap = set()
    def dfs(v, parent):
        index[v] = index_counter[0]
        low[v] = index_counter[0]
        index_counter[0] += 1
        children = 0
        for u in graph["adj"].get(v, []):
            if u not in index:
                children += 1
                dfs(u, v)
                low[v] = min(low[v], low[u])
                if parent == -1 and children > 1:
                    ap.add(v)
                if parent != -1 and low[u] >= index[v]:
                    ap.add(v)
            elif u != parent:
                low[v] = min(low[v], index[u])
    for v in graph["adj"]:
        if v not in index:
            dfs(v, -1)
    return list(ap)
```

## Euler Path and Circuit (Hierholzer's)

```python
def euler_circuit(graph):
    from collections import defaultdict
    adj = defaultdict(list)
    for v in graph["adj"]:
        adj[v] = list(graph["adj"][v])
    stack = [next(iter(graph["adj"]))]
    path = []
    while stack:
        v = stack[-1]
        if adj[v]:
            u = adj[v].pop()
            stack.append(u)
        else:
            path.append(stack.pop())
    return path[::-1]
```

## Hamiltonian Path/Cycle (Backtracking)

```python
def hamiltonian_path(graph, n):
    def backtrack(v, path, visited):
        if len(path) == n:
            return path[:]
        for u in graph["adj"].get(v, []):
            if u not in visited:
                visited.add(u)
                path.append(u)
                result = backtrack(u, path, visited)
                if result:
                    return result
                path.pop()
                visited.remove(u)
        return None
    for start in graph["adj"]:
        result = backtrack(start, [start], {start})
        if result:
            return result
    return None
```

## Graph Coloring (m-Coloring)

```python
def m_coloring(graph, m):
    color = {}
    def is_safe(v, c):
        for u in graph["adj"].get(v, []):
            if color.get(u) == c:
                return False
        return True
    def backtrack(v):
        if v not in graph["adj"]:
            return True
        for c in range(m):
            if is_safe(v, c):
                color[v] = c
                if backtrack(v + 1 if v + 1 in graph["adj"] else next((u for u in graph["adj"] if u not in color), None)):
                    return True
                del color[v]
        return False
    return backtrack(next(iter(graph["adj"]))) if graph["adj"] else True
```

## Maximum Flow (Ford-Fulkerson / Edmonds-Karp Overview)

Ford-Fulkerson uses augmenting paths; Edmonds-Karp uses BFS to find shortest augmenting path. Time O(VE^2) for Edmonds-Karp. Residual graph, capacity, flow. Min-cut = max-flow by max-flow min-cut theorem.

## Minimum Cut

Minimum cut separates source and sink with minimum total capacity of cut edges. Found by running max-flow, then BFS/DFS from source in residual graph to identify reachable nodes. Cut = edges from reachable to non-reachable.

## Clone Graph (Deep Copy)

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

## Transpose Graph

```python
def transpose_graph(graph):
    trans = {"adj": {}, "directed": True}
    for v in graph["adj"]:
        trans["adj"][v] = []
    for v in graph["adj"]:
        for u in graph["adj"][v]:
            trans["adj"][u].append(v)
    return trans
```

## Check If Graph Contains Cycle (Undirected)

```python
def has_cycle_undirected(graph):
    visited = set()
    def dfs(v, parent):
        visited.add(v)
        for u in graph["adj"].get(v, []):
            if u not in visited:
                if dfs(u, v):
                    return True
            elif u != parent:
                return True
        return False
    for v in graph["adj"]:
        if v not in visited and dfs(v, -1):
            return True
    return False
```

## Check If Graph Contains Cycle (Directed)

```python
def has_cycle_directed(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {}
    def dfs(v):
        color[v] = GRAY
        for u in graph["adj"].get(v, []):
            if u not in color:
                if dfs(u):
                    return True
            elif color[u] == GRAY:
                return True
        color[v] = BLACK
        return False
    for v in graph["adj"]:
        if v not in color and dfs(v):
            return True
    return False
```
