# Graph - Minimum Spanning Tree

## Theory

MST connects all vertices with minimum total edge weight. Has exactly V-1 edges. Cut property: minimum weight edge crossing any cut belongs to MST. Used for network design, clustering.

## MST Properties (V-1 Edges, Cut Property)

- Tree: acyclic, connected. Exactly V-1 edges.
- Cut property: For any cut (S, V-S), the minimum weight edge crossing the cut is in some MST.
- Cycle property: For any cycle, the maximum weight edge is not in any MST (if unique).

## Kruskal's (Sort Edges + Union-Find)

```python
def kruskal(n, edges):
    edges.sort(key=lambda x: x[2])
    parent = list(range(n))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        parent[px] = py
        return True
    mst_cost = 0
    for u, v, w in edges:
        if union(u, v):
            mst_cost += w
    return mst_cost
```

## Prim's (Grow from Node Min-Heap)

```python
def prim(n, graph, start=0):
    import heapq
    visited = [False] * n
    pq = [(0, start)]
    mst_cost = 0
    while pq:
        w, v = heapq.heappop(pq)
        if visited[v]:
            continue
        visited[v] = True
        mst_cost += w
        for u, weight in graph[v]:
            if not visited[u]:
                heapq.heappush(pq, (weight, u))
    return mst_cost
```

## Comparison Table

| Aspect | Kruskal | Prim |
|--------|---------|------|
| Approach | Add minimum edge globally | Grow from vertex |
| Data structure | Sort + Union-Find | Min-heap |
| Best for | Sparse (E small) | Dense (E large) |
| Time | O(E log E) | O(E log V) with heap |
| Implementation | Simpler | Slightly more complex |

## Second Minimum Spanning Tree Concept

Find MST, then for each edge not in MST, add it and remove the max edge in the cycle formed. The minimum such replacement gives 2nd MST. Or: exclude each MST edge one at a time and compute new MST.

## Min Cost Connect All Points

```python
def min_cost_connect_points(points):
    n = len(points)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
            edges.append((d, i, j))
    edges.sort()
    parent = list(range(n))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    cost = 0
    for d, u, v in edges:
        if find(u) != find(v):
            parent[find(u)] = find(v)
            cost += d
    return cost
```

## Connecting Cities

```python
def minimum_cost(n, connections):
    connections.sort(key=lambda x: x[2])
    parent = list(range(n + 1))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    cost = 0
    count = 0
    for u, v, w in connections:
        if find(u) != find(v):
            parent[find(u)] = find(v)
            cost += w
            count += 1
    return cost if count == n - 1 else -1
```

## Optimize Water Distribution

```python
def min_cost_to_supply_water(n, wells, pipes):
    edges = [(w, 0, i + 1) for i, w in enumerate(wells)]
    edges.extend((w, u, v) for u, v, w in pipes)
    edges.sort()
    parent = list(range(n + 1))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    cost = 0
    for w, u, v in edges:
        if find(u) != find(v):
            parent[find(u)] = find(v)
            cost += w
    return cost
```

## Critical and Pseudo-Critical Edges in MST

```python
def find_critical_and_pseudo_critical_edges(n, edges):
    for i, e in enumerate(edges):
        e.append(i)
    edges.sort(key=lambda x: x[2])
    def get_mst(skip=None, force=None):
        parent = list(range(n))
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        cost = 0
        if force is not None:
            u, v, w = edges[force][:3]
            parent[find(u)] = find(v)
            cost += w
        for i, (u, v, w, _) in enumerate(edges):
            if i == skip:
                continue
            if find(u) != find(v):
                parent[find(u)] = find(v)
                cost += w
        return cost if len(set(find(i) for i in range(n))) == 1 else float('inf')
    mst_cost = get_mst()
    critical = []
    pseudo = []
    for i in range(len(edges)):
        if get_mst(skip=i) > mst_cost:
            critical.append(edges[i][3])
        elif get_mst(force=i) == mst_cost:
            pseudo.append(edges[i][3])
    return [critical, pseudo]
```

## Boruvka's Overview

Also called Sollin's algorithm. In each round, each component picks its minimum outgoing edge. Merge components. Repeat until one component. O(E log V). Useful for parallel/distributed settings.
