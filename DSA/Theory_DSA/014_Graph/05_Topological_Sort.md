# Graph - Topological Sort

## Theory

A topological ordering of a directed graph is a linear ordering of vertices such that for every directed edge (u, v), u comes before v. Only exists for DAGs (Directed Acyclic Graphs). Used for dependency resolution, build order, course scheduling.

## Kahn's Algorithm (BFS In-Degree)

```python
def topological_sort_kahn(graph):
    from collections import deque
    indegree = {v: 0 for v in graph["adj"]}
    for v in graph["adj"]:
        for u in graph["adj"][v]:
            indegree[u] = indegree.get(u, 0) + 1
    q = deque(v for v in indegree if indegree[v] == 0)
    result = []
    while q:
        v = q.popleft()
        result.append(v)
        for u in graph["adj"].get(v, []):
            indegree[u] -= 1
            if indegree[u] == 0:
                q.append(u)
    return result if len(result) == len(graph["adj"]) else []
```

## DFS-Based (Reverse Post-Order)

```python
def topological_sort_dfs(graph):
    visited = set()
    result = []
    def dfs(v):
        visited.add(v)
        for u in graph["adj"].get(v, []):
            if u not in visited:
                dfs(u)
        result.append(v)
    for v in graph["adj"]:
        if v not in visited:
            dfs(v)
    return result[::-1]
```

## Detect If Topological Order Exists

```python
def has_topological_order(graph):
    order = topological_sort_kahn(graph)
    return len(order) == len(graph["adj"])
```

## All Topological Orderings (Backtracking)

```python
def all_topological_orders(graph):
    indegree = {v: 0 for v in graph["adj"]}
    for v in graph["adj"]:
        for u in graph["adj"][v]:
            indegree[u] = indegree.get(u, 0) + 1
    result = []
    def backtrack(path, remaining_indegree):
        if len(path) == len(graph["adj"]):
            result.append(path[:])
            return
        for v in graph["adj"]:
            if v not in path and remaining_indegree[v] == 0:
                path.append(v)
                for u in graph["adj"].get(v, []):
                    remaining_indegree[u] -= 1
                backtrack(path, remaining_indegree)
                path.pop()
                for u in graph["adj"].get(v, []):
                    remaining_indegree[u] += 1
    backtrack([], indegree.copy())
    return result
```

## Course Schedule I

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

## Course Schedule II

```python
def find_order(num_courses, prerequisites):
    from collections import defaultdict, deque
    adj = defaultdict(list)
    indegree = [0] * num_courses
    for a, b in prerequisites:
        adj[b].append(a)
        indegree[a] += 1
    q = deque(i for i in range(num_courses) if indegree[i] == 0)
    result = []
    while q:
        v = q.popleft()
        result.append(v)
        for u in adj[v]:
            indegree[u] -= 1
            if indegree[u] == 0:
                q.append(u)
    return result if len(result) == num_courses else []
```

## Alien Dictionary

```python
def alien_order(words):
    from collections import defaultdict, deque
    adj = defaultdict(set)
    indegree = {}
    for w in words:
        for c in w:
            indegree[c] = 0
    for i in range(len(words) - 1):
        a, b = words[i], words[i+1]
        for j in range(min(len(a), len(b))):
            if a[j] != b[j]:
                if b[j] not in adj[a[j]]:
                    adj[a[j]].add(b[j])
                    indegree[b[j]] = indegree.get(b[j], 0) + 1
                break
        else:
            if len(a) > len(b):
                return ""
    q = deque(c for c in indegree if indegree[c] == 0)
    result = []
    while q:
        c = q.popleft()
        result.append(c)
        for n in adj[c]:
            indegree[n] -= 1
            if indegree[n] == 0:
                q.append(n)
    return "".join(result) if len(result) == len(indegree) else ""
```

## Sequence Reconstruction

```python
def sequence_reconstruction(nums, sequences):
    from collections import defaultdict, deque
    n = len(nums)
    adj = defaultdict(set)
    indegree = {i: 0 for i in range(1, n + 1)}
    for seq in sequences:
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i+1]
            if b not in adj[a]:
                adj[a].add(b)
                indegree[b] += 1
    q = deque(i for i in indegree if indegree[i] == 0)
    result = []
    while q:
        if len(q) > 1:
            return False
        v = q.popleft()
        result.append(v)
        for u in adj[v]:
            indegree[u] -= 1
            if indegree[u] == 0:
                q.append(u)
    return result == nums
```

## Minimum Height Trees

```python
def find_min_height_trees(n, edges):
    if n == 1:
        return [0]
    from collections import defaultdict, deque
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    leaves = deque(i for i in range(n) if len(adj[i]) == 1)
    remaining = n
    while remaining > 2:
        size = len(leaves)
        remaining -= size
        for _ in range(size):
            v = leaves.popleft()
            u = adj[v].pop()
            adj[u].remove(v)
            if len(adj[u]) == 1:
                leaves.append(u)
    return list(leaves)
```

## Parallel Courses

```python
def minimum_semesters(n, relations):
    from collections import defaultdict, deque
    adj = defaultdict(list)
    indegree = [0] * (n + 1)
    for a, b in relations:
        adj[a].append(b)
        indegree[b] += 1
    q = deque(i for i in range(1, n + 1) if indegree[i] == 0)
    semesters = 0
    count = 0
    while q:
        semesters += 1
        for _ in range(len(q)):
            v = q.popleft()
            count += 1
            for u in adj[v]:
                indegree[u] -= 1
                if indegree[u] == 0:
                    q.append(u)
    return semesters if count == n else -1
```

## Parallel Courses III (With Time)

```python
def minimum_time(n, relations, time):
    from collections import defaultdict, deque
    adj = defaultdict(list)
    indegree = [0] * (n + 1)
    for a, b in relations:
        adj[a].append(b)
        indegree[b] += 1
    dist = [0] * (n + 1)
    for i in range(1, n + 1):
        dist[i] = time[i - 1]
    q = deque(i for i in range(1, n + 1) if indegree[i] == 0)
    while q:
        v = q.popleft()
        for u in adj[v]:
            dist[u] = max(dist[u], dist[v] + time[u - 1])
            indegree[u] -= 1
            if indegree[u] == 0:
                q.append(u)
    return max(dist)
```

## Longest Path in DAG

```python
def longest_path_dag(graph, weights):
    from collections import deque
    indegree = {v: 0 for v in graph["adj"]}
    for v in graph["adj"]:
        for u in graph["adj"][v]:
            indegree[u] = indegree.get(u, 0) + 1
    dist = {v: weights.get(v, 0) for v in graph["adj"]}
    q = deque(v for v in indegree if indegree[v] == 0)
    while q:
        v = q.popleft()
        for u in graph["adj"].get(v, []):
            dist[u] = max(dist[u], dist[v] + weights.get(u, 0))
            indegree[u] -= 1
            if indegree[u] == 0:
                q.append(u)
    return max(dist.values())
```

## Sort Items by Groups

```python
def sort_items(n, m, group, before_items):
    from collections import defaultdict, deque
    for i in range(n):
        if group[i] == -1:
            group[i] = m + i
    item_adj = defaultdict(list)
    group_adj = defaultdict(set)
    for i in range(n):
        for j in before_items[i]:
            item_adj[j].append(i)
            if group[i] != group[j]:
                group_adj[group[j]].add(group[i])
    group_indegree = defaultdict(int)
    for g in group_adj:
        for u in group_adj[g]:
            group_indegree[u] += 1
    all_groups = set(group)
    q = deque(g for g in all_groups if group_indegree.get(g, 0) == 0)
    group_order = []
    while q:
        g = q.popleft()
        group_order.append(g)
        for u in group_adj.get(g, []):
            group_indegree[u] -= 1
            if group_indegree[u] == 0:
                q.append(u)
    if len(group_order) != len(all_groups):
        return []
    result = []
    for g in group_order:
        items = [i for i in range(n) if group[i] == g]
        indegree_g = {i: sum(1 for j in before_items[i] if group[j] == g) for i in items}
        q = deque(i for i in items if indegree_g[i] == 0)
        while q:
            v = q.popleft()
            result.append(v)
            for u in item_adj[v]:
                if group[u] == g:
                    indegree_g[u] -= 1
                    if indegree_g[u] == 0:
                        q.append(u)
    return result if len(result) == n else []
```

## Build Order

```python
def build_order(projects, dependencies):
    from collections import defaultdict, deque
    adj = defaultdict(list)
    indegree = {p: 0 for p in projects}
    for a, b in dependencies:
        adj[a].append(b)
        indegree[b] += 1
    q = deque(p for p in projects if indegree[p] == 0)
    result = []
    while q:
        v = q.popleft()
        result.append(v)
        for u in adj[v]:
            indegree[u] -= 1
            if indegree[u] == 0:
                q.append(u)
    return result if len(result) == len(projects) else None
```
