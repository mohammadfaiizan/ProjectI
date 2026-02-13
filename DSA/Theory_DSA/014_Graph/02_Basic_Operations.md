# Graph - Basic Operations

## Create Graph (Adjacency List using Dict)

```python
def create_graph(directed=False):
    return {"adj": {}, "directed": directed}

def create_graph_from_edges(edges, directed=False):
    graph = {"adj": {}, "directed": directed}
    for u, v in edges:
        if u not in graph["adj"]:
            graph["adj"][u] = []
        graph["adj"][u].append(v)
        if not directed:
            if v not in graph["adj"]:
                graph["adj"][v] = []
            graph["adj"][v].append(u)
    return graph
```

## Add Vertex

```python
def add_vertex(graph, v):
    if v not in graph["adj"]:
        graph["adj"][v] = []
```

## Add Edge (Directed and Undirected)

```python
def add_edge(graph, u, v):
    if u not in graph["adj"]:
        graph["adj"][u] = []
    graph["adj"][u].append(v)
    if not graph["directed"]:
        if v not in graph["adj"]:
            graph["adj"][v] = []
        graph["adj"][v].append(u)
```

## Remove Vertex

```python
def remove_vertex(graph, v):
    if v not in graph["adj"]:
        return
    for u in graph["adj"]:
        if u != v and v in graph["adj"][u]:
            graph["adj"][u].remove(v)
    del graph["adj"][v]
```

## Remove Edge

```python
def remove_edge(graph, u, v):
    if u in graph["adj"] and v in graph["adj"][u]:
        graph["adj"][u].remove(v)
    if not graph["directed"] and v in graph["adj"] and u in graph["adj"][v]:
        graph["adj"][v].remove(u)
```

## Get All Neighbors

```python
def get_neighbors(graph, v):
    return graph["adj"].get(v, [])
```

## Check If Edge Exists

```python
def has_edge(graph, u, v):
    return v in graph["adj"].get(u, [])
```

## Get Degree / In-Degree / Out-Degree

```python
def get_degree(graph, v):
    return len(graph["adj"].get(v, []))

def get_in_degree(graph, v):
    count = 0
    for u in graph["adj"]:
        if v in graph["adj"][u]:
            count += 1
    return count

def get_out_degree(graph, v):
    return len(graph["adj"].get(v, []))
```

## Count Vertices

```python
def count_vertices(graph):
    return len(graph["adj"])
```

## Count Edges

```python
def count_edges(graph):
    total = sum(len(neighbors) for neighbors in graph["adj"].values())
    return total if graph["directed"] else total // 2
```

## Check If Connected (DFS)

```python
def is_connected_dfs(graph):
    if not graph["adj"]:
        return True
    start = next(iter(graph["adj"]))
    visited = set()
    def dfs(v):
        visited.add(v)
        for u in graph["adj"].get(v, []):
            if u not in visited:
                dfs(u)
    dfs(start)
    return len(visited) == len(graph["adj"])
```

## Check If Connected (BFS)

```python
def is_connected_bfs(graph):
    if not graph["adj"]:
        return True
    from collections import deque
    start = next(iter(graph["adj"]))
    visited = {start}
    q = deque([start])
    while q:
        v = q.popleft()
        for u in graph["adj"].get(v, []):
            if u not in visited:
                visited.add(u)
                q.append(u)
    return len(visited) == len(graph["adj"])
```

## DFS Traversal (Recursive)

```python
def dfs_recursive(graph, start):
    visited = set()
    result = []
    def dfs(v):
        visited.add(v)
        result.append(v)
        for u in graph["adj"].get(v, []):
            if u not in visited:
                dfs(u)
    dfs(start)
    return result
```

## BFS Traversal (Queue)

```python
def bfs(graph, start):
    from collections import deque
    visited = {start}
    result = []
    q = deque([start])
    while q:
        v = q.popleft()
        result.append(v)
        for u in graph["adj"].get(v, []):
            if u not in visited:
                visited.add(u)
                q.append(u)
    return result
```

## Print Adjacency List

```python
def print_adjacency_list(graph):
    for v in sorted(graph["adj"]):
        neighbors = graph["adj"][v]
        print(f"{v}: {neighbors}")
```
