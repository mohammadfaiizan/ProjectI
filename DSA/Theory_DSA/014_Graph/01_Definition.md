# Graph - Definition and Concepts

## Graph Terminology

### Vertex (Node)
A fundamental unit of a graph. Each vertex represents an entity. Denoted as V or n.

### Edge
A connection between two vertices. Denoted as E or m. An edge (u, v) connects vertex u to vertex v.

### Path
A sequence of vertices v1, v2, ..., vk such that (vi, vi+1) is an edge for each i. The length of a path is the number of edges.

### Cycle
A path that starts and ends at the same vertex. In a simple graph, no vertex (except start/end) repeats.

### Degree
- **Degree (undirected)**: Number of edges incident to a vertex. Sum of all degrees = 2 * E.
- **In-degree (directed)**: Number of edges coming into a vertex.
- **Out-degree (directed)**: Number of edges going out of a vertex. Sum of in-degrees = sum of out-degrees = E.

### Connected
- **Connected (undirected)**: There exists a path between every pair of vertices.
- **Strongly connected (directed)**: For every pair (u, v), there exists a path from u to v and from v to u.

## Graph Types

| Type | Description |
|------|-------------|
| Directed | Edges have direction (u, v) != (v, u) |
| Undirected | Edges are bidirectional |
| Weighted | Each edge has a weight/cost |
| Unweighted | All edges have equal weight (typically 1) |
| DAG | Directed Acyclic Graph; no cycles |
| Bipartite | Vertices can be partitioned into two sets with no edges within same set |
| Complete | Every pair of vertices is connected by an edge. Kn has n(n-1)/2 edges |
| Sparse | E << V^2; few edges relative to vertices |
| Dense | E close to V^2; many edges |

## Graph Representations

### Adjacency List
Each vertex maps to a list of its neighbors. Space O(V + E). Best for sparse graphs.

```python
graph = {
    0: [1, 2],
    1: [0, 2, 3],
    2: [0, 1],
    3: [1]
}
```

For weighted: store (neighbor, weight) pairs.

### Adjacency Matrix
2D array where matrix[i][j] = 1 if edge exists (or weight), else 0 or infinity. Space O(V^2). Best for dense graphs and edge existence checks.

```python
n = 4
matrix = [[0] * n for _ in range(n)]
matrix[0][1] = 1
matrix[0][2] = 1
matrix[1][0] = 1
matrix[1][2] = 1
matrix[1][3] = 1
matrix[2][0] = 1
matrix[2][1] = 1
matrix[3][1] = 1
```

### Edge List
List of (u, v) or (u, v, w) tuples. Space O(E). Used for Kruskal's MST, Bellman-Ford.

```python
edges = [(0, 1), (0, 2), (1, 2), (1, 3)]
weighted_edges = [(0, 1, 5), (0, 2, 3), (1, 2, 2), (1, 3, 7)]
```

## When to Use Each Representation

| Representation | Use When |
|----------------|----------|
| Adjacency List | Sparse graphs, DFS/BFS, most algorithms |
| Adjacency Matrix | Dense graphs, frequent edge existence checks, Floyd-Warshall |
| Edge List | Kruskal's MST, Bellman-Ford, when iterating all edges |

## Time Complexity Table

| Operation | Adjacency List | Adjacency Matrix |
|-----------|----------------|------------------|
| Add vertex | O(1) | O(V^2) to resize |
| Add edge | O(1) | O(1) |
| Remove vertex | O(V + E) | O(V^2) |
| Remove edge | O(degree) | O(1) |
| Check edge exists | O(degree) | O(1) |
| Get neighbors | O(degree) | O(V) |
| Space | O(V + E) | O(V^2) |
| DFS/BFS | O(V + E) | O(V^2) |
