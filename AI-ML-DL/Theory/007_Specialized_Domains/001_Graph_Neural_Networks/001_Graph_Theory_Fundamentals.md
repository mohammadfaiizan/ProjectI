# Graph Theory Fundamentals

## Table of Contents

1. [Introduction to Graph Structures](#introduction-to-graph-structures)
2. [Graph Representations](#graph-representations)
3. [Adjacency and Incidence Matrices](#adjacency-and-incidence-matrices)
4. [Graph Properties and Metrics](#graph-properties-and-metrics)
5. [Degree Distribution and Centrality](#degree-distribution-and-centrality)
6. [Connectivity and Path Analysis](#connectivity-and-path-analysis)
7. [Graph Laplacian](#graph-laplacian)
8. [Spectral Graph Theory](#spectral-graph-theory)
9. [Graph Isomorphism and Equivalence](#graph-isomorphism-and-equivalence)
10. [Key Takeaways](#key-takeaways)

---

## Introduction to Graph Structures

Graph theory provides the mathematical foundation for understanding relational data structures. A graph $G = (V, E)$ consists of a set of vertices (nodes) $V$ and a set of edges $E$ connecting pairs of vertices. Graphs serve as natural representations for numerous real-world phenomena, including social networks, molecular structures, knowledge graphs, and citation networks.

### Basic Definitions

**Definition 1 (Graph)**: A graph $G$ is an ordered pair $(V, E)$ where:
- $V = \{v_1, v_2, \ldots, v_n\}$ is a finite set of vertices (nodes)
- $E \subseteq V \times V$ is a set of edges (links)

For an undirected graph, edges are unordered pairs: $(u, v) = (v, u)$. For a directed graph (digraph), edges are ordered pairs: $(u, v) \neq (v, u)$.

**Definition 2 (Weighted Graph)**: A weighted graph assigns a weight $w_{uv}$ to each edge $(u, v) \in E$. The weight function $w: E \rightarrow \mathbb{R}$ captures edge strength, distance, or similarity.

**Definition 3 (Multigraph)**: A multigraph allows multiple edges between the same pair of vertices, enabling representation of multiple relationships.

### Graph Types

| Graph Type | Description | Edge Set |
|------------|-------------|----------|
| Undirected | Symmetric relationships | $E \subseteq \{\{u,v\} : u,v \in V\}$ |
| Directed | Asymmetric relationships | $E \subseteq \{(u,v) : u,v \in V\}$ |
| Weighted | Edges with numerical values | $w: E \rightarrow \mathbb{R}$ |
| Bipartite | Two disjoint vertex sets | $V = V_1 \cup V_2$, $E \subseteq V_1 \times V_2$ |
| Hypergraph | Edges connect multiple vertices | $E \subseteq 2^V$ |

---

## Graph Representations

Efficient representation of graphs is crucial for computational algorithms. Different representations offer trade-offs between memory efficiency and query performance.

### Adjacency List

The adjacency list representation stores, for each vertex, a list of its neighbors:

```python
# Adjacency list representation
graph = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3],
    3: [1, 2]
}
```

**Space Complexity**: $O(|V| + |E|)$ for sparse graphs, optimal for memory efficiency.

**Time Complexity**:
- Check edge existence: $O(\deg(v))$
- Iterate neighbors: $O(\deg(v))$
- Add edge: $O(1)$

### Edge List

The edge list representation stores all edges as pairs:

```python
# Edge list representation
edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
```

**Space Complexity**: $O(|E|)$

**Time Complexity**:
- Check edge existence: $O(|E|)$
- Iterate neighbors: $O(|E|)$

### Adjacency Matrix

The adjacency matrix $A \in \{0,1\}^{|V| \times |V|}$ encodes edge existence:

$$A_{ij} = \begin{cases}
1 & \text{if } (i,j) \in E \\
0 & \text{otherwise}
\end{cases}$$

For undirected graphs, $A$ is symmetric: $A = A^T$. For weighted graphs, $A_{ij} = w_{ij}$ if edge exists, $0$ otherwise.

**Space Complexity**: $O(|V|^2)$

**Time Complexity**:
- Check edge existence: $O(1)$
- Iterate neighbors: $O(|V|)$
- Matrix operations: Efficient for dense graphs

---

## Adjacency and Incidence Matrices

### Adjacency Matrix Properties

The adjacency matrix $A$ of a graph $G$ with $n$ vertices has several important properties:

**Theorem 1**: For an undirected graph, $A$ is symmetric and the $k$-th power $A^k$ counts walks of length $k$ between vertices.

The number of walks of length $k$ from vertex $i$ to vertex $j$ is given by $(A^k)_{ij}$.

**Theorem 2**: The trace of $A^2$ equals twice the number of edges in an undirected graph:

$$\text{tr}(A^2) = 2|E|$$

**Theorem 3**: The eigenvalues of $A$ provide information about graph structure. For a $d$-regular graph, the largest eigenvalue is $\lambda_1 = d$.

### Incidence Matrix

The incidence matrix $B \in \{0,1\}^{|V| \times |E|}$ relates vertices to edges:

$$B_{ve} = \begin{cases}
1 & \text{if vertex } v \text{ is incident to edge } e \\
0 & \text{otherwise}
\end{cases}$$

For directed graphs, we distinguish:
- $B_{ve} = 1$ if $v$ is the tail of edge $e$
- $B_{ve} = -1$ if $v$ is the head of edge $e$
- $B_{ve} = 0$ otherwise

**Relationship**: For an undirected graph, $A = BB^T - D$, where $D$ is the degree matrix.

---

## Graph Properties and Metrics

### Basic Graph Metrics

**Order**: The number of vertices $n = |V|$

**Size**: The number of edges $m = |E|$

**Density**: The ratio of actual edges to possible edges:

$$\rho = \frac{2|E|}{|V|(|V|-1)}$$

for undirected graphs, and $\rho = \frac{|E|}{|V|(|V|-1)}$ for directed graphs.

**Average Degree**: For an undirected graph:

$$\bar{d} = \frac{2|E|}{|V|} = \frac{1}{|V|} \sum_{v \in V} \deg(v)$$

### Degree Distribution

The degree distribution $P(k)$ gives the probability that a randomly chosen vertex has degree $k$:

$$P(k) = \frac{|\{v \in V : \deg(v) = k\}|}{|V|}$$

Many real-world networks exhibit power-law degree distributions: $P(k) \propto k^{-\gamma}$ where $\gamma > 0$ is the scaling exponent.

### Clustering Coefficient

The local clustering coefficient $C_i$ measures the fraction of triangles among neighbors of vertex $i$:

$$C_i = \frac{2e_i}{k_i(k_i - 1)}$$

where $e_i$ is the number of edges between neighbors of $i$, and $k_i$ is the degree of $i$.

The global clustering coefficient is:

$$C = \frac{1}{|V|} \sum_{i \in V} C_i$$

---

## Degree Distribution and Centrality

### Degree Centrality

Degree centrality measures the importance of a vertex based on its number of connections:

$$C_D(v) = \deg(v)$$

Normalized degree centrality:

$$C_D'(v) = \frac{\deg(v)}{|V| - 1}$$

### Betweenness Centrality

Betweenness centrality measures the fraction of shortest paths passing through a vertex:

$$C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

where $\sigma_{st}$ is the total number of shortest paths from $s$ to $t$, and $\sigma_{st}(v)$ is the number of those paths passing through $v$.

### Closeness Centrality

Closeness centrality measures the inverse of the average shortest path distance:

$$C_C(v) = \frac{1}{\sum_{u \neq v} d(u,v)}$$

where $d(u,v)$ is the shortest path distance between $u$ and $v$.

### Eigenvector Centrality

Eigenvector centrality assigns importance proportional to the sum of neighbors' centralities:

$$C_E(v) = \frac{1}{\lambda} \sum_{u \in N(v)} C_E(u)$$

This corresponds to the principal eigenvector of the adjacency matrix: $AC_E = \lambda C_E$.

### PageRank

PageRank extends eigenvector centrality with a damping factor:

$$PR(v) = \frac{1-d}{|V|} + d \sum_{u \in N_{in}(v)} \frac{PR(u)}{\deg_{out}(u)}$$

where $d \in [0,1]$ is the damping factor (typically $0.85$), and $N_{in}(v)$ are in-neighbors of $v$.

---

## Connectivity and Path Analysis

### Paths and Cycles

**Definition 4 (Path)**: A path $P = (v_0, v_1, \ldots, v_k)$ is a sequence of vertices where $(v_i, v_{i+1}) \in E$ for all $i$.

**Definition 5 (Cycle)**: A cycle is a path where $v_0 = v_k$ and all other vertices are distinct.

**Definition 6 (Simple Path)**: A simple path contains no repeated vertices.

### Connectivity

**Definition 7 (Connected Graph)**: An undirected graph is connected if there exists a path between every pair of vertices.

**Definition 8 (Strongly Connected)**: A directed graph is strongly connected if there exists a directed path from any vertex to any other vertex.

**Definition 9 (Weakly Connected)**: A directed graph is weakly connected if the underlying undirected graph is connected.

### Shortest Paths

The shortest path problem finds the minimum-weight path between two vertices. For unweighted graphs, this reduces to finding the path with minimum number of edges.

**Dijkstra's Algorithm**: Solves single-source shortest paths for non-negative weights in $O(|E| + |V| \log |V|)$ using a priority queue.

**Floyd-Warshall Algorithm**: Computes all-pairs shortest paths in $O(|V|^3)$ using dynamic programming:

$$d_{ij}^{(k)} = \min(d_{ij}^{(k-1)}, d_{ik}^{(k-1)} + d_{kj}^{(k-1)})$$

### Graph Diameter

The diameter $\text{diam}(G)$ is the maximum shortest path distance between any pair of vertices:

$$\text{diam}(G) = \max_{u,v \in V} d(u,v)$$

The average path length is:

$$\bar{\ell} = \frac{1}{|V|(|V|-1)} \sum_{u \neq v} d(u,v)$$

---

## Graph Laplacian

The graph Laplacian is a fundamental operator in spectral graph theory, connecting combinatorial graph properties to spectral analysis.

### Unnormalized Laplacian

The unnormalized graph Laplacian is defined as:

$$L = D - A$$

where $D$ is the degree matrix (diagonal matrix with $D_{ii} = \deg(i)$) and $A$ is the adjacency matrix.

**Properties**:
1. $L$ is symmetric and positive semidefinite
2. $L$ has $|V|$ non-negative eigenvalues: $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$
3. The number of zero eigenvalues equals the number of connected components
4. For any vector $f \in \mathbb{R}^{|V|}$:

$$f^T L f = \frac{1}{2} \sum_{(i,j) \in E} (f_i - f_j)^2$$

### Normalized Laplacian

The symmetric normalized Laplacian is:

$$L_{sym} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$$

The random walk normalized Laplacian is:

$$L_{rw} = D^{-1} L = I - D^{-1} A$$

**Properties**:
- Eigenvalues of $L_{sym}$ lie in $[0, 2]$
- $\lambda_2(L_{sym})$ (algebraic connectivity) measures graph connectivity
- Small $\lambda_2$ indicates the graph can be easily disconnected

### Laplacian Eigenvectors

The eigenvectors of $L$ provide a natural embedding of vertices:

- The first eigenvector (constant) corresponds to $\lambda_1 = 0$
- The second eigenvector (Fiedler vector) provides a natural graph cut
- Higher eigenvectors capture finer-grained structure

---

## Spectral Graph Theory

Spectral graph theory studies graphs through the eigenvalues and eigenvectors of matrices associated with graphs.

### Eigenvalue Bounds

**Theorem 4 (Rayleigh Quotient)**: For the Laplacian $L$:

$$\lambda_2 = \min_{f \perp \mathbf{1}} \frac{f^T L f}{f^T f}$$

**Theorem 5 (Cheeger's Inequality)**: For a graph $G$:

$$\frac{h_G^2}{2} \leq \lambda_2 \leq 2h_G$$

where $h_G$ is the Cheeger constant (isoperimetric number):

$$h_G = \min_{S \subset V} \frac{|\partial S|}{\min(|S|, |V \setminus S|)}$$

### Graph Partitioning

Spectral clustering uses eigenvectors of the Laplacian to partition graphs:

1. Compute $k$ smallest eigenvectors of $L$
2. Embed vertices using these eigenvectors
3. Apply $k$-means clustering in the embedding space

### Heat Kernel and Diffusion

The graph heat kernel $H_t = e^{-tL}$ describes diffusion on the graph:

$$H_t = \sum_{k=0}^{\infty} \frac{(-tL)^k}{k!}$$

The heat kernel provides a smooth interpolation between local and global graph structure.

### Graph Signal Processing

A graph signal is a function $f: V \rightarrow \mathbb{R}$ assigning values to vertices. The graph Fourier transform uses Laplacian eigenvectors as basis:

$$\hat{f}(\lambda_i) = \langle f, u_i \rangle = \sum_{v \in V} f(v) u_i(v)$$

where $u_i$ is the eigenvector corresponding to eigenvalue $\lambda_i$.

---

## Graph Isomorphism and Equivalence

### Graph Isomorphism

**Definition 10 (Graph Isomorphism)**: Two graphs $G_1 = (V_1, E_1)$ and $G_2 = (V_2, E_2)$ are isomorphic if there exists a bijection $\phi: V_1 \rightarrow V_2$ such that $(u,v) \in E_1$ if and only if $(\phi(u), \phi(v)) \in E_2$.

Graph isomorphism is a fundamental problem in graph theory. While no polynomial-time algorithm is known, practical algorithms exist for many cases.

### Graph Invariants

Graph invariants are properties preserved under isomorphism:
- Number of vertices and edges
- Degree sequence
- Spectrum (eigenvalues)
- Number of triangles, cycles
- Chromatic number

### Weisfeiler-Lehman Test

The Weisfeiler-Lehman (WL) test provides a polynomial-time heuristic for graph isomorphism:

1. Initialize labels: $h_v^{(0)} = \deg(v)$
2. Iteratively refine labels:

$$h_v^{(k+1)} = \text{hash}(h_v^{(k)}, \{h_u^{(k)} : u \in N(v)\})$$

3. Graphs are non-isomorphic if label multisets differ at any iteration

The WL test is related to the expressiveness of message-passing GNNs.

---

## Key Takeaways

1. **Graph Representations**: Adjacency matrices provide efficient edge queries but require $O(|V|^2)$ space. Adjacency lists are optimal for sparse graphs with $O(|V| + |E|)$ space.

2. **Graph Laplacian**: The Laplacian $L = D - A$ encodes graph structure. Its eigenvalues reveal connectivity (number of zero eigenvalues equals connected components) and provide natural embeddings.

3. **Spectral Properties**: The second smallest eigenvalue $\lambda_2$ (algebraic connectivity) measures how easily a graph can be disconnected, with connections to Cheeger's inequality.

4. **Centrality Measures**: Different centrality metrics capture different notions of importance: degree (local), betweenness (bridging), closeness (reachability), eigenvector (influence).

5. **Connectivity**: Graph connectivity determines information flow. Strong connectivity in directed graphs enables reachability between all vertex pairs.

6. **Spectral Graph Theory**: Eigenvectors of the Laplacian provide natural coordinates for graph embedding, enabling spectral clustering and graph signal processing.

7. **Graph Invariants**: Properties like degree sequence and spectrum are preserved under isomorphism, enabling graph comparison and classification.

8. **Weisfeiler-Lehman Test**: The WL test provides a polynomial-time heuristic for graph isomorphism and establishes theoretical limits for message-passing GNN expressiveness.

9. **Degree Distribution**: Real-world networks often exhibit power-law degree distributions, indicating scale-free structure with hub vertices.

10. **Graph Metrics**: Clustering coefficient, diameter, and average path length characterize small-world properties and network topology.
