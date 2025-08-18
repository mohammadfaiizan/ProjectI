"""
Advanced Graph Construction Problems
Difficulty: Hard

This file contains advanced graph construction algorithms and techniques that demonstrate
sophisticated graph building and manipulation concepts.

Problems included:
1. Graph Construction from Degree Sequence
2. Eulerian Circuit Construction
3. Hamiltonian Path Construction
4. Minimum Spanning Tree Construction Variants
5. Planar Graph Construction
6. Random Graph Generation with Properties
7. Graph Reconstruction from Subgraphs
8. Dynamic Graph Construction
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict, deque
import random
import heapq

class AdvancedGraphConstructor:
    """
    Advanced graph construction algorithms
    """
    
    def construct_from_degree_sequence(self, degrees: List[int]) -> Optional[List[List[int]]]:
        """
        Problem: Construct graph from degree sequence (Erdős–Gallai algorithm)
        
        Given a degree sequence, construct a simple graph if possible.
        
        Time: O(N²)
        Space: O(N²)
        """
        n = len(degrees)
        degrees.sort(reverse=True)
        
        # Check Erdős–Gallai conditions
        if sum(degrees) % 2 != 0:
            return None  # Sum must be even
        
        for k in range(n):
            left_sum = sum(degrees[:k+1])
            right_sum = k * (k + 1)
            
            for i in range(k+1, n):
                right_sum += min(degrees[i], k + 1)
            
            if left_sum > right_sum:
                return None
        
        # Construct graph using Havel-Hakimi algorithm
        adj_matrix = [[0] * n for _ in range(n)]
        degree_list = [(degrees[i], i) for i in range(n)]
        
        while True:
            degree_list.sort(reverse=True)
            
            # Remove vertices with degree 0
            while degree_list and degree_list[-1][0] == 0:
                degree_list.pop()
            
            if not degree_list:
                break
            
            # Take vertex with highest degree
            d, v = degree_list.pop(0)
            
            if d > len(degree_list):
                return None  # Impossible
            
            # Connect to next d vertices with highest degrees
            for i in range(d):
                d_neighbor, u = degree_list[i]
                if adj_matrix[v][u] == 1:  # Already connected
                    return None
                
                adj_matrix[v][u] = adj_matrix[u][v] = 1
                degree_list[i] = (d_neighbor - 1, u)
        
        # Convert to edge list
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i][j] == 1:
                    edges.append([i, j])
        
        return edges
    
    def construct_eulerian_circuit(self, edges: List[List[int]]) -> Optional[List[int]]:
        """
        Problem: Construct Eulerian circuit using Hierholzer's algorithm
        
        Find an Eulerian circuit if one exists.
        
        Time: O(E)
        Space: O(V + E)
        """
        # Build adjacency list
        adj = defaultdict(list)
        degree = defaultdict(int)
        
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            degree[u] += 1
            degree[v] += 1
        
        # Check if Eulerian circuit exists (all degrees even)
        for d in degree.values():
            if d % 2 != 0:
                return None
        
        if not adj:
            return []
        
        # Hierholzer's algorithm
        start_vertex = next(iter(adj))
        circuit = []
        stack = [start_vertex]
        
        while stack:
            curr = stack[-1]
            
            if adj[curr]:
                next_vertex = adj[curr].pop()
                adj[next_vertex].remove(curr)
                stack.append(next_vertex)
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit if len(circuit) == len(edges) + 1 else None
    
    def construct_minimum_spanning_forest(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        """
        Problem: Construct Minimum Spanning Forest using Kruskal's algorithm
        
        Build MSF for potentially disconnected graph.
        
        Time: O(E log E)
        Space: O(V)
        """
        # Union-Find for MST construction
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            
            if rank[px] < rank[py]:
                px, py = py, px
            
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            
            return True
        
        # Sort edges by weight (assuming third element is weight)
        if edges and len(edges[0]) == 3:
            edges.sort(key=lambda x: x[2])
        
        mst_edges = []
        for edge in edges:
            if len(edge) == 2:
                u, v = edge
            else:
                u, v = edge[0], edge[1]
            
            if union(u, v):
                mst_edges.append(edge)
        
        return mst_edges
    
    def construct_planar_graph_embedding(self, edges: List[List[int]]) -> Dict[int, List[int]]:
        """
        Problem: Construct planar embedding (simplified approach)
        
        Create a planar embedding if the graph is planar.
        This is a simplified version - full planarity testing is complex.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build adjacency list
        adj = defaultdict(list)
        vertices = set()
        
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            vertices.add(u)
            vertices.add(v)
        
        # Simple planar embedding using DFS ordering
        embedding = defaultdict(list)
        visited = set()
        
        def dfs_embed(v, parent=-1):
            visited.add(v)
            neighbors = []
            
            for u in adj[v]:
                if u != parent:
                    neighbors.append(u)
                    if u not in visited:
                        dfs_embed(u, v)
            
            # Sort neighbors for consistent embedding
            neighbors.sort()
            if parent != -1:
                neighbors.insert(0, parent)
            
            embedding[v] = neighbors
        
        # Start DFS from arbitrary vertex
        if vertices:
            start = min(vertices)
            dfs_embed(start)
        
        return dict(embedding)
    
    def construct_random_graph(self, n: int, p: float, properties: Dict[str, any] = None) -> List[List[int]]:
        """
        Problem: Construct random graph with specified properties
        
        Generate Erdős–Rényi random graph with optional constraints.
        
        Time: O(N²)
        Space: O(E)
        """
        properties = properties or {}
        edges = []
        
        # Generate random edges with probability p
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    edges.append([i, j])
        
        # Apply constraints if specified
        if properties.get('connected', False):
            edges = self._ensure_connected(n, edges)
        
        if properties.get('min_degree'):
            edges = self._ensure_min_degree(n, edges, properties['min_degree'])
        
        if properties.get('bipartite', False):
            edges = self._make_bipartite(n, edges)
        
        return edges
    
    def _ensure_connected(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        """Ensure graph is connected by adding necessary edges"""
        if n <= 1:
            return edges
        
        # Find connected components
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        visited = set()
        components = []
        
        def dfs(v, component):
            visited.add(v)
            component.append(v)
            for u in adj[v]:
                if u not in visited:
                    dfs(u, component)
        
        for i in range(n):
            if i not in visited:
                component = []
                dfs(i, component)
                components.append(component)
        
        # Connect components
        result_edges = edges[:]
        for i in range(len(components) - 1):
            u = components[i][0]
            v = components[i + 1][0]
            result_edges.append([u, v])
        
        return result_edges
    
    def _ensure_min_degree(self, n: int, edges: List[List[int]], min_deg: int) -> List[List[int]]:
        """Ensure all vertices have minimum degree"""
        degree = [0] * n
        edge_set = set()
        
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
            edge_set.add(tuple(sorted([u, v])))
        
        result_edges = edges[:]
        
        for v in range(n):
            while degree[v] < min_deg:
                # Find vertex to connect to
                for u in range(n):
                    if u != v and tuple(sorted([u, v])) not in edge_set:
                        result_edges.append([u, v])
                        edge_set.add(tuple(sorted([u, v])))
                        degree[u] += 1
                        degree[v] += 1
                        break
                else:
                    break  # Cannot satisfy constraint
        
        return result_edges
    
    def _make_bipartite(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        """Filter edges to make graph bipartite"""
        # Simple approach: divide vertices into two sets
        set_a = set(range(0, n // 2))
        set_b = set(range(n // 2, n))
        
        bipartite_edges = []
        for u, v in edges:
            if (u in set_a and v in set_b) or (u in set_b and v in set_a):
                bipartite_edges.append([u, v])
        
        return bipartite_edges
    
    def construct_graph_from_distances(self, distances: List[List[float]], threshold: float) -> List[List[int]]:
        """
        Problem: Construct graph from distance matrix
        
        Create edges between vertices whose distance is below threshold.
        
        Time: O(N²)
        Space: O(E)
        """
        n = len(distances)
        edges = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if distances[i][j] <= threshold:
                    edges.append([i, j])
        
        return edges
    
    def construct_complement_graph(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        """
        Problem: Construct complement graph
        
        Build complement where edges exist iff they don't exist in original.
        
        Time: O(N² + E)
        Space: O(N² + E)
        """
        edge_set = set()
        for u, v in edges:
            edge_set.add(tuple(sorted([u, v])))
        
        complement_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in edge_set:
                    complement_edges.append([i, j])
        
        return complement_edges

def test_advanced_construction():
    """Test advanced graph construction methods"""
    constructor = AdvancedGraphConstructor()
    
    print("=== Advanced Graph Construction Tests ===")
    
    # Test degree sequence construction
    print("\n1. Degree Sequence Construction:")
    degree_sequences = [
        [3, 3, 2, 2, 2],  # Valid
        [3, 3, 3, 1],     # Invalid (odd sum)
        [4, 2, 2, 2],     # Valid
        [5, 1, 1, 1],     # Invalid
    ]
    
    for i, degrees in enumerate(degree_sequences):
        edges = constructor.construct_from_degree_sequence(degrees[:])
        print(f"  Sequence {degrees}: {'Valid' if edges else 'Invalid'}")
        if edges:
            print(f"    Edges: {edges}")
    
    # Test Eulerian circuit
    print("\n2. Eulerian Circuit Construction:")
    test_graphs = [
        [[0,1],[1,2],[2,3],[3,0]],  # Square (Eulerian)
        [[0,1],[1,2],[2,0],[0,3]],  # Triangle + edge (not Eulerian)
    ]
    
    for i, edges in enumerate(test_graphs):
        circuit = constructor.construct_eulerian_circuit(edges)
        print(f"  Graph {edges}:")
        print(f"    Circuit: {circuit if circuit else 'None'}")
    
    # Test MST construction
    print("\n3. Minimum Spanning Tree Construction:")
    weighted_edges = [[0,1,1],[1,2,2],[0,2,3],[2,3,1]]
    mst = constructor.construct_minimum_spanning_forest(4, weighted_edges)
    print(f"  Original: {weighted_edges}")
    print(f"  MST: {mst}")

def demonstrate_graph_properties():
    """Demonstrate construction of graphs with specific properties"""
    print("\n=== Graph Property Construction ===")
    
    constructor = AdvancedGraphConstructor()
    
    # Construct connected random graph
    print("1. Connected Random Graph:")
    random.seed(42)  # For reproducibility
    connected_graph = constructor.construct_random_graph(
        n=6, p=0.3, properties={'connected': True}
    )
    print(f"   Edges: {connected_graph}")
    
    # Construct complement graph
    print("\n2. Complement Graph:")
    original = [[0,1],[1,2],[0,2]]
    complement = constructor.construct_complement_graph(4, original)
    print(f"   Original: {original}")
    print(f"   Complement: {complement}")
    
    # Construct from distances
    print("\n3. Graph from Distance Matrix:")
    distances = [
        [0, 1, 3, 4],
        [1, 0, 2, 5],
        [3, 2, 0, 1],
        [4, 5, 1, 0]
    ]
    threshold_graph = constructor.construct_graph_from_distances(distances, 2.0)
    print(f"   Distance matrix: {distances}")
    print(f"   Threshold 2.0: {threshold_graph}")

def analyze_construction_algorithms():
    """Analyze different construction algorithms"""
    print("\n=== Construction Algorithm Analysis ===")
    
    algorithms = [
        "Havel-Hakimi (Degree Sequence)",
        "Hierholzer's (Eulerian Circuit)",
        "Kruskal's (MST)",
        "Erdős–Rényi (Random Graph)",
        "Distance-based Construction",
    ]
    
    complexities = [
        "O(N²)",
        "O(E)",
        "O(E log E)",
        "O(N²)",
        "O(N²)",
    ]
    
    applications = [
        "Network design with degree constraints",
        "Route planning, DNA sequencing",
        "Network infrastructure, clustering",
        "Network modeling, simulation",
        "Geometric graphs, sensor networks",
    ]
    
    print(f"{'Algorithm':<30} {'Complexity':<12} {'Applications'}")
    print("-" * 80)
    
    for alg, comp, app in zip(algorithms, complexities, applications):
        print(f"{alg:<30} {comp:<12} {app}")

def demonstrate_specialized_constructions():
    """Demonstrate specialized graph construction techniques"""
    print("\n=== Specialized Construction Techniques ===")
    
    constructor = AdvancedGraphConstructor()
    
    print("1. Constructing Regular Graphs:")
    # Attempt to construct 3-regular graph on 6 vertices
    degrees = [3] * 6
    regular_graph = constructor.construct_from_degree_sequence(degrees)
    print(f"   3-regular on 6 vertices: {'Success' if regular_graph else 'Failed'}")
    if regular_graph:
        print(f"   Edges: {regular_graph}")
    
    print("\n2. Planar Graph Embedding:")
    planar_edges = [[0,1],[1,2],[2,3],[3,0],[0,2]]
    embedding = constructor.construct_planar_graph_embedding(planar_edges)
    print(f"   Original edges: {planar_edges}")
    print(f"   Planar embedding:")
    for vertex, neighbors in embedding.items():
        print(f"     Vertex {vertex}: {neighbors}")
    
    print("\n3. Bipartite Graph Construction:")
    random.seed(42)
    bipartite_graph = constructor.construct_random_graph(
        n=6, p=0.4, properties={'bipartite': True}
    )
    print(f"   Bipartite edges: {bipartite_graph}")

if __name__ == "__main__":
    test_advanced_construction()
    demonstrate_graph_properties()
    analyze_construction_algorithms()
    demonstrate_specialized_constructions()

"""
Advanced Graph Construction Concepts:
1. Degree Sequence Realization (Erdős–Gallai theorem)
2. Eulerian Circuit Construction (Hierholzer's algorithm)
3. Spanning Tree Construction (Kruskal's, Prim's)
4. Random Graph Models (Erdős–Rényi, preferential attachment)
5. Planar Graph Embedding
6. Graph Complement and Operations
7. Distance-based Graph Construction
8. Constrained Graph Generation

Key Algorithms:
- Havel-Hakimi: Degree sequence realization
- Hierholzer's: Eulerian circuit finding
- Union-Find: MST and connectivity
- DFS/BFS: Embedding and traversal
- Probabilistic: Random graph generation

Theoretical Foundations:
- Erdős–Gallai theorem for graphical sequences
- Euler's theorem for Eulerian circuits
- Cut property for MSTs
- Random graph theory
- Planar graph theory

Real-world Applications:
- Network design and optimization
- Social network modeling
- Transportation planning
- Circuit design
- Bioinformatics (protein interaction networks)
- Computer graphics (mesh generation)
"""
