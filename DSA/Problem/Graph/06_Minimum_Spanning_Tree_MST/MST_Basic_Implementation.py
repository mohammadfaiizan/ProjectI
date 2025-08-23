"""
MST Basic Implementation Collection
Difficulty: Easy

This file contains comprehensive implementations of fundamental Minimum Spanning Tree (MST) 
algorithms with detailed explanations and educational demonstrations.

Algorithms Covered:
1. Prim's Algorithm (Multiple Variants)
2. Kruskal's Algorithm with Union-Find
3. Borůvka's Algorithm
4. MST Verification and Properties
5. Weighted Graph Representations
6. MST Applications and Extensions
"""

from typing import List, Tuple, Dict, Set, Optional
import heapq
from collections import defaultdict

class UnionFind:
    """Union-Find (Disjoint Set Union) data structure for cycle detection"""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
    def find(self, x: int) -> int:
        """Find root of set containing x with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Union two sets, return True if they were different sets"""
        root_x, root_y = self.find(x), self.find(y)
        
        if root_x == root_y:
            return False
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        if self.rank[root_x] == root_y:
            self.rank[root_x] += 1
        
        self.components -= 1
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """Check if two elements are in the same set"""
        return self.find(x) == self.find(y)
    
    def is_connected_graph(self) -> bool:
        """Check if graph is connected (single component)"""
        return self.components == 1

class WeightedGraph:
    """Weighted graph representation for MST algorithms"""
    
    def __init__(self, vertices: int):
        self.V = vertices
        self.edges = []
        self.adj_list = defaultdict(list)
        self.adj_matrix = [[float('inf')] * vertices for _ in range(vertices)]
        
        # Initialize diagonal to 0
        for i in range(vertices):
            self.adj_matrix[i][i] = 0
    
    def add_edge(self, u: int, v: int, weight: float):
        """Add weighted edge to graph"""
        self.edges.append((weight, u, v))
        self.adj_list[u].append((v, weight))
        self.adj_list[v].append((u, weight))
        self.adj_matrix[u][v] = weight
        self.adj_matrix[v][u] = weight
    
    def get_edges(self) -> List[Tuple[float, int, int]]:
        """Get all edges as (weight, u, v) tuples"""
        return self.edges[:]
    
    def get_adjacency_list(self) -> Dict[int, List[Tuple[int, float]]]:
        """Get adjacency list representation"""
        return dict(self.adj_list)
    
    def get_adjacency_matrix(self) -> List[List[float]]:
        """Get adjacency matrix representation"""
        return [row[:] for row in self.adj_matrix]

class MSTAlgorithms:
    """Collection of MST algorithm implementations"""
    
    def prims_algorithm_heap(self, graph: WeightedGraph) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Prim's Algorithm using Min-Heap (Priority Queue)
        
        Time: O(E log V)
        Space: O(V + E)
        """
        if graph.V == 0:
            return 0, []
        
        adj_list = graph.get_adjacency_list()
        visited = [False] * graph.V
        min_heap = [(0, 0, -1)]  # (weight, vertex, parent)
        mst_edges = []
        total_weight = 0
        
        while min_heap:
            weight, u, parent = heapq.heappop(min_heap)
            
            if visited[u]:
                continue
            
            # Add vertex to MST
            visited[u] = True
            total_weight += weight
            
            if parent != -1:
                mst_edges.append((parent, u))
            
            # Add edges to unvisited neighbors
            for v, edge_weight in adj_list.get(u, []):
                if not visited[v]:
                    heapq.heappush(min_heap, (edge_weight, v, u))
        
        return total_weight, mst_edges
    
    def prims_algorithm_matrix(self, graph: WeightedGraph) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Prim's Algorithm using Adjacency Matrix
        
        Time: O(V^2)
        Space: O(V^2)
        """
        if graph.V == 0:
            return 0, []
        
        adj_matrix = graph.get_adjacency_matrix()
        visited = [False] * graph.V
        min_edge = [float('inf')] * graph.V
        parent = [-1] * graph.V
        min_edge[0] = 0
        
        mst_edges = []
        total_weight = 0
        
        for _ in range(graph.V):
            # Find minimum weight unvisited vertex
            u = -1
            for v in range(graph.V):
                if not visited[v] and (u == -1 or min_edge[v] < min_edge[u]):
                    u = v
            
            # Add to MST
            visited[u] = True
            total_weight += min_edge[u]
            
            if parent[u] != -1:
                mst_edges.append((parent[u], u))
            
            # Update minimum edges to unvisited vertices
            for v in range(graph.V):
                if not visited[v] and adj_matrix[u][v] < min_edge[v]:
                    min_edge[v] = adj_matrix[u][v]
                    parent[v] = u
        
        return total_weight, mst_edges
    
    def kruskals_algorithm(self, graph: WeightedGraph) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Kruskal's Algorithm with Union-Find
        
        Time: O(E log E)
        Space: O(V)
        """
        edges = graph.get_edges()
        edges.sort()  # Sort by weight
        
        uf = UnionFind(graph.V)
        mst_edges = []
        total_weight = 0
        
        for weight, u, v in edges:
            if uf.union(u, v):
                mst_edges.append((u, v))
                total_weight += weight
                
                # Early termination
                if len(mst_edges) == graph.V - 1:
                    break
        
        return total_weight, mst_edges
    
    def boruvkas_algorithm(self, graph: WeightedGraph) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Borůvka's Algorithm (Parallel MST)
        
        Time: O(E log V)
        Space: O(V + E)
        """
        uf = UnionFind(graph.V)
        mst_edges = []
        total_weight = 0
        
        while uf.components > 1:
            # Find cheapest edge for each component
            cheapest = [-1] * graph.V
            cheapest_weight = [float('inf')] * graph.V
            
            for weight, u, v in graph.get_edges():
                root_u, root_v = uf.find(u), uf.find(v)
                
                if root_u != root_v:
                    # Update cheapest edge for component u
                    if weight < cheapest_weight[root_u]:
                        cheapest_weight[root_u] = weight
                        cheapest[root_u] = (u, v)
                    
                    # Update cheapest edge for component v
                    if weight < cheapest_weight[root_v]:
                        cheapest_weight[root_v] = weight
                        cheapest[root_v] = (u, v)
            
            # Add cheapest edges to MST
            added_edges = set()
            for i in range(graph.V):
                if cheapest[i] != -1:
                    u, v = cheapest[i]
                    edge = tuple(sorted([u, v]))
                    
                    if edge not in added_edges and uf.union(u, v):
                        mst_edges.append((u, v))
                        total_weight += cheapest_weight[i]
                        added_edges.add(edge)
        
        return total_weight, mst_edges
    
    def reverse_delete_algorithm(self, graph: WeightedGraph) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Reverse-Delete Algorithm (Educational)
        
        Start with all edges, remove heaviest that don't disconnect graph.
        
        Time: O(E^2 log E)
        Space: O(V + E)
        """
        edges = graph.get_edges()
        edges.sort(reverse=True)  # Sort by weight (descending)
        
        # Start with all edges
        current_edges = set()
        for weight, u, v in graph.get_edges():
            current_edges.add((u, v, weight))
        
        # Try to remove each edge in order of decreasing weight
        for weight, u, v in edges:
            edge = (u, v, weight)
            if edge in current_edges:
                # Remove edge temporarily
                current_edges.remove(edge)
                
                # Check if graph is still connected
                if not self._is_connected_with_edges(graph.V, current_edges):
                    # Removing this edge disconnects graph, keep it
                    current_edges.add(edge)
        
        # Calculate total weight and format result
        total_weight = sum(weight for u, v, weight in current_edges)
        mst_edges = [(u, v) for u, v, weight in current_edges]
        
        return total_weight, mst_edges
    
    def _is_connected_with_edges(self, vertices: int, edges: Set[Tuple[int, int, float]]) -> bool:
        """Helper function to check if graph is connected"""
        if not edges:
            return vertices <= 1
        
        # Build adjacency list from edge set
        adj = defaultdict(list)
        for u, v, weight in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # DFS to check connectivity
        visited = [False] * vertices
        start = next(iter(adj.keys())) if adj else 0
        
        def dfs(node):
            visited[node] = True
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        dfs(start)
        
        # Check if all vertices with edges are visited
        vertices_with_edges = set()
        for u, v, weight in edges:
            vertices_with_edges.add(u)
            vertices_with_edges.add(v)
        
        return all(visited[v] for v in vertices_with_edges)

class MSTVerification:
    """Tools for verifying MST properties and correctness"""
    
    def verify_mst(self, graph: WeightedGraph, mst_edges: List[Tuple[int, int]]) -> Dict[str, bool]:
        """Verify if given edges form a valid MST"""
        results = {}
        
        # Check 1: Correct number of edges
        results['correct_edge_count'] = len(mst_edges) == graph.V - 1
        
        # Check 2: No cycles (tree property)
        results['no_cycles'] = self._has_no_cycles(graph.V, mst_edges)
        
        # Check 3: Connects all vertices (spanning property)
        results['spans_all_vertices'] = self._spans_all_vertices(graph.V, mst_edges)
        
        # Check 4: All edges exist in original graph
        original_edges = set()
        for weight, u, v in graph.get_edges():
            original_edges.add((u, v))
            original_edges.add((v, u))
        
        results['valid_edges'] = all(
            (u, v) in original_edges or (v, u) in original_edges 
            for u, v in mst_edges
        )
        
        return results
    
    def _has_no_cycles(self, vertices: int, edges: List[Tuple[int, int]]) -> bool:
        """Check if edge set has no cycles using Union-Find"""
        uf = UnionFind(vertices)
        
        for u, v in edges:
            if not uf.union(u, v):
                return False  # Found cycle
        
        return True
    
    def _spans_all_vertices(self, vertices: int, edges: List[Tuple[int, int]]) -> bool:
        """Check if edges connect all vertices"""
        if not edges:
            return vertices <= 1
        
        # Build adjacency list
        adj = defaultdict(list)
        vertex_set = set()
        
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            vertex_set.add(u)
            vertex_set.add(v)
        
        # DFS from any vertex
        visited = [False] * vertices
        start = next(iter(vertex_set))
        
        def dfs(node):
            visited[node] = True
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        dfs(start)
        
        # Check if all vertices are reachable
        return all(visited[v] for v in vertex_set)
    
    def compare_mst_algorithms(self, graph: WeightedGraph) -> Dict[str, Tuple[float, List[Tuple[int, int]]]]:
        """Compare results from different MST algorithms"""
        algorithms = MSTAlgorithms()
        
        results = {
            'prims_heap': algorithms.prims_algorithm_heap(graph),
            'prims_matrix': algorithms.prims_algorithm_matrix(graph),
            'kruskals': algorithms.kruskals_algorithm(graph),
            'boruvkas': algorithms.boruvkas_algorithm(graph),
        }
        
        return results

def test_mst_algorithms():
    """Test MST algorithms with various graphs"""
    print("=== Testing MST Algorithms ===")
    
    # Test case 1: Simple graph
    graph1 = WeightedGraph(4)
    graph1.add_edge(0, 1, 10)
    graph1.add_edge(0, 2, 6)
    graph1.add_edge(0, 3, 5)
    graph1.add_edge(1, 3, 15)
    graph1.add_edge(2, 3, 4)
    
    print(f"\nTest Graph 1: 4 vertices, 5 edges")
    print(f"Edges: (0,1,10), (0,2,6), (0,3,5), (1,3,15), (2,3,4)")
    
    algorithms = MSTAlgorithms()
    verifier = MSTVerification()
    
    # Test all algorithms
    results = verifier.compare_mst_algorithms(graph1)
    
    for name, (weight, edges) in results.items():
        print(f"\n{name}: Weight = {weight}, Edges = {edges}")
        verification = verifier.verify_mst(graph1, edges)
        print(f"Verification: {verification}")
    
    # Test case 2: Complete graph
    print(f"\n" + "="*50)
    print(f"Test Graph 2: Complete graph K4")
    
    graph2 = WeightedGraph(4)
    weights = [
        (0, 1, 1), (0, 2, 3), (0, 3, 4),
        (1, 2, 2), (1, 3, 5), (2, 3, 6)
    ]
    
    for u, v, w in weights:
        graph2.add_edge(u, v, w)
    
    results2 = verifier.compare_mst_algorithms(graph2)
    
    for name, (weight, edges) in results2.items():
        print(f"{name}: Weight = {weight}, Edges = {edges}")

def demonstrate_union_find():
    """Demonstrate Union-Find operations"""
    print("\n=== Union-Find Demo ===")
    
    uf = UnionFind(5)
    print(f"Initial: 5 components")
    
    operations = [
        (0, 1, "Union 0 and 1"),
        (2, 3, "Union 2 and 3"),
        (0, 2, "Union 0 and 2 (connects components)"),
        (1, 4, "Union 1 and 4"),
    ]
    
    for u, v, description in operations:
        connected_before = uf.connected(u, v)
        union_result = uf.union(u, v)
        
        print(f"\n{description}")
        print(f"  Before: connected({u},{v}) = {connected_before}")
        print(f"  Union result: {union_result}")
        print(f"  Components: {uf.components}")
        print(f"  After: connected({u},{v}) = {uf.connected(u, v)}")

def demonstrate_mst_properties():
    """Demonstrate MST properties with examples"""
    print("\n=== MST Properties Demo ===")
    
    print("MST Fundamental Properties:")
    
    print("\n1. **Tree Property:**")
    print("   • Exactly V-1 edges for V vertices")
    print("   • Connected and acyclic")
    print("   • Removing any edge disconnects the graph")
    print("   • Adding any edge creates exactly one cycle")
    
    print("\n2. **Cut Property:**")
    print("   • Consider any cut (partition of vertices)")
    print("   • The minimum weight edge crossing the cut")
    print("   • is in some MST of the graph")
    print("   • This justifies Prim's algorithm")
    
    print("\n3. **Cycle Property:**")
    print("   • Consider any cycle in the graph")
    print("   • The maximum weight edge in the cycle")
    print("   • is not in any MST of the graph")
    print("   • This justifies Kruskal's algorithm")
    
    print("\n4. **Uniqueness:**")
    print("   • If all edge weights are distinct: MST is unique")
    print("   • If weights can be equal: multiple MSTs possible")
    print("   • All MSTs have the same total weight")
    
    # Example demonstrating cut property
    print("\nCut Property Example:")
    print("Graph: 0--1--2")
    print("       |  |  |")
    print("       3--4--5")
    print("Cut: {0,1,2} vs {3,4,5}")
    print("Edges crossing cut: (0,3), (1,4), (2,5)")
    print("Minimum weight edge crossing cut is in MST")

def demonstrate_algorithm_choices():
    """Demonstrate when to choose different MST algorithms"""
    print("\n=== Algorithm Choice Guide ===")
    
    print("Choosing the Right MST Algorithm:")
    
    print("\n**Graph Density:**")
    print("• **Dense graphs (E ≈ V²):**")
    print("  - Prim's with adjacency matrix: O(V²)")
    print("  - Simple implementation, cache-friendly")
    
    print("• **Sparse graphs (E << V²):**")
    print("  - Kruskal's algorithm: O(E log E)")
    print("  - Prim's with heap: O(E log V)")
    print("  - Kruskal's often faster for very sparse graphs")
    
    print("\n**Implementation Considerations:**")
    print("• **Prim's Algorithm:**")
    print("  - Easier to implement incrementally")
    print("  - Better for online algorithms")
    print("  - Natural for dense graphs")
    
    print("• **Kruskal's Algorithm:**")
    print("  - Requires Union-Find data structure")
    print("  - Better for sparse graphs")
    print("  - Natural for parallel processing")
    
    print("• **Borůvka's Algorithm:**")
    print("  - Best for parallel computation")
    print("  - Historically important")
    print("  - More complex implementation")
    
    print("\n**Practical Guidelines:**")
    print("• **E ≤ V log V:** Kruskal's algorithm")
    print("• **V log V < E ≤ V²/log V:** Prim's with heap")
    print("• **E > V²/log V:** Prim's with adjacency matrix")
    print("• **Parallel processing:** Borůvka's algorithm")

def analyze_mst_applications():
    """Analyze practical applications of MST algorithms"""
    print("\n=== MST Applications Analysis ===")
    
    print("Real-World MST Applications:")
    
    print("\n1. **Network Design:**")
    print("   • Telecommunication networks")
    print("   • Computer network topology")
    print("   • Power grid design")
    print("   • Transportation networks")
    print("   • Goal: Connect all nodes with minimum cost")
    
    print("\n2. **Approximation Algorithms:**")
    print("   • TSP 2-approximation using MST")
    print("   • Steiner tree approximation")
    print("   • Facility location problems")
    print("   • Network design with constraints")
    
    print("\n3. **Clustering:**")
    print("   • Single-linkage clustering")
    print("   • Remove k-1 heaviest edges for k clusters")
    print("   • Hierarchical clustering dendrograms")
    print("   • Image segmentation")
    
    print("\n4. **Circuit Design:**")
    print("   • VLSI layout optimization")
    print("   • Minimizing wire length")
    print("   • Reducing manufacturing cost")
    print("   • Signal integrity optimization")
    
    print("\n5. **Data Compression:**")
    print("   • Huffman coding tree construction")
    print("   • Optimal prefix codes")
    print("   • Minimum redundancy codes")
    print("   • Information theory applications")
    
    print("\nKey Insights:")
    print("• MST provides theoretical foundation for many optimization problems")
    print("• Often used as building block in more complex algorithms")
    print("• Approximation ratio guarantees for NP-hard problems")
    print("• Practical efficiency makes it suitable for large-scale applications")

if __name__ == "__main__":
    test_mst_algorithms()
    demonstrate_union_find()
    demonstrate_mst_properties()
    demonstrate_algorithm_choices()
    analyze_mst_applications()

"""
Minimum Spanning Tree (MST) Concepts:
1. Fundamental MST Algorithms (Prim's, Kruskal's, Borůvka's)
2. Union-Find Data Structure for Cycle Detection
3. Graph Representations (Matrix vs List vs Edge List)
4. MST Properties (Cut Property, Cycle Property)
5. Algorithm Selection Based on Graph Characteristics

Key Algorithm Insights:
- Prim's: Vertex-centric, grows tree from single component
- Kruskal's: Edge-centric, uses Union-Find for cycle detection
- Borůvka's: Component-centric, parallel algorithm design
- All algorithms produce optimal MST with same total weight

Implementation Patterns:
- Priority queue for efficient minimum selection
- Union-Find for efficient cycle detection
- Path compression and union by rank optimizations
- Early termination when MST is complete

Performance Characteristics:
- Dense graphs: Prim's with matrix O(V²)
- Sparse graphs: Kruskal's O(E log E) or Prim's with heap O(E log V)
- Parallel processing: Borůvka's O(E log V)
- Choice depends on graph density and implementation requirements

MST Properties and Theory:
- Cut property justifies Prim's algorithm correctness
- Cycle property justifies Kruskal's algorithm correctness
- MST uniqueness depends on edge weight distinctness
- Tree properties: V-1 edges, connected, acyclic

Real-world Applications:
- Network infrastructure design and optimization
- Approximation algorithms for NP-hard problems
- Clustering and data analysis applications
- Circuit design and layout optimization
- Data compression and coding theory

This collection provides comprehensive MST algorithm
foundations for advanced graph optimization problems.
"""
