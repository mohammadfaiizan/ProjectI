"""
Min-Cut Algorithms - Comprehensive Implementation
Difficulty: Medium

This file provides comprehensive implementations of minimum cut algorithms for graphs,
including various approaches for finding minimum cuts in different types of networks.

Key Concepts:
1. Min-Cut Max-Flow Theorem
2. Stoer-Wagner Algorithm for Undirected Graphs
3. Karger's Randomized Algorithm
4. Global Min-Cut vs s-t Min-Cut
5. Cut Tree Construction
6. Applications in Network Reliability
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict, deque
import heapq
import random
import math
from copy import deepcopy

class MinCutAlgorithms:
    """Comprehensive min-cut algorithm implementations"""
    
    def __init__(self):
        self.reset_statistics()
    
    def reset_statistics(self):
        """Reset algorithm statistics"""
        self.stats = {
            'iterations': 0,
            'contractions': 0,
            'flow_computations': 0,
            'cut_value': 0,
            'vertices_processed': 0
        }
    
    def min_cut_max_flow(self, graph: Dict[int, Dict[int, int]], source: int, sink: int) -> Dict:
        """
        Approach 1: Min-Cut via Max-Flow (Ford-Fulkerson)
        
        Find s-t min-cut using max-flow algorithm and residual graph analysis.
        
        Time: O(V * E^2) using Edmonds-Karp
        Space: O(V + E)
        """
        self.reset_statistics()
        
        # Build residual graph
        residual = defaultdict(lambda: defaultdict(int))
        
        for u in graph:
            for v in graph[u]:
                residual[u][v] = graph[u][v]
                residual[v][u] = 0  # Initialize reverse edge
        
        def bfs_find_path():
            """BFS to find augmenting path"""
            parent = {source: None}
            visited = {source}
            queue = deque([source])
            
            while queue:
                current = queue.popleft()
                
                for neighbor in residual[current]:
                    if neighbor not in visited and residual[current][neighbor] > 0:
                        visited.add(neighbor)
                        parent[neighbor] = current
                        queue.append(neighbor)
                        
                        if neighbor == sink:
                            return parent
            
            return None
        
        max_flow = 0
        
        # Find maximum flow
        while True:
            self.stats['iterations'] += 1
            
            parent = bfs_find_path()
            if parent is None:
                break
            
            # Find bottleneck capacity
            path_flow = float('inf')
            current = sink
            
            while current != source:
                prev = parent[current]
                path_flow = min(path_flow, residual[prev][current])
                current = prev
            
            # Update residual graph
            current = sink
            while current != source:
                prev = parent[current]
                residual[prev][current] -= path_flow
                residual[current][prev] += path_flow
                current = prev
            
            max_flow += path_flow
            self.stats['flow_computations'] += 1
        
        # Find min-cut using reachability in residual graph
        reachable = set()
        queue = deque([source])
        reachable.add(source)
        
        while queue:
            current = queue.popleft()
            for neighbor in residual[current]:
                if neighbor not in reachable and residual[current][neighbor] > 0:
                    reachable.add(neighbor)
                    queue.append(neighbor)
        
        # Find cut edges
        cut_edges = []
        cut_capacity = 0
        
        for u in graph:
            for v in graph[u]:
                if u in reachable and v not in reachable:
                    cut_edges.append((u, v, graph[u][v]))
                    cut_capacity += graph[u][v]
        
        self.stats['cut_value'] = cut_capacity
        
        return {
            'cut_value': cut_capacity,
            'cut_edges': cut_edges,
            'source_side': list(reachable),
            'sink_side': [v for v in graph.keys() if v not in reachable],
            'max_flow': max_flow,
            'algorithm': 'max_flow_min_cut',
            'statistics': self.stats.copy()
        }
    
    def global_min_cut_stoer_wagner(self, graph: Dict[int, Dict[int, int]]) -> Dict:
        """
        Approach 2: Stoer-Wagner Algorithm for Global Min-Cut
        
        Find global minimum cut in undirected weighted graph.
        
        Time: O(V^3)
        Space: O(V^2)
        """
        self.reset_statistics()
        
        if not graph:
            return {'cut_value': 0, 'cut_partition': ([], []), 'algorithm': 'stoer_wagner'}
        
        # Convert to adjacency matrix for easier manipulation
        vertices = list(graph.keys())
        n = len(vertices)
        vertex_to_idx = {v: i for i, v in enumerate(vertices)}
        
        # Build adjacency matrix
        adj_matrix = [[0] * n for _ in range(n)]
        for u in graph:
            for v in graph[u]:
                if u != v:  # No self-loops
                    i, j = vertex_to_idx[u], vertex_to_idx[v]
                    adj_matrix[i][j] = graph[u][v]
                    adj_matrix[j][i] = graph[u][v]  # Undirected
        
        min_cut_value = float('inf')
        best_partition = None
        
        # Stoer-Wagner phases
        active_vertices = list(range(n))
        
        while len(active_vertices) > 1:
            self.stats['iterations'] += 1
            
            # Find most tightly connected pair
            cut_value, s, t = self._stoer_wagner_phase(adj_matrix, active_vertices)
            
            if cut_value < min_cut_value:
                min_cut_value = cut_value
                # Create partition
                s_side = [vertices[s]]
                t_side = [vertices[i] for i in active_vertices if i != s]
                best_partition = (s_side, t_side)
            
            # Contract s and t
            self._contract_vertices(adj_matrix, s, t, active_vertices)
            active_vertices.remove(t)
            self.stats['contractions'] += 1
        
        self.stats['cut_value'] = min_cut_value
        
        return {
            'cut_value': min_cut_value,
            'cut_partition': best_partition,
            'algorithm': 'stoer_wagner',
            'statistics': self.stats.copy()
        }
    
    def _stoer_wagner_phase(self, adj_matrix: List[List[int]], active_vertices: List[int]) -> Tuple[int, int, int]:
        """Single phase of Stoer-Wagner algorithm"""
        n = len(active_vertices)
        if n < 2:
            return 0, active_vertices[0], active_vertices[0]
        
        # Start with arbitrary vertex
        added = [False] * len(adj_matrix)
        weights = [0] * len(adj_matrix)
        
        # Add first vertex
        start = active_vertices[0]
        added[start] = True
        
        # Update weights to first vertex
        for v in active_vertices:
            if v != start:
                weights[v] = adj_matrix[start][v]
        
        s = t = start
        
        # Add vertices one by one
        for _ in range(n - 1):
            # Find vertex with maximum weight to current set
            max_weight = -1
            next_vertex = -1
            
            for v in active_vertices:
                if not added[v] and weights[v] > max_weight:
                    max_weight = weights[v]
                    next_vertex = v
            
            s = t
            t = next_vertex
            added[t] = True
            
            # Update weights
            for v in active_vertices:
                if not added[v]:
                    weights[v] += adj_matrix[t][v]
        
        # Cut value is the weight of t to the rest
        cut_value = sum(adj_matrix[t][v] for v in active_vertices if v != t)
        
        return cut_value, s, t
    
    def _contract_vertices(self, adj_matrix: List[List[int]], s: int, t: int, active_vertices: List[int]):
        """Contract vertices s and t in adjacency matrix"""
        # Merge t into s
        for v in active_vertices:
            if v != s and v != t:
                adj_matrix[s][v] += adj_matrix[t][v]
                adj_matrix[v][s] += adj_matrix[v][t]
                adj_matrix[t][v] = adj_matrix[v][t] = 0
        
        adj_matrix[s][t] = adj_matrix[t][s] = 0
    
    def global_min_cut_karger(self, graph: Dict[int, Dict[int, int]], iterations: int = None) -> Dict:
        """
        Approach 3: Karger's Randomized Min-Cut Algorithm
        
        Find global minimum cut using randomized edge contraction.
        
        Time: O(V^2 * log V) with high probability
        Space: O(V + E)
        """
        self.reset_statistics()
        
        if not graph:
            return {'cut_value': 0, 'algorithm': 'karger'}
        
        vertices = list(graph.keys())
        n = len(vertices)
        
        if iterations is None:
            iterations = max(1, int(n * n * math.log(n)))
        
        min_cut_value = float('inf')
        best_partition = None
        
        for trial in range(iterations):
            self.stats['iterations'] += 1
            
            # Create edge list for this trial
            edges = []
            for u in graph:
                for v in graph[u]:
                    if u < v:  # Avoid duplicate edges in undirected graph
                        weight = graph[u][v]
                        edges.extend([(u, v)] * weight)  # Multiple copies for weighted edges
            
            if len(edges) < 2:
                continue
            
            # Contract until 2 vertices remain
            vertex_groups = {v: {v} for v in vertices}
            remaining_vertices = set(vertices)
            
            while len(remaining_vertices) > 2:
                # Pick random edge
                edge_idx = random.randint(0, len(edges) - 1)
                u, v = edges[edge_idx]
                
                # Find current groups
                group_u = None
                group_v = None
                
                for vertex in remaining_vertices:
                    if u in vertex_groups[vertex]:
                        group_u = vertex
                    if v in vertex_groups[vertex]:
                        group_v = vertex
                
                if group_u == group_v:
                    # Edge within same group, remove it
                    edges.pop(edge_idx)
                    continue
                
                # Contract groups
                vertex_groups[group_u].update(vertex_groups[group_v])
                remaining_vertices.remove(group_v)
                
                # Update edges
                new_edges = []
                for edge_u, edge_v in edges:
                    # Map vertices to their groups
                    mapped_u = group_u if (edge_u in vertex_groups[group_u]) else edge_u
                    mapped_v = group_u if (edge_v in vertex_groups[group_u]) else edge_v
                    
                    # Find correct group for mapped vertices
                    for vertex in remaining_vertices:
                        if mapped_u in vertex_groups[vertex]:
                            mapped_u = vertex
                        if mapped_v in vertex_groups[vertex]:
                            mapped_v = vertex
                    
                    if mapped_u != mapped_v:
                        new_edges.append((mapped_u, mapped_v))
                
                edges = new_edges
                self.stats['contractions'] += 1
            
            # Count cut edges
            cut_value = len(edges)
            
            if cut_value < min_cut_value:
                min_cut_value = cut_value
                remaining_list = list(remaining_vertices)
                if len(remaining_list) >= 2:
                    partition_1 = list(vertex_groups[remaining_list[0]])
                    partition_2 = list(vertex_groups[remaining_list[1]])
                    best_partition = (partition_1, partition_2)
        
        self.stats['cut_value'] = min_cut_value
        
        return {
            'cut_value': min_cut_value,
            'cut_partition': best_partition,
            'algorithm': 'karger',
            'iterations_run': iterations,
            'statistics': self.stats.copy()
        }
    
    def min_cut_push_relabel(self, graph: Dict[int, Dict[int, int]], source: int, sink: int) -> Dict:
        """
        Approach 4: Min-Cut using Push-Relabel Max-Flow
        
        Find s-t min-cut using push-relabel algorithm for max-flow.
        
        Time: O(V^2 * E)
        Space: O(V + E)
        """
        self.reset_statistics()
        
        vertices = set([source, sink])
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
        
        vertices = list(vertices)
        n = len(vertices)
        
        # Initialize data structures
        capacity = defaultdict(lambda: defaultdict(int))
        flow = defaultdict(lambda: defaultdict(int))
        excess = defaultdict(int)
        height = defaultdict(int)
        
        # Build capacity matrix
        for u in graph:
            for v in graph[u]:
                capacity[u][v] = graph[u][v]
        
        # Initialize preflow
        height[source] = n
        for v in vertices:
            if capacity[source][v] > 0:
                flow[source][v] = capacity[source][v]
                flow[v][source] = -capacity[source][v]
                excess[v] = capacity[source][v]
                excess[source] -= capacity[source][v]
        
        def push(u, v):
            """Push flow from u to v"""
            send = min(excess[u], capacity[u][v] - flow[u][v])
            flow[u][v] += send
            flow[v][u] -= send
            excess[u] -= send
            excess[v] += send
            return send > 0
        
        def relabel(u):
            """Increase height of vertex u"""
            min_height = float('inf')
            for v in vertices:
                if capacity[u][v] - flow[u][v] > 0:
                    min_height = min(min_height, height[v])
            
            if min_height < float('inf'):
                height[u] = min_height + 1
                return True
            return False
        
        def get_active_vertex():
            """Find vertex with positive excess"""
            for v in vertices:
                if v != source and v != sink and excess[v] > 0:
                    return v
            return None
        
        # Main push-relabel loop
        while True:
            self.stats['iterations'] += 1
            
            u = get_active_vertex()
            if u is None:
                break
            
            # Try to push
            pushed = False
            for v in vertices:
                if (capacity[u][v] - flow[u][v] > 0 and 
                    height[u] == height[v] + 1):
                    if push(u, v):
                        pushed = True
                        break
            
            # If no push possible, relabel
            if not pushed:
                relabel(u)
        
        max_flow = sum(flow[source][v] for v in vertices)
        
        # Find min-cut
        reachable = set()
        queue = deque([source])
        reachable.add(source)
        
        while queue:
            current = queue.popleft()
            for v in vertices:
                if v not in reachable and capacity[current][v] - flow[current][v] > 0:
                    reachable.add(v)
                    queue.append(v)
        
        cut_edges = []
        cut_capacity = 0
        
        for u in graph:
            for v in graph[u]:
                if u in reachable and v not in reachable:
                    cut_edges.append((u, v, graph[u][v]))
                    cut_capacity += graph[u][v]
        
        self.stats['cut_value'] = cut_capacity
        
        return {
            'cut_value': cut_capacity,
            'cut_edges': cut_edges,
            'source_side': list(reachable),
            'sink_side': [v for v in vertices if v not in reachable],
            'max_flow': max_flow,
            'algorithm': 'push_relabel_min_cut',
            'statistics': self.stats.copy()
        }
    
    def all_pairs_min_cut(self, graph: Dict[int, Dict[int, int]]) -> Dict:
        """
        Approach 5: All-Pairs Min-Cut using Multiple Max-Flow
        
        Find minimum cut between all pairs of vertices.
        
        Time: O(V^3 * E^2)
        Space: O(V^3)
        """
        self.reset_statistics()
        
        vertices = list(graph.keys())
        n = len(vertices)
        
        # Store all pairwise min-cuts
        min_cuts = {}
        global_min_cut = float('inf')
        global_min_pair = None
        
        for i in range(n):
            for j in range(i + 1, n):
                source, sink = vertices[i], vertices[j]
                
                # Compute s-t min-cut
                result = self.min_cut_max_flow(graph, source, sink)
                cut_value = result['cut_value']
                
                min_cuts[(source, sink)] = result
                
                if cut_value < global_min_cut:
                    global_min_cut = cut_value
                    global_min_pair = (source, sink)
                
                self.stats['vertices_processed'] += 1
        
        return {
            'global_min_cut': global_min_cut,
            'global_min_pair': global_min_pair,
            'all_min_cuts': min_cuts,
            'algorithm': 'all_pairs_min_cut',
            'statistics': self.stats.copy()
        }
    
    def min_cut_tree_construction(self, graph: Dict[int, Dict[int, int]]) -> Dict:
        """
        Approach 6: Gomory-Hu Cut Tree Construction
        
        Build cut tree that preserves all pairwise min-cut values.
        
        Time: O(V^2 * max_flow_time)
        Space: O(V^2)
        """
        self.reset_statistics()
        
        vertices = list(graph.keys())
        n = len(vertices)
        
        if n <= 1:
            return {'tree': {}, 'algorithm': 'cut_tree'}
        
        # Initialize tree - each vertex connected to first vertex
        tree = defaultdict(dict)
        parent = {v: vertices[0] for v in vertices[1:]}
        parent[vertices[0]] = None
        
        # Process each vertex
        for i in range(1, n):
            current = vertices[i]
            target = parent[current]
            
            # Find min-cut between current and its parent
            result = self.min_cut_max_flow(graph, current, target)
            cut_value = result['cut_value']
            source_side = set(result['source_side'])
            
            # Add edge to tree
            tree[current][target] = cut_value
            tree[target][current] = cut_value
            
            # Update parent relationships
            for j in range(i + 1, n):
                other = vertices[j]
                if parent[other] == target and other in source_side:
                    parent[other] = current
            
            self.stats['vertices_processed'] += 1
        
        return {
            'tree': dict(tree),
            'cut_values': {(u, v): tree[u][v] for u in tree for v in tree[u] if u < v},
            'algorithm': 'cut_tree',
            'statistics': self.stats.copy()
        }

def test_min_cut_algorithms():
    """Test all min-cut algorithms"""
    solver = MinCutAlgorithms()
    
    print("=== Testing Min-Cut Algorithms ===")
    
    # Test graphs
    test_graphs = [
        # Simple graph
        ({
            0: {1: 10, 2: 8},
            1: {3: 5, 2: 2},
            2: {3: 10},
            3: {}
        }, [(0, 3), (1, 2)], "Simple graph"),
        
        # Symmetric graph for global min-cut
        ({
            0: {1: 5, 2: 3},
            1: {0: 5, 2: 2, 3: 4},
            2: {0: 3, 1: 2, 3: 6},
            3: {1: 4, 2: 6}
        }, [(0, 1), (0, 2), (1, 3), (2, 3)], "Symmetric graph"),
        
        # Linear chain
        ({
            0: {1: 3},
            1: {2: 2},
            2: {3: 4},
            3: {}
        }, [(0, 3), (1, 2)], "Linear chain"),
    ]
    
    for graph, test_pairs, description in test_graphs:
        print(f"\n--- {description} ---")
        print(f"Graph: {graph}")
        
        # Test s-t min-cut algorithms
        for source, sink in test_pairs:
            print(f"\nMin-cut from {source} to {sink}:")
            
            algorithms = [
                ("Max-Flow", solver.min_cut_max_flow),
                ("Push-Relabel", solver.min_cut_push_relabel),
            ]
            
            for alg_name, alg_func in algorithms:
                try:
                    result = alg_func(graph, source, sink)
                    cut_value = result['cut_value']
                    iterations = result['statistics']['iterations']
                    print(f"  {alg_name:12} | Cut: {cut_value:2} | Iter: {iterations:2}")
                except Exception as e:
                    print(f"  {alg_name:12} | ERROR: {str(e)[:20]}")
        
        # Test global min-cut algorithms
        print(f"\nGlobal min-cut:")
        
        global_algorithms = [
            ("Stoer-Wagner", solver.global_min_cut_stoer_wagner),
            ("Karger", lambda g: solver.global_min_cut_karger(g, 50)),
        ]
        
        for alg_name, alg_func in global_algorithms:
            try:
                result = alg_func(graph)
                cut_value = result['cut_value']
                print(f"  {alg_name:12} | Cut: {cut_value:2}")
            except Exception as e:
                print(f"  {alg_name:12} | ERROR: {str(e)[:20]}")

def demonstrate_min_cut_applications():
    """Demonstrate practical applications of min-cut algorithms"""
    print("\n=== Min-Cut Applications Demo ===")
    
    print("Network Reliability Analysis:")
    
    # Example: Communication network
    network = {
        'A': {'B': 10, 'C': 15},
        'B': {'A': 10, 'D': 12, 'E': 8},
        'C': {'A': 15, 'E': 20, 'F': 5},
        'D': {'B': 12, 'F': 7},
        'E': {'B': 8, 'C': 20, 'F': 10},
        'F': {'C': 5, 'D': 7, 'E': 10}
    }
    
    print(f"Network topology: {network}")
    
    # Convert to numeric for algorithms
    vertex_map = {v: i for i, v in enumerate(network.keys())}
    numeric_graph = {}
    
    for u in network:
        numeric_graph[vertex_map[u]] = {}
        for v in network[u]:
            numeric_graph[vertex_map[u]][vertex_map[v]] = network[u][v]
    
    solver = MinCutAlgorithms()
    
    # Find global minimum cut
    result = solver.global_min_cut_stoer_wagner(numeric_graph)
    
    print(f"\nNetwork Analysis Results:")
    print(f"• Global min-cut value: {result['cut_value']}")
    print(f"• This represents the minimum capacity that must fail to disconnect the network")
    print(f"• Network reliability bottleneck capacity")
    
    # Application scenarios
    print(f"\nPractical Applications:")
    print(f"1. **Network Design:** Identify critical links to reinforce")
    print(f"2. **Fault Tolerance:** Plan redundancy for minimum cut edges")
    print(f"3. **Attack Resilience:** Understand vulnerability points")
    print(f"4. **Capacity Planning:** Optimize resource allocation")

def analyze_cut_theory():
    """Analyze theoretical aspects of min-cut problems"""
    print("\n=== Min-Cut Theory Analysis ===")
    
    print("Fundamental Theorems:")
    
    print("\n1. **Max-Flow Min-Cut Theorem:**")
    print("   • In any flow network, max s-t flow = min s-t cut")
    print("   • Provides duality between flow and cut problems")
    print("   • Enables efficient min-cut computation via max-flow")
    print("   • Guarantees optimality of flow-based algorithms")
    
    print("\n2. **Global Min-Cut Properties:**")
    print("   • Global min-cut ≤ any s-t min-cut")
    print("   • Can be found without specifying source/sink")
    print("   • Stoer-Wagner provides deterministic O(V³) algorithm")
    print("   • Karger's algorithm gives randomized approach")
    
    print("\n3. **Cut Tree (Gomory-Hu Tree):**")
    print("   • Compact representation of all pairwise min-cuts")
    print("   • Tree with V-1 edges preserves all min-cut values")
    print("   • Path between vertices gives their min-cut value")
    print("   • Enables efficient all-pairs min-cut queries")
    
    print("\n4. **Complexity Landscape:**")
    print("   • s-t min-cut: Same as max-flow complexity")
    print("   • Global min-cut: O(V³) deterministic, O(V² log V) randomized")
    print("   • All-pairs: O(V) queries on cut tree after O(V² max-flow) preprocessing")
    print("   • Weighted vs unweighted: Similar complexity bounds")
    
    print("\n5. **Algorithmic Techniques:**")
    print("   • Flow-based: Leverage max-flow algorithms")
    print("   • Contraction: Karger's randomized edge contraction")
    print("   • Phase-based: Stoer-Wagner systematic vertex addition")
    print("   • Tree construction: Gomory-Hu incremental approach")

def demonstrate_optimization_strategies():
    """Demonstrate optimization strategies for min-cut problems"""
    print("\n=== Min-Cut Optimization Strategies ===")
    
    print("Algorithm Selection Guidelines:")
    
    print("\n1. **Problem Type Considerations:**")
    print("   • s-t min-cut: Use max-flow algorithms (Edmonds-Karp, Push-Relabel)")
    print("   • Global min-cut: Stoer-Wagner for deterministic, Karger for randomized")
    print("   • Multiple queries: Build cut tree for O(1) query time")
    print("   • Unweighted graphs: Specialized algorithms may be faster")
    
    print("\n2. **Graph Structure Optimization:**")
    print("   • Sparse graphs: Prefer algorithms with better E dependence")
    print("   • Dense graphs: Matrix-based algorithms may be efficient")
    print("   • Planar graphs: Specialized algorithms available")
    print("   • Small cuts: Early termination strategies")
    
    print("\n3. **Implementation Optimizations:**")
    print("   • Adjacency representation: Lists vs matrices")
    print("   • Priority queues: Fibonacci heaps for dense graphs")
    print("   • Parallelization: Independent flow computations")
    print("   • Memory optimization: Streaming algorithms for large graphs")
    
    print("\n4. **Approximation Strategies:**")
    print("   • Karger's algorithm: Trade accuracy for speed")
    print("   • Sampling: Random edge/vertex sampling")
    print("   • Sparsification: Maintain cut structure with fewer edges")
    print("   • Local search: Iterative improvement heuristics")
    
    print("\n5. **Practical Considerations:**")
    print("   • Numerical stability: Integer vs floating-point arithmetic")
    print("   • Preprocessing: Graph simplification and reduction")
    print("   • Caching: Reuse computations for similar queries")
    print("   • Monitoring: Progress tracking for long-running algorithms")

if __name__ == "__main__":
    test_min_cut_algorithms()
    demonstrate_min_cut_applications()
    analyze_cut_theory()
    demonstrate_optimization_strategies()

"""
Min-Cut Algorithms - Key Insights:

1. **Algorithm Categories:**
   - Flow-based: Leverage max-flow min-cut theorem
   - Contraction-based: Karger's randomized approach
   - Phase-based: Stoer-Wagner systematic method
   - Tree-based: Gomory-Hu cut tree construction

2. **Complexity Trade-offs:**
   - Deterministic vs randomized algorithms
   - Worst-case vs expected performance
   - Space efficiency vs query time
   - Preprocessing cost vs query efficiency

3. **Problem Variants:**
   - s-t min-cut: Specific source and sink
   - Global min-cut: Minimum over all possible cuts
   - All-pairs: Min-cut between every vertex pair
   - Weighted vs unweighted graphs

4. **Theoretical Foundations:**
   - Max-flow min-cut duality
   - Cut tree properties and construction
   - Randomized algorithm analysis
   - Approximation guarantees

5. **Practical Applications:**
   - Network reliability and fault tolerance
   - Image segmentation and clustering
   - VLSI design and circuit analysis
   - Social network analysis

Min-cut algorithms provide fundamental tools for
network analysis and optimization problems.
"""
