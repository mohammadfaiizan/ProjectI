"""
Global Min-Cut Algorithms - Comprehensive Implementation
Difficulty: Hard

This file provides comprehensive implementations of global minimum cut algorithms,
which find the minimum cut in an undirected graph without specifying source and sink.

Key Concepts:
1. Stoer-Wagner Algorithm
2. Karger's Randomized Contraction
3. Karger-Stein Algorithm
4. Nagamochi-Ibaraki Algorithm
5. Gomory-Hu Tree Construction
6. Cut Tree Applications
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict, deque
import heapq
import random
import math
from copy import deepcopy

class GlobalMinCutAlgorithms:
    """Comprehensive global min-cut algorithm implementations"""
    
    def __init__(self):
        self.reset_statistics()
    
    def reset_statistics(self):
        """Reset algorithm statistics"""
        self.stats = {
            'phases': 0,
            'contractions': 0,
            'iterations': 0,
            'cut_value': float('inf'),
            'vertices_processed': 0,
            'edges_processed': 0
        }
    
    def global_min_cut_stoer_wagner(self, graph: Dict[int, Dict[int, int]]) -> Dict:
        """
        Approach 1: Stoer-Wagner Algorithm
        
        Deterministic algorithm for global min-cut in weighted undirected graphs.
        
        Time: O(V^3)
        Space: O(V^2)
        """
        self.reset_statistics()
        
        if not graph:
            return {'cut_value': 0, 'cut_partition': ([], []), 'algorithm': 'stoer_wagner'}
        
        vertices = list(graph.keys())
        n = len(vertices)
        
        if n <= 1:
            return {'cut_value': 0, 'cut_partition': (vertices, []), 'algorithm': 'stoer_wagner'}
        
        # Convert to adjacency matrix for easier manipulation
        vertex_to_idx = {v: i for i, v in enumerate(vertices)}
        adj_matrix = [[0] * n for _ in range(n)]
        
        for u in graph:
            for v in graph[u]:
                if u != v:  # No self-loops
                    i, j = vertex_to_idx[u], vertex_to_idx[v]
                    adj_matrix[i][j] = graph[u][v]
                    adj_matrix[j][i] = graph[u][v]
        
        min_cut_value = float('inf')
        best_partition = None
        
        # Stoer-Wagner phases
        active_vertices = list(range(n))
        vertex_sets = [[vertices[i]] for i in range(n)]  # Track which original vertices each represents
        
        while len(active_vertices) > 1:
            self.stats['phases'] += 1
            
            # Find most tightly connected pair
            cut_value, s, t = self._stoer_wagner_phase(adj_matrix, active_vertices)
            
            if cut_value < min_cut_value:
                min_cut_value = cut_value
                # Create partition
                s_side = vertex_sets[s][:]
                t_side = []
                for i in active_vertices:
                    if i != s:
                        t_side.extend(vertex_sets[i])
                best_partition = (s_side, t_side)
            
            # Contract s and t
            self._contract_vertices_stoer_wagner(adj_matrix, s, t, active_vertices, vertex_sets)
            active_vertices.remove(t)
            self.stats['contractions'] += 1
        
        self.stats['cut_value'] = min_cut_value
        
        return {
            'cut_value': min_cut_value,
            'cut_partition': best_partition,
            'algorithm': 'stoer_wagner',
            'statistics': self.stats.copy()
        }
    
    def global_min_cut_karger_basic(self, graph: Dict[int, Dict[int, int]], iterations: int = None) -> Dict:
        """
        Approach 2: Karger's Basic Randomized Algorithm
        
        Randomized contraction algorithm with multiple trials.
        
        Time: O(V^2 * log V) with high probability
        Space: O(V + E)
        """
        self.reset_statistics()
        
        if not graph:
            return {'cut_value': 0, 'algorithm': 'karger_basic'}
        
        vertices = list(graph.keys())
        n = len(vertices)
        
        if n <= 1:
            return {'cut_value': 0, 'cut_partition': (vertices, []), 'algorithm': 'karger_basic'}
        
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
                    if u < v:  # Avoid duplicate edges
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
            'algorithm': 'karger_basic',
            'iterations_run': iterations,
            'statistics': self.stats.copy()
        }
    
    def global_min_cut_karger_stein(self, graph: Dict[int, Dict[int, int]]) -> Dict:
        """
        Approach 3: Karger-Stein Algorithm
        
        Improved randomized algorithm with recursive structure.
        
        Time: O(V^2 * log^3 V) with high probability
        Space: O(V + E)
        """
        self.reset_statistics()
        
        if not graph:
            return {'cut_value': 0, 'algorithm': 'karger_stein'}
        
        vertices = list(graph.keys())
        n = len(vertices)
        
        if n <= 6:
            # Use brute force for small graphs
            return self._brute_force_min_cut(graph)
        
        # Contract to n/sqrt(2) vertices
        target_size = max(2, int(n / math.sqrt(2)))
        
        # Run two independent contractions
        contracted_graph1 = self._contract_to_size(graph, target_size)
        contracted_graph2 = self._contract_to_size(graph, target_size)
        
        # Recursively solve both
        result1 = self.global_min_cut_karger_stein(contracted_graph1)
        result2 = self.global_min_cut_karger_stein(contracted_graph2)
        
        # Return better result
        if result1['cut_value'] <= result2['cut_value']:
            result1['algorithm'] = 'karger_stein'
            return result1
        else:
            result2['algorithm'] = 'karger_stein'
            return result2
    
    def global_min_cut_nagamochi_ibaraki(self, graph: Dict[int, Dict[int, int]]) -> Dict:
        """
        Approach 4: Nagamochi-Ibaraki Algorithm
        
        Deterministic algorithm based on maximum adjacency ordering.
        
        Time: O(V^2 + VE)
        Space: O(V + E)
        """
        self.reset_statistics()
        
        if not graph:
            return {'cut_value': 0, 'algorithm': 'nagamochi_ibaraki'}
        
        vertices = list(graph.keys())
        n = len(vertices)
        
        if n <= 1:
            return {'cut_value': 0, 'cut_partition': (vertices, []), 'algorithm': 'nagamochi_ibaraki'}
        
        min_cut_value = float('inf')
        best_partition = None
        
        # Work on a copy of the graph
        working_graph = deepcopy(graph)
        vertex_sets = {v: {v} for v in vertices}
        
        while len(working_graph) > 1:
            self.stats['phases'] += 1
            
            # Find maximum adjacency ordering
            ordering = self._maximum_adjacency_ordering(working_graph)
            
            if len(ordering) < 2:
                break
            
            # Last two vertices in ordering
            s, t = ordering[-2], ordering[-1]
            
            # Calculate cut value (degree of t)
            cut_value = sum(working_graph[t].values())
            
            if cut_value < min_cut_value:
                min_cut_value = cut_value
                # Create partition
                t_side = list(vertex_sets[t])
                s_side = []
                for v in working_graph:
                    if v != t:
                        s_side.extend(vertex_sets[v])
                best_partition = (s_side, t_side)
            
            # Contract s and t
            self._contract_vertices_ni(working_graph, s, t, vertex_sets)
            self.stats['contractions'] += 1
        
        self.stats['cut_value'] = min_cut_value
        
        return {
            'cut_value': min_cut_value,
            'cut_partition': best_partition,
            'algorithm': 'nagamochi_ibaraki',
            'statistics': self.stats.copy()
        }
    
    def global_min_cut_gomory_hu_tree(self, graph: Dict[int, Dict[int, int]]) -> Dict:
        """
        Approach 5: Gomory-Hu Tree Construction
        
        Build cut tree that preserves all pairwise min-cut values.
        
        Time: O(V^2 * max_flow_time)
        Space: O(V^2)
        """
        self.reset_statistics()
        
        if not graph:
            return {'cut_value': 0, 'algorithm': 'gomory_hu_tree'}
        
        vertices = list(graph.keys())
        n = len(vertices)
        
        if n <= 1:
            return {'cut_value': 0, 'cut_partition': (vertices, []), 'algorithm': 'gomory_hu_tree'}
        
        # Build Gomory-Hu tree
        tree = defaultdict(dict)
        parent = {v: vertices[0] for v in vertices[1:]}
        parent[vertices[0]] = None
        
        min_cut_value = float('inf')
        best_partition = None
        
        # Process each vertex
        for i in range(1, n):
            current = vertices[i]
            target = parent[current]
            
            # Find min-cut between current and its parent
            cut_value, partition = self._max_flow_min_cut(graph, current, target)
            
            if cut_value < min_cut_value:
                min_cut_value = cut_value
                best_partition = partition
            
            # Add edge to tree
            tree[current][target] = cut_value
            tree[target][current] = cut_value
            
            # Update parent relationships
            source_side = set(partition[0])
            for j in range(i + 1, n):
                other = vertices[j]
                if parent[other] == target and other in source_side:
                    parent[other] = current
            
            self.stats['vertices_processed'] += 1
        
        self.stats['cut_value'] = min_cut_value
        
        return {
            'cut_value': min_cut_value,
            'cut_partition': best_partition,
            'cut_tree': dict(tree),
            'algorithm': 'gomory_hu_tree',
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
    
    def _contract_vertices_stoer_wagner(self, adj_matrix: List[List[int]], s: int, t: int, 
                                      active_vertices: List[int], vertex_sets: List[List]):
        """Contract vertices s and t in Stoer-Wagner"""
        # Merge t into s
        for v in active_vertices:
            if v != s and v != t:
                adj_matrix[s][v] += adj_matrix[t][v]
                adj_matrix[v][s] += adj_matrix[v][t]
                adj_matrix[t][v] = adj_matrix[v][t] = 0
        
        adj_matrix[s][t] = adj_matrix[t][s] = 0
        
        # Merge vertex sets
        vertex_sets[s].extend(vertex_sets[t])
    
    def _contract_to_size(self, graph: Dict[int, Dict[int, int]], target_size: int) -> Dict[int, Dict[int, int]]:
        """Contract graph to target size using random contractions"""
        if len(graph) <= target_size:
            return graph
        
        # Create edge list
        edges = []
        for u in graph:
            for v in graph[u]:
                if u < v:
                    weight = graph[u][v]
                    edges.extend([(u, v)] * weight)
        
        # Contract until target size
        vertex_groups = {v: {v} for v in graph.keys()}
        remaining_vertices = set(graph.keys())
        
        while len(remaining_vertices) > target_size and edges:
            # Pick random edge
            edge_idx = random.randint(0, len(edges) - 1)
            u, v = edges[edge_idx]
            
            # Find current groups
            group_u = group_v = None
            for vertex in remaining_vertices:
                if u in vertex_groups[vertex]:
                    group_u = vertex
                if v in vertex_groups[vertex]:
                    group_v = vertex
            
            if group_u == group_v:
                edges.pop(edge_idx)
                continue
            
            # Contract groups
            vertex_groups[group_u].update(vertex_groups[group_v])
            remaining_vertices.remove(group_v)
            
            # Update edges
            new_edges = []
            for edge_u, edge_v in edges:
                mapped_u = mapped_v = None
                for vertex in remaining_vertices:
                    if edge_u in vertex_groups[vertex]:
                        mapped_u = vertex
                    if edge_v in vertex_groups[vertex]:
                        mapped_v = vertex
                
                if mapped_u and mapped_v and mapped_u != mapped_v:
                    new_edges.append((mapped_u, mapped_v))
            
            edges = new_edges
        
        # Build contracted graph
        contracted_graph = defaultdict(lambda: defaultdict(int))
        
        for u, v in edges:
            contracted_graph[u][v] += 1
            contracted_graph[v][u] += 1
        
        return dict(contracted_graph)
    
    def _brute_force_min_cut(self, graph: Dict[int, Dict[int, int]]) -> Dict:
        """Brute force min-cut for small graphs"""
        vertices = list(graph.keys())
        n = len(vertices)
        
        if n <= 1:
            return {'cut_value': 0, 'cut_partition': (vertices, [])}
        
        min_cut_value = float('inf')
        best_partition = None
        
        # Try all possible partitions
        for mask in range(1, (1 << n) - 1):
            partition_1 = []
            partition_2 = []
            
            for i in range(n):
                if mask & (1 << i):
                    partition_1.append(vertices[i])
                else:
                    partition_2.append(vertices[i])
            
            # Calculate cut value
            cut_value = 0
            for u in partition_1:
                for v in partition_2:
                    if v in graph[u]:
                        cut_value += graph[u][v]
            
            if cut_value < min_cut_value:
                min_cut_value = cut_value
                best_partition = (partition_1, partition_2)
        
        return {'cut_value': min_cut_value, 'cut_partition': best_partition}
    
    def _maximum_adjacency_ordering(self, graph: Dict[int, Dict[int, int]]) -> List[int]:
        """Find maximum adjacency ordering for Nagamochi-Ibaraki"""
        vertices = list(graph.keys())
        if not vertices:
            return []
        
        ordering = []
        remaining = set(vertices)
        weights = defaultdict(int)
        
        # Start with arbitrary vertex
        current = vertices[0]
        ordering.append(current)
        remaining.remove(current)
        
        # Update weights
        for neighbor in graph[current]:
            if neighbor in remaining:
                weights[neighbor] += graph[current][neighbor]
        
        # Add vertices in maximum adjacency order
        while remaining:
            # Find vertex with maximum weight
            max_weight = -1
            next_vertex = None
            
            for v in remaining:
                if weights[v] > max_weight:
                    max_weight = weights[v]
                    next_vertex = v
            
            if next_vertex is None:
                next_vertex = next(iter(remaining))
            
            ordering.append(next_vertex)
            remaining.remove(next_vertex)
            
            # Update weights
            for neighbor in graph[next_vertex]:
                if neighbor in remaining:
                    weights[neighbor] += graph[next_vertex][neighbor]
        
        return ordering
    
    def _contract_vertices_ni(self, graph: Dict[int, Dict[int, int]], s: int, t: int, 
                            vertex_sets: Dict[int, Set[int]]):
        """Contract vertices for Nagamochi-Ibaraki"""
        # Merge t into s
        for neighbor in graph[t]:
            if neighbor != s:
                graph[s][neighbor] = graph[s].get(neighbor, 0) + graph[t][neighbor]
                graph[neighbor][s] = graph[neighbor].get(s, 0) + graph[neighbor][t]
                del graph[neighbor][t]
        
        # Remove t
        del graph[t]
        if s in graph[s]:
            del graph[s][s]
        
        # Merge vertex sets
        vertex_sets[s].update(vertex_sets[t])
        del vertex_sets[t]
    
    def _max_flow_min_cut(self, graph: Dict[int, Dict[int, int]], source: int, sink: int) -> Tuple[int, Tuple[List[int], List[int]]]:
        """Find max flow and min cut between source and sink"""
        # Simple Edmonds-Karp implementation
        residual = defaultdict(lambda: defaultdict(int))
        
        for u in graph:
            for v in graph[u]:
                residual[u][v] = graph[u][v]
        
        max_flow = 0
        
        while True:
            # BFS to find augmenting path
            parent = {}
            queue = deque([source])
            parent[source] = None
            
            while queue and sink not in parent:
                current = queue.popleft()
                
                for neighbor in residual[current]:
                    if neighbor not in parent and residual[current][neighbor] > 0:
                        parent[neighbor] = current
                        queue.append(neighbor)
            
            if sink not in parent:
                break
            
            # Find bottleneck
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
        
        # Find min cut
        reachable = set()
        queue = deque([source])
        reachable.add(source)
        
        while queue:
            current = queue.popleft()
            for neighbor in residual[current]:
                if neighbor not in reachable and residual[current][neighbor] > 0:
                    reachable.add(neighbor)
                    queue.append(neighbor)
        
        source_side = list(reachable)
        sink_side = [v for v in graph.keys() if v not in reachable]
        
        return max_flow, (source_side, sink_side)

def test_global_min_cut_algorithms():
    """Test all global min-cut algorithms"""
    print("=== Testing Global Min-Cut Algorithms ===")
    
    # Test graphs
    test_graphs = [
        # Simple graph
        ({
            0: {1: 5, 2: 3},
            1: {0: 5, 2: 2, 3: 4},
            2: {0: 3, 1: 2, 3: 6},
            3: {1: 4, 2: 6}
        }, "Simple 4-vertex graph"),
        
        # Linear chain
        ({
            0: {1: 3},
            1: {0: 3, 2: 2},
            2: {1: 2, 3: 4},
            3: {2: 4}
        }, "Linear chain"),
        
        # Complete graph K4
        ({
            0: {1: 1, 2: 1, 3: 1},
            1: {0: 1, 2: 1, 3: 1},
            2: {0: 1, 1: 1, 3: 1},
            3: {0: 1, 1: 1, 2: 1}
        }, "Complete K4"),
    ]
    
    algorithms = [
        ("Stoer-Wagner", lambda g: GlobalMinCutAlgorithms().global_min_cut_stoer_wagner(g)),
        ("Karger Basic", lambda g: GlobalMinCutAlgorithms().global_min_cut_karger_basic(g, 50)),
        ("Nagamochi-Ibaraki", lambda g: GlobalMinCutAlgorithms().global_min_cut_nagamochi_ibaraki(g)),
        ("Gomory-Hu Tree", lambda g: GlobalMinCutAlgorithms().global_min_cut_gomory_hu_tree(g)),
    ]
    
    for graph, description in test_graphs:
        print(f"\n--- {description} ---")
        print(f"Vertices: {list(graph.keys())}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(graph)
                cut_value = result['cut_value']
                phases = result['statistics'].get('phases', 0)
                
                print(f"{alg_name:18} | Cut: {cut_value:2} | Phases: {phases:2}")
                
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:20]}")

def demonstrate_global_min_cut_theory():
    """Demonstrate global min-cut theory and applications"""
    print("\n=== Global Min-Cut Theory ===")
    
    print("Problem Definition:")
    print("• Find minimum cut in undirected graph")
    print("• No specified source and sink vertices")
    print("• Cut separates graph into two non-empty parts")
    print("• Minimize total weight of cut edges")
    
    print("\nKey Algorithms:")
    print("1. Stoer-Wagner: Deterministic O(V³) algorithm")
    print("2. Karger: Randomized O(V² log V) with high probability")
    print("3. Karger-Stein: Improved randomized O(V² log³ V)")
    print("4. Nagamochi-Ibaraki: Deterministic O(V² + VE)")
    print("5. Gomory-Hu Tree: All-pairs min-cut in O(V² max-flow)")
    
    print("\nApplications:")
    print("• Network reliability analysis")
    print("• VLSI circuit design")
    print("• Image segmentation")
    print("• Social network analysis")
    print("• Clustering and partitioning")

if __name__ == "__main__":
    test_global_min_cut_algorithms()
    demonstrate_global_min_cut_theory()

"""
Global Min-Cut Algorithms - Key Insights:

1. **Problem Characteristics:**
   - Find minimum cut without specified source/sink
   - Undirected graph with edge weights
   - Cut separates graph into two non-empty parts
   - Applications in network analysis and clustering

2. **Algorithm Categories:**
   - Deterministic: Stoer-Wagner, Nagamochi-Ibaraki
   - Randomized: Karger, Karger-Stein
   - Tree-based: Gomory-Hu tree construction
   - Approximation: Various heuristic approaches

3. **Key Techniques:**
   - Maximum adjacency ordering
   - Random edge contraction
   - Recursive divide-and-conquer
   - Cut tree construction
   - Flow-based reductions

4. **Complexity Analysis:**
   - Stoer-Wagner: O(V³) deterministic
   - Karger: O(V² log V) randomized
   - Nagamochi-Ibaraki: O(V² + VE)
   - Gomory-Hu: O(V² × max-flow time)

5. **Practical Considerations:**
   - Deterministic vs randomized trade-offs
   - Memory usage for large graphs
   - Parallelization opportunities
   - Approximation for very large instances

Global min-cut algorithms provide fundamental tools
for graph partitioning and network analysis problems.
"""
