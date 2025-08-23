"""
Advanced Shortest Path Algorithms Collection
Difficulty: Hard

This file contains a comprehensive collection of advanced shortest path algorithms
and techniques that demonstrate cutting-edge concepts in graph theory and optimization.

Advanced Algorithms Included:
1. A* Search Algorithm
2. Johnson's All-Pairs Shortest Path
3. Bidirectional Dijkstra
4. Contraction Hierarchies
5. Dynamic Shortest Path
6. Parallel Shortest Path
7. Approximate Shortest Path
8. Multi-Objective Shortest Path
"""

from typing import List, Tuple, Dict, Set, Optional
import heapq
from collections import defaultdict, deque
import math

class AdvancedShortestPath:
    
    def a_star_search(self, graph: Dict[int, List[Tuple[int, float]]], start: int, goal: int, 
                     heuristic: Dict[int, float]) -> Tuple[float, List[int]]:
        """
        A* Search Algorithm
        
        Heuristic-guided shortest path algorithm for optimal pathfinding.
        
        Time: O(E log V) with admissible heuristic
        Space: O(V)
        """
        # Priority queue: (f_score, g_score, node, path)
        open_set = [(heuristic[start], 0, start, [start])]
        closed_set = set()
        g_score = {start: 0}
        
        while open_set:
            f, g, current, path = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            if current == goal:
                return g, path
            
            closed_set.add(current)
            
            for neighbor, weight in graph.get(current, []):
                if neighbor in closed_set:
                    continue
                
                tentative_g = g + weight
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic.get(neighbor, 0)
                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, new_path))
        
        return float('inf'), []
    
    def johnson_all_pairs(self, n: int, edges: List[Tuple[int, int, float]]) -> List[List[float]]:
        """
        Johnson's All-Pairs Shortest Path Algorithm
        
        Efficient all-pairs shortest path for sparse graphs with negative edges.
        
        Time: O(V^2 log V + VE)
        Space: O(V^2)
        """
        INF = float('inf')
        
        # Step 1: Add artificial vertex and run Bellman-Ford
        # Create modified graph with artificial vertex 0
        modified_edges = edges[:]
        for i in range(1, n + 1):
            modified_edges.append((0, i, 0))
        
        # Bellman-Ford from artificial vertex
        distances = [INF] * (n + 1)
        distances[0] = 0
        
        # Relax edges V times
        for _ in range(n):
            for u, v, w in modified_edges:
                if distances[u] != INF and distances[u] + w < distances[v]:
                    distances[v] = distances[u] + w
        
        # Check for negative cycles
        for u, v, w in modified_edges:
            if distances[u] != INF and distances[u] + w < distances[v]:
                raise ValueError("Graph contains negative cycle")
        
        # Step 2: Reweight edges using potentials
        h = distances[1:]  # Potentials (exclude artificial vertex)
        
        # Build adjacency list with reweighted edges
        graph = defaultdict(list)
        for u, v, w in edges:
            new_weight = w + h[u - 1] - h[v - 1]
            graph[u].append((v, new_weight))
        
        # Step 3: Run Dijkstra from each vertex
        result = [[INF] * n for _ in range(n)]
        
        for start in range(1, n + 1):
            # Dijkstra from start
            dist = [INF] * (n + 1)
            dist[start] = 0
            pq = [(0, start)]
            
            while pq:
                d, u = heapq.heappop(pq)
                
                if d > dist[u]:
                    continue
                
                for v, w in graph[u]:
                    if dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w
                        heapq.heappush(pq, (dist[v], v))
            
            # Restore original distances
            for end in range(1, n + 1):
                if dist[end] != INF:
                    result[start - 1][end - 1] = dist[end] - h[start - 1] + h[end - 1]
        
        return result
    
    def bidirectional_dijkstra(self, graph: Dict[int, List[Tuple[int, float]]], 
                              start: int, goal: int) -> float:
        """
        Bidirectional Dijkstra Algorithm
        
        Search from both start and goal simultaneously to reduce search space.
        
        Time: O(E log V) with potential for significant speedup
        Space: O(V)
        """
        # Build reverse graph
        reverse_graph = defaultdict(list)
        for u in graph:
            for v, w in graph[u]:
                reverse_graph[v].append((u, w))
        
        # Forward search from start
        forward_dist = {start: 0}
        forward_pq = [(0, start)]
        forward_visited = set()
        
        # Backward search from goal
        backward_dist = {goal: 0}
        backward_pq = [(0, goal)]
        backward_visited = set()
        
        best_distance = float('inf')
        
        while forward_pq or backward_pq:
            # Forward step
            if forward_pq:
                f_dist, f_node = heapq.heappop(forward_pq)
                
                if f_node not in forward_visited:
                    forward_visited.add(f_node)
                    
                    # Check if met backward search
                    if f_node in backward_visited:
                        best_distance = min(best_distance, f_dist + backward_dist[f_node])
                    
                    # Expand forward
                    for neighbor, weight in graph.get(f_node, []):
                        new_dist = f_dist + weight
                        if neighbor not in forward_dist or new_dist < forward_dist[neighbor]:
                            forward_dist[neighbor] = new_dist
                            heapq.heappush(forward_pq, (new_dist, neighbor))
            
            # Backward step
            if backward_pq:
                b_dist, b_node = heapq.heappop(backward_pq)
                
                if b_node not in backward_visited:
                    backward_visited.add(b_node)
                    
                    # Check if met forward search
                    if b_node in forward_visited:
                        best_distance = min(best_distance, b_dist + forward_dist[b_node])
                    
                    # Expand backward
                    for neighbor, weight in reverse_graph.get(b_node, []):
                        new_dist = b_dist + weight
                        if neighbor not in backward_dist or new_dist < backward_dist[neighbor]:
                            backward_dist[neighbor] = new_dist
                            heapq.heappush(backward_pq, (new_dist, neighbor))
            
            # Early termination check
            min_forward = min(forward_pq)[0] if forward_pq else float('inf')
            min_backward = min(backward_pq)[0] if backward_pq else float('inf')
            
            if best_distance <= min_forward + min_backward:
                break
        
        return best_distance
    
    def contraction_hierarchies_preprocess(self, n: int, edges: List[Tuple[int, int, float]]) -> Dict:
        """
        Contraction Hierarchies Preprocessing
        
        Preprocess graph for fast shortest path queries.
        
        Time: O(V log V + E) preprocessing
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        # Node ordering based on importance (simplified heuristic)
        def node_importance(node):
            """Calculate node importance for contraction order"""
            degree = len(graph[node])
            shortcuts_needed = 0
            
            # Estimate shortcuts needed if this node is contracted
            neighbors = [v for v, w in graph[node]]
            for i, u in enumerate(neighbors):
                for j, v in enumerate(neighbors):
                    if i != j:
                        shortcuts_needed += 1
            
            return degree + shortcuts_needed
        
        # Order nodes by importance
        node_order = sorted(range(n), key=node_importance)
        
        # Contract nodes in order
        contracted = set()
        shortcuts = []
        
        for node in node_order:
            neighbors = [(v, w) for v, w in graph[node] if v not in contracted]
            
            # Add shortcuts for all pairs of neighbors
            for i, (u, w1) in enumerate(neighbors):
                for j, (v, w2) in enumerate(neighbors):
                    if i != j:
                        # Check if direct path u->v is longer than u->node->v
                        direct_dist = self._shortest_path_between(graph, u, v, contracted | {node})
                        shortcut_dist = w1 + w2
                        
                        if shortcut_dist < direct_dist:
                            shortcuts.append((u, v, shortcut_dist, node))
            
            contracted.add(node)
        
        return {
            'node_order': node_order,
            'shortcuts': shortcuts,
            'original_graph': graph
        }
    
    def _shortest_path_between(self, graph: Dict, start: int, end: int, 
                              forbidden: Set[int]) -> float:
        """Helper function for contraction hierarchies"""
        if start == end:
            return 0
        
        distances = {start: 0}
        pq = [(0, start)]
        
        while pq:
            dist, node = heapq.heappop(pq)
            
            if node == end:
                return dist
            
            if dist > distances.get(node, float('inf')):
                continue
            
            for neighbor, weight in graph.get(node, []):
                if neighbor in forbidden:
                    continue
                
                new_dist = dist + weight
                if new_dist < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
        
        return float('inf')
    
    def dynamic_shortest_path_decremental(self, graph: Dict[int, List[Tuple[int, float]]], 
                                        start: int, removed_edges: List[Tuple[int, int]]) -> Dict[int, float]:
        """
        Dynamic Shortest Path - Decremental
        
        Maintain shortest paths when edges are removed from graph.
        
        Time: O(V log V + E) per edge removal (worst case)
        Space: O(V + E)
        """
        # Initial shortest path tree
        distances = self._dijkstra_single_source(graph, start)
        
        # Remove edges and update affected paths
        modified_graph = self._copy_graph(graph)
        
        for u, v in removed_edges:
            # Remove edge from graph
            modified_graph[u] = [(neighbor, weight) for neighbor, weight in modified_graph[u] 
                               if neighbor != v]
            modified_graph[v] = [(neighbor, weight) for neighbor, weight in modified_graph[v] 
                               if neighbor != u]
            
            # Check if removed edge affects shortest paths
            affected_nodes = set()
            
            # If edge (u,v) was in shortest path tree, need to recompute
            if distances.get(v, float('inf')) == distances.get(u, float('inf')) + self._edge_weight(graph, u, v):
                affected_nodes.add(v)
            if distances.get(u, float('inf')) == distances.get(v, float('inf')) + self._edge_weight(graph, v, u):
                affected_nodes.add(u)
            
            # Recompute distances from affected nodes
            if affected_nodes:
                new_distances = self._dijkstra_single_source(modified_graph, start)
                distances = new_distances
        
        return distances
    
    def _dijkstra_single_source(self, graph: Dict[int, List[Tuple[int, float]]], start: int) -> Dict[int, float]:
        """Helper: Standard Dijkstra implementation"""
        distances = {start: 0}
        pq = [(0, start)]
        
        while pq:
            dist, node = heapq.heappop(pq)
            
            if dist > distances.get(node, float('inf')):
                continue
            
            for neighbor, weight in graph.get(node, []):
                new_dist = dist + weight
                if new_dist < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
        
        return distances
    
    def _copy_graph(self, graph: Dict[int, List[Tuple[int, float]]]) -> Dict[int, List[Tuple[int, float]]]:
        """Helper: Deep copy graph"""
        return {u: list(edges) for u, edges in graph.items()}
    
    def _edge_weight(self, graph: Dict[int, List[Tuple[int, float]]], u: int, v: int) -> float:
        """Helper: Get edge weight"""
        for neighbor, weight in graph.get(u, []):
            if neighbor == v:
                return weight
        return float('inf')
    
    def parallel_delta_stepping(self, graph: Dict[int, List[Tuple[int, float]]], 
                               start: int, delta: float) -> Dict[int, float]:
        """
        Parallel Delta-Stepping Algorithm
        
        Parallel shortest path algorithm for distributed computing.
        
        Time: O(V + E + D) where D is diameter
        Space: O(V + E)
        """
        distances = {start: 0}
        buckets = defaultdict(list)  # bucket[i] contains nodes with distance in [i*delta, (i+1)*delta)
        buckets[0].append(start)
        
        current_bucket = 0
        
        while True:
            # Find next non-empty bucket
            while current_bucket not in buckets or not buckets[current_bucket]:
                current_bucket += 1
                if current_bucket > len(graph) * max(w for edges in graph.values() for _, w in edges) / delta:
                    break
            
            if current_bucket not in buckets or not buckets[current_bucket]:
                break
            
            # Process current bucket
            current_nodes = buckets[current_bucket][:]
            buckets[current_bucket].clear()
            
            # Light edges (weight <= delta)
            light_updates = []
            heavy_updates = []
            
            for node in current_nodes:
                for neighbor, weight in graph.get(node, []):
                    new_dist = distances[node] + weight
                    
                    if new_dist < distances.get(neighbor, float('inf')):
                        if weight <= delta:
                            light_updates.append((neighbor, new_dist))
                        else:
                            heavy_updates.append((neighbor, new_dist))
            
            # Apply light edge updates (can be done in parallel)
            for neighbor, new_dist in light_updates:
                if new_dist < distances.get(neighbor, float('inf')):
                    old_bucket = int(distances.get(neighbor, float('inf')) // delta)
                    new_bucket = int(new_dist // delta)
                    
                    distances[neighbor] = new_dist
                    
                    # Remove from old bucket and add to new bucket
                    if neighbor in buckets[old_bucket]:
                        buckets[old_bucket].remove(neighbor)
                    buckets[new_bucket].append(neighbor)
            
            # Apply heavy edge updates
            for neighbor, new_dist in heavy_updates:
                if new_dist < distances.get(neighbor, float('inf')):
                    new_bucket = int(new_dist // delta)
                    distances[neighbor] = new_dist
                    buckets[new_bucket].append(neighbor)
        
        return distances
    
    def approximate_shortest_path(self, graph: Dict[int, List[Tuple[int, float]]], 
                                start: int, epsilon: float) -> Dict[int, float]:
        """
        (1 + ε)-Approximate Shortest Path
        
        Trade accuracy for speed in shortest path computation.
        
        Time: O(E + V log log C) where C is max edge weight
        Space: O(V)
        """
        if epsilon <= 0:
            return self._dijkstra_single_source(graph, start)
        
        # Scale factor for rounding
        max_weight = max(weight for edges in graph.values() for _, weight in edges)
        scale = epsilon / (2 * len(graph))
        
        # Round edge weights
        rounded_graph = {}
        for u in graph:
            rounded_graph[u] = []
            for v, weight in graph[u]:
                rounded_weight = math.floor(weight / scale)
                rounded_graph[u].append((v, rounded_weight))
        
        # Run Dijkstra on rounded graph
        rounded_distances = self._dijkstra_single_source(rounded_graph, start)
        
        # Scale back distances
        approximate_distances = {}
        for node, dist in rounded_distances.items():
            approximate_distances[node] = dist * scale
        
        return approximate_distances
    
    def multi_objective_shortest_path(self, graph: Dict[int, List[Tuple[int, Tuple[float, float]]]], 
                                    start: int, end: int) -> List[Tuple[float, float, List[int]]]:
        """
        Multi-Objective Shortest Path (Pareto Optimal)
        
        Find Pareto optimal paths considering multiple objectives.
        
        Time: O(E * |Pareto|) where |Pareto| is number of Pareto optimal solutions
        Space: O(V * |Pareto|)
        """
        # Each node can have multiple Pareto optimal labels
        pareto_labels = defaultdict(list)  # node -> list of (cost1, cost2, path)
        pareto_labels[start] = [(0, 0, [start])]
        
        # Priority queue with labels
        pq = [(0, 0, start, [start])]  # (cost1, cost2, node, path)
        
        pareto_optimal_paths = []
        
        while pq:
            cost1, cost2, node, path = heapq.heappop(pq)
            
            # Check if this label is still Pareto optimal
            if not self._is_pareto_optimal((cost1, cost2), pareto_labels[node]):
                continue
            
            if node == end:
                pareto_optimal_paths.append((cost1, cost2, path))
                continue
            
            # Expand neighbors
            for neighbor, (edge_cost1, edge_cost2) in graph.get(node, []):
                new_cost1 = cost1 + edge_cost1
                new_cost2 = cost2 + edge_cost2
                new_path = path + [neighbor]
                
                # Check if new label is Pareto optimal
                if self._is_pareto_optimal((new_cost1, new_cost2), pareto_labels[neighbor]):
                    # Remove dominated labels
                    pareto_labels[neighbor] = [
                        label for label in pareto_labels[neighbor]
                        if not self._dominates((new_cost1, new_cost2), label[:2])
                    ]
                    
                    pareto_labels[neighbor].append((new_cost1, new_cost2, new_path))
                    heapq.heappush(pq, (new_cost1, new_cost2, neighbor, new_path))
        
        return pareto_optimal_paths
    
    def _is_pareto_optimal(self, new_costs: Tuple[float, float], 
                          existing_labels: List[Tuple[float, float, List[int]]]) -> bool:
        """Check if new costs are Pareto optimal"""
        for cost1, cost2, _ in existing_labels:
            if cost1 <= new_costs[0] and cost2 <= new_costs[1]:
                if cost1 < new_costs[0] or cost2 < new_costs[1]:
                    return False
        return True
    
    def _dominates(self, costs1: Tuple[float, float], costs2: Tuple[float, float]) -> bool:
        """Check if costs1 dominates costs2"""
        return (costs1[0] <= costs2[0] and costs1[1] <= costs2[1] and 
                (costs1[0] < costs2[0] or costs1[1] < costs2[1]))

def test_advanced_algorithms():
    """Test advanced shortest path algorithms"""
    solution = AdvancedShortestPath()
    
    print("=== Testing Advanced Shortest Path Algorithms ===")
    
    # Test graph
    graph = {
        1: [(2, 4), (3, 2)],
        2: [(3, 1), (4, 5)],
        3: [(4, 8), (5, 10)],
        4: [(5, 2)],
        5: []
    }
    
    # Test A* (need heuristic)
    heuristic = {1: 7, 2: 6, 3: 2, 4: 1, 5: 0}
    distance, path = solution.a_star_search(graph, 1, 5, heuristic)
    print(f"A* Search: Distance = {distance}, Path = {path}")
    
    # Test Bidirectional Dijkstra
    distance = solution.bidirectional_dijkstra(graph, 1, 5)
    print(f"Bidirectional Dijkstra: Distance = {distance}")
    
    # Test parallel delta-stepping
    distances = solution.parallel_delta_stepping(graph, 1, 2.0)
    print(f"Parallel Delta-Stepping: {distances}")
    
    # Test approximate shortest path
    approx_distances = solution.approximate_shortest_path(graph, 1, 0.1)
    print(f"Approximate Shortest Path: {approx_distances}")

def demonstrate_advanced_concepts():
    """Demonstrate advanced shortest path concepts"""
    print("\n=== Advanced Shortest Path Concepts ===")
    
    print("1. **Heuristic Search (A*):**")
    print("   • Uses heuristic to guide search toward goal")
    print("   • Admissible heuristic guarantees optimality")
    print("   • Effective for pathfinding in games and robotics")
    
    print("\n2. **Bidirectional Search:**")
    print("   • Search from both start and goal simultaneously")
    print("   • Reduces search space significantly")
    print("   • Meet-in-the-middle approach")
    
    print("\n3. **Hierarchical Methods:**")
    print("   • Preprocess graph for fast queries")
    print("   • Contraction hierarchies, hub labeling")
    print("   • Trade preprocessing time for query speed")
    
    print("\n4. **Parallel Algorithms:**")
    print("   • Delta-stepping for shared memory")
    print("   • Distributed shortest paths")
    print("   • GPGPU implementations")
    
    print("\n5. **Dynamic Algorithms:**")
    print("   • Maintain shortest paths as graph changes")
    print("   • Incremental and decremental updates")
    print("   • Online algorithms for streaming graphs")
    
    print("\n6. **Approximation Algorithms:**")
    print("   • Trade accuracy for speed")
    print("   • (1+ε)-approximation schemes")
    print("   • Useful for large-scale graphs")

def analyze_algorithm_selection():
    """Analyze when to use different advanced algorithms"""
    print("\n=== Algorithm Selection Guide ===")
    
    print("Choose algorithm based on:")
    
    print("\n**Graph Characteristics:**")
    print("• **Size:** V, E counts")
    print("• **Density:** Sparse vs dense")
    print("• **Weight distribution:** Uniform vs varied")
    print("• **Structure:** Planar, hierarchical, random")
    
    print("\n**Query Patterns:**")
    print("• **Single query:** Standard Dijkstra, A*")
    print("• **Many queries:** Preprocessing worthwhile")
    print("• **All-pairs:** Floyd-Warshall, Johnson's")
    print("• **Dynamic:** Incremental algorithms")
    
    print("\n**Performance Requirements:**")
    print("• **Real-time:** A*, approximate algorithms")
    print("• **Batch processing:** Parallel algorithms")
    print("• **Memory limited:** Space-efficient variants")
    print("• **Distributed:** Parallel/distributed algorithms")
    
    print("\n**Special Requirements:**")
    print("• **Multiple objectives:** Pareto optimal algorithms")
    print("• **Negative weights:** Bellman-Ford, Johnson's")
    print("• **Time-dependent:** Specialized algorithms")
    print("• **Uncertain weights:** Robust optimization")

def compare_preprocessing_strategies():
    """Compare different preprocessing strategies"""
    print("\n=== Preprocessing Strategies Comparison ===")
    
    print("1. **Contraction Hierarchies:**")
    print("   • Preprocessing: O(V log V + E)")
    print("   • Query: O(log V)")
    print("   • Space: O(V + E)")
    print("   • Best for: Road networks, hierarchical graphs")
    
    print("\n2. **Hub Labeling:**")
    print("   • Preprocessing: O(V^2)")
    print("   • Query: O(1) average")
    print("   • Space: O(V * h) where h = average hub set size")
    print("   • Best for: Small-world networks")
    
    print("\n3. **Distance Oracles:**")
    print("   • Preprocessing: O(V^2)")
    print("   • Query: O(1)")
    print("   • Space: O(V^2)")
    print("   • Best for: When space is not a constraint")
    
    print("\n4. **Landmark-based:**")
    print("   • Preprocessing: O(k * (V log V + E))")
    print("   • Query: O(k)")
    print("   • Space: O(k * V)")
    print("   • Best for: Large graphs with good landmarks")

if __name__ == "__main__":
    test_advanced_algorithms()
    demonstrate_advanced_concepts()
    analyze_algorithm_selection()
    compare_preprocessing_strategies()

"""
Advanced Shortest Path Concepts:
1. Heuristic-Guided Search (A*)
2. All-Pairs Algorithms (Johnson's)
3. Bidirectional and Meet-in-Middle
4. Preprocessing and Query Optimization
5. Parallel and Distributed Algorithms
6. Dynamic and Online Algorithms
7. Approximation and Trade-offs
8. Multi-Objective Optimization

Key Advanced Techniques:
- A* with admissible heuristics for goal-directed search
- Johnson's algorithm for all-pairs with negative weights
- Bidirectional search for reduced search space
- Contraction hierarchies for preprocessing optimization
- Delta-stepping for parallel computation
- Dynamic algorithms for changing graphs
- Approximation schemes for large-scale problems
- Pareto optimal solutions for multiple objectives

Performance Optimizations:
- Preprocessing strategies for repeated queries
- Parallel algorithms for multi-core systems
- Hierarchical decomposition for complex graphs
- Approximation algorithms for real-time requirements
- Cache-efficient implementations for large graphs

Algorithm Selection Criteria:
- Graph size and structure characteristics
- Query frequency and patterns
- Performance and memory constraints
- Special requirements (real-time, distributed, etc.)
- Trade-offs between preprocessing and query time

Real-world Applications:
- GPS navigation with hierarchical road networks
- Game AI with A* pathfinding
- Social network analysis with parallel algorithms
- Transportation optimization with multi-objective criteria
- Network routing with dynamic updates

This collection demonstrates state-of-the-art
shortest path algorithms for advanced applications.
"""
