"""
Maximum Flow Basic - Comprehensive Algorithm Implementation
Difficulty: Easy

This file provides comprehensive implementations of fundamental maximum flow algorithms,
including Ford-Fulkerson, Edmonds-Karp, and other classical approaches for computing
maximum flow in networks.

Key Concepts:
1. Ford-Fulkerson Algorithm
2. Edmonds-Karp Algorithm (BFS-based)
3. DFS-based Flow Algorithms
4. Residual Graph Construction
5. Augmenting Path Finding
6. Flow Network Analysis
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import sys

class MaximumFlowBasic:
    """Comprehensive basic maximum flow algorithm implementations"""
    
    def __init__(self):
        self.reset_statistics()
    
    def reset_statistics(self):
        """Reset algorithm performance statistics"""
        self.stats = {
            'iterations': 0,
            'paths_found': 0,
            'total_flow_pushed': 0,
            'residual_graph_updates': 0,
            'bfs_calls': 0,
            'dfs_calls': 0
        }
    
    def max_flow_ford_fulkerson_dfs(self, graph: Dict[int, Dict[int, int]], source: int, sink: int) -> Dict:
        """
        Approach 1: Ford-Fulkerson with DFS
        
        Classic Ford-Fulkerson algorithm using DFS to find augmenting paths.
        
        Time: O(E * max_flow) - can be exponential
        Space: O(V + E)
        """
        self.reset_statistics()
        
        if source == sink:
            return {'max_flow': float('inf'), 'algorithm': 'ford_fulkerson_dfs'}
        
        # Build residual graph
        residual = defaultdict(lambda: defaultdict(int))
        
        for u in graph:
            for v in graph[u]:
                residual[u][v] = graph[u][v]
                residual[v][u] = 0  # Initialize reverse edge
        
        def dfs_find_path(current, target, visited, path, min_capacity):
            """DFS to find augmenting path"""
            self.stats['dfs_calls'] += 1
            
            if current == target:
                return min_capacity
            
            visited.add(current)
            
            for neighbor in residual[current]:
                if neighbor not in visited and residual[current][neighbor] > 0:
                    bottleneck = min(min_capacity, residual[current][neighbor])
                    path.append((current, neighbor))
                    
                    flow = dfs_find_path(neighbor, target, visited, path, bottleneck)
                    
                    if flow > 0:
                        return flow
                    
                    path.pop()
            
            return 0
        
        max_flow = 0
        
        while True:
            self.stats['iterations'] += 1
            visited = set()
            path = []
            
            # Find augmenting path using DFS
            flow = dfs_find_path(source, sink, visited, path, float('inf'))
            
            if flow == 0:
                break  # No more augmenting paths
            
            # Update residual graph along the path
            for u, v in path:
                residual[u][v] -= flow
                residual[v][u] += flow
                self.stats['residual_graph_updates'] += 1
            
            max_flow += flow
            self.stats['paths_found'] += 1
            self.stats['total_flow_pushed'] += flow
        
        return {
            'max_flow': max_flow,
            'algorithm': 'ford_fulkerson_dfs',
            'residual_graph': dict(residual),
            'statistics': self.stats.copy()
        }
    
    def max_flow_edmonds_karp(self, graph: Dict[int, Dict[int, int]], source: int, sink: int) -> Dict:
        """
        Approach 2: Edmonds-Karp Algorithm (BFS-based Ford-Fulkerson)
        
        Uses BFS to find shortest augmenting paths, guaranteeing polynomial time.
        
        Time: O(V * E^2)
        Space: O(V + E)
        """
        self.reset_statistics()
        
        if source == sink:
            return {'max_flow': float('inf'), 'algorithm': 'edmonds_karp'}
        
        # Build residual graph
        residual = defaultdict(lambda: defaultdict(int))
        
        for u in graph:
            for v in graph[u]:
                residual[u][v] = graph[u][v]
                residual[v][u] = 0
        
        def bfs_find_path():
            """BFS to find shortest augmenting path"""
            self.stats['bfs_calls'] += 1
            
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
        
        while True:
            self.stats['iterations'] += 1
            
            # Find augmenting path using BFS
            parent = bfs_find_path()
            
            if parent is None:
                break  # No more augmenting paths
            
            # Find minimum capacity along the path
            path_flow = float('inf')
            current = sink
            path = []
            
            while current != source:
                prev = parent[current]
                path.append((prev, current))
                path_flow = min(path_flow, residual[prev][current])
                current = prev
            
            # Update residual graph
            for u, v in path:
                residual[u][v] -= path_flow
                residual[v][u] += path_flow
                self.stats['residual_graph_updates'] += 1
            
            max_flow += path_flow
            self.stats['paths_found'] += 1
            self.stats['total_flow_pushed'] += path_flow
        
        return {
            'max_flow': max_flow,
            'algorithm': 'edmonds_karp',
            'residual_graph': dict(residual),
            'statistics': self.stats.copy()
        }
    
    def max_flow_capacity_scaling(self, graph: Dict[int, Dict[int, int]], source: int, sink: int) -> Dict:
        """
        Approach 3: Capacity Scaling Algorithm
        
        Improves efficiency by scaling capacities and finding high-capacity paths first.
        
        Time: O(E^2 * log(max_capacity))
        Space: O(V + E)
        """
        self.reset_statistics()
        
        if source == sink:
            return {'max_flow': float('inf'), 'algorithm': 'capacity_scaling'}
        
        # Build residual graph and find maximum capacity
        residual = defaultdict(lambda: defaultdict(int))
        max_capacity = 0
        
        for u in graph:
            for v in graph[u]:
                residual[u][v] = graph[u][v]
                residual[v][u] = 0
                max_capacity = max(max_capacity, graph[u][v])
        
        def bfs_find_path_with_capacity(min_capacity):
            """BFS to find path with minimum capacity threshold"""
            self.stats['bfs_calls'] += 1
            
            parent = {source: None}
            visited = {source}
            queue = deque([source])
            
            while queue:
                current = queue.popleft()
                
                for neighbor in residual[current]:
                    if (neighbor not in visited and 
                        residual[current][neighbor] >= min_capacity):
                        visited.add(neighbor)
                        parent[neighbor] = current
                        queue.append(neighbor)
                        
                        if neighbor == sink:
                            return parent
            
            return None
        
        max_flow = 0
        
        # Start with highest power of 2 <= max_capacity
        capacity_threshold = 1
        while capacity_threshold <= max_capacity:
            capacity_threshold *= 2
        capacity_threshold //= 2
        
        while capacity_threshold >= 1:
            self.stats['iterations'] += 1
            
            while True:
                # Find augmenting path with minimum capacity threshold
                parent = bfs_find_path_with_capacity(capacity_threshold)
                
                if parent is None:
                    break
                
                # Find actual flow along the path
                path_flow = float('inf')
                current = sink
                path = []
                
                while current != source:
                    prev = parent[current]
                    path.append((prev, current))
                    path_flow = min(path_flow, residual[prev][current])
                    current = prev
                
                # Update residual graph
                for u, v in path:
                    residual[u][v] -= path_flow
                    residual[v][u] += path_flow
                    self.stats['residual_graph_updates'] += 1
                
                max_flow += path_flow
                self.stats['paths_found'] += 1
                self.stats['total_flow_pushed'] += path_flow
            
            capacity_threshold //= 2
        
        return {
            'max_flow': max_flow,
            'algorithm': 'capacity_scaling',
            'residual_graph': dict(residual),
            'statistics': self.stats.copy()
        }
    
    def max_flow_shortest_augmenting_path(self, graph: Dict[int, Dict[int, int]], source: int, sink: int) -> Dict:
        """
        Approach 4: Shortest Augmenting Path Algorithm
        
        Always chooses the shortest augmenting path to improve efficiency.
        
        Time: O(V^2 * E)
        Space: O(V + E)
        """
        self.reset_statistics()
        
        if source == sink:
            return {'max_flow': float('inf'), 'algorithm': 'shortest_augmenting_path'}
        
        # Build residual graph
        residual = defaultdict(lambda: defaultdict(int))
        
        for u in graph:
            for v in graph[u]:
                residual[u][v] = graph[u][v]
                residual[v][u] = 0
        
        def bfs_shortest_path():
            """BFS to find shortest augmenting path"""
            self.stats['bfs_calls'] += 1
            
            distance = {source: 0}
            parent = {source: None}
            queue = deque([source])
            
            while queue:
                current = queue.popleft()
                
                for neighbor in residual[current]:
                    if (neighbor not in distance and 
                        residual[current][neighbor] > 0):
                        distance[neighbor] = distance[current] + 1
                        parent[neighbor] = current
                        queue.append(neighbor)
                        
                        if neighbor == sink:
                            return parent, distance[sink]
            
            return None, float('inf')
        
        max_flow = 0
        
        while True:
            self.stats['iterations'] += 1
            
            # Find shortest augmenting path
            parent, path_length = bfs_shortest_path()
            
            if parent is None:
                break
            
            # Find minimum capacity along the path
            path_flow = float('inf')
            current = sink
            path = []
            
            while current != source:
                prev = parent[current]
                path.append((prev, current))
                path_flow = min(path_flow, residual[prev][current])
                current = prev
            
            # Update residual graph
            for u, v in path:
                residual[u][v] -= path_flow
                residual[v][u] += path_flow
                self.stats['residual_graph_updates'] += 1
            
            max_flow += path_flow
            self.stats['paths_found'] += 1
            self.stats['total_flow_pushed'] += path_flow
        
        return {
            'max_flow': max_flow,
            'algorithm': 'shortest_augmenting_path',
            'path_length': path_length if parent else 0,
            'residual_graph': dict(residual),
            'statistics': self.stats.copy()
        }
    
    def max_flow_multiple_paths(self, graph: Dict[int, Dict[int, int]], source: int, sink: int) -> Dict:
        """
        Approach 5: Multiple Paths Algorithm
        
        Finds multiple augmenting paths simultaneously for better efficiency.
        
        Time: O(V * E^2)
        Space: O(V + E)
        """
        self.reset_statistics()
        
        if source == sink:
            return {'max_flow': float('inf'), 'algorithm': 'multiple_paths'}
        
        # Build residual graph
        residual = defaultdict(lambda: defaultdict(int))
        
        for u in graph:
            for v in graph[u]:
                residual[u][v] = graph[u][v]
                residual[v][u] = 0
        
        def find_multiple_paths():
            """Find multiple vertex-disjoint augmenting paths"""
            self.stats['bfs_calls'] += 1
            
            # First BFS to find shortest path and mark distances
            distance = {source: 0}
            queue = deque([source])
            
            while queue:
                current = queue.popleft()
                
                for neighbor in residual[current]:
                    if (neighbor not in distance and 
                        residual[current][neighbor] > 0):
                        distance[neighbor] = distance[current] + 1
                        queue.append(neighbor)
            
            if sink not in distance:
                return []
            
            # DFS to find all paths of shortest length
            paths = []
            
            def dfs_paths(current, path, capacity):
                if current == sink:
                    paths.append((path[:], capacity))
                    return
                
                for neighbor in residual[current]:
                    if (neighbor in distance and 
                        distance[neighbor] == distance[current] + 1 and
                        residual[current][neighbor] > 0 and
                        neighbor not in path):  # Vertex-disjoint
                        
                        new_capacity = min(capacity, residual[current][neighbor])
                        path.append(neighbor)
                        dfs_paths(neighbor, path, new_capacity)
                        path.pop()
            
            dfs_paths(source, [source], float('inf'))
            return paths[:3]  # Limit to first 3 paths for efficiency
        
        max_flow = 0
        
        while True:
            self.stats['iterations'] += 1
            
            # Find multiple augmenting paths
            paths = find_multiple_paths()
            
            if not paths:
                break
            
            # Process all paths
            total_iteration_flow = 0
            
            for path, capacity in paths:
                if capacity > 0:
                    # Update residual graph for this path
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        residual[u][v] -= capacity
                        residual[v][u] += capacity
                        self.stats['residual_graph_updates'] += 1
                    
                    total_iteration_flow += capacity
                    self.stats['paths_found'] += 1
                    self.stats['total_flow_pushed'] += capacity
            
            max_flow += total_iteration_flow
        
        return {
            'max_flow': max_flow,
            'algorithm': 'multiple_paths',
            'residual_graph': dict(residual),
            'statistics': self.stats.copy()
        }
    
    def max_flow_push_relabel_basic(self, graph: Dict[int, Dict[int, int]], source: int, sink: int) -> Dict:
        """
        Approach 6: Basic Push-Relabel Algorithm
        
        Uses preflow-push method with height labels for efficiency.
        
        Time: O(V^2 * E)
        Space: O(V + E)
        """
        self.reset_statistics()
        
        if source == sink:
            return {'max_flow': float('inf'), 'algorithm': 'push_relabel_basic'}
        
        # Get all vertices
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
        for v in graph.get(source, []):
            flow_amount = capacity[source][v]
            flow[source][v] = flow_amount
            flow[v][source] = -flow_amount
            excess[v] = flow_amount
            excess[source] -= flow_amount
        
        def push(u, v):
            """Push flow from u to v"""
            send = min(excess[u], capacity[u][v] - flow[u][v])
            flow[u][v] += send
            flow[v][u] -= send
            excess[u] -= send
            excess[v] += send
            self.stats['residual_graph_updates'] += 1
            return send
        
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
            """Find vertex with positive excess (excluding source and sink)"""
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
            
            # Try to push to eligible neighbors
            pushed = False
            for v in vertices:
                if (capacity[u][v] - flow[u][v] > 0 and 
                    height[u] == height[v] + 1):
                    push(u, v)
                    pushed = True
                    break
            
            # If no push possible, relabel
            if not pushed:
                relabel(u)
        
        max_flow = sum(flow[source][v] for v in vertices)
        
        return {
            'max_flow': max_flow,
            'algorithm': 'push_relabel_basic',
            'final_heights': dict(height),
            'final_excess': dict(excess),
            'statistics': self.stats.copy()
        }

def test_maximum_flow_algorithms():
    """Test all maximum flow algorithms"""
    solver = MaximumFlowBasic()
    
    print("=== Testing Maximum Flow Algorithms ===")
    
    # Test graphs
    test_graphs = [
        # Simple flow network
        ({
            0: {1: 10, 2: 8},
            1: {3: 5, 2: 2},
            2: {3: 10},
            3: {}
        }, 0, 3, 13, "Simple network"),
        
        # Classic flow network
        ({
            0: {1: 16, 2: 13},
            1: {2: 10, 3: 12},
            2: {1: 4, 4: 14},
            3: {2: 9, 5: 20},
            4: {3: 7, 5: 4},
            5: {}
        }, 0, 5, 23, "Classic network"),
        
        # Linear chain
        ({
            0: {1: 5},
            1: {2: 3},
            2: {3: 7},
            3: {}
        }, 0, 3, 3, "Linear chain"),
        
        # Parallel paths
        ({
            0: {1: 10, 2: 10},
            1: {3: 10},
            2: {3: 10},
            3: {}
        }, 0, 3, 20, "Parallel paths"),
        
        # Single edge
        ({
            0: {1: 100},
            1: {}
        }, 0, 1, 100, "Single edge"),
    ]
    
    algorithms = [
        ("Ford-Fulkerson DFS", solver.max_flow_ford_fulkerson_dfs),
        ("Edmonds-Karp", solver.max_flow_edmonds_karp),
        ("Capacity Scaling", solver.max_flow_capacity_scaling),
        ("Shortest Path", solver.max_flow_shortest_augmenting_path),
        ("Multiple Paths", solver.max_flow_multiple_paths),
        ("Push-Relabel", solver.max_flow_push_relabel_basic),
    ]
    
    for graph, source, sink, expected, name in test_graphs:
        print(f"\n--- {name} (Expected: {expected}) ---")
        print(f"Source: {source}, Sink: {sink}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(graph, source, sink)
                flow = result['max_flow']
                stats = result['statistics']
                
                correct = "✓" if flow == expected else "✗"
                iterations = stats.get('iterations', 0)
                paths = stats.get('paths_found', 0)
                
                print(f"{alg_name:18} | {correct} | Flow: {flow:3} | Iter: {iterations:2} | Paths: {paths:2}")
                
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

def demonstrate_flow_algorithm_steps():
    """Demonstrate step-by-step flow algorithm execution"""
    print("\n=== Flow Algorithm Steps Demo ===")
    
    graph = {
        0: {1: 10, 2: 8},
        1: {3: 5, 2: 2},
        2: {3: 10},
        3: {}
    }
    
    print(f"Graph: {graph}")
    print(f"Source: 0, Sink: 3")
    
    solver = MaximumFlowBasic()
    result = solver.max_flow_edmonds_karp(graph, 0, 3)
    
    print(f"\nEdmonds-Karp Algorithm:")
    print(f"Maximum Flow: {result['max_flow']}")
    print(f"Iterations: {result['statistics']['iterations']}")
    print(f"Augmenting paths found: {result['statistics']['paths_found']}")
    print(f"Total flow pushed: {result['statistics']['total_flow_pushed']}")
    
    print(f"\nFinal residual graph (showing remaining capacities):")
    residual = result['residual_graph']
    for u in sorted(residual.keys()):
        for v in sorted(residual[u].keys()):
            if residual[u][v] > 0:
                print(f"  Edge ({u} → {v}): {residual[u][v]}")

def demonstrate_algorithm_comparison():
    """Demonstrate comparison between different algorithms"""
    print("\n=== Algorithm Comparison Demo ===")
    
    graph = {
        0: {1: 16, 2: 13},
        1: {2: 10, 3: 12},
        2: {1: 4, 4: 14},
        3: {2: 9, 5: 20},
        4: {3: 7, 5: 4},
        5: {}
    }
    
    print(f"Classic flow network with 6 vertices")
    print(f"Source: 0, Sink: 5")
    
    solver = MaximumFlowBasic()
    
    algorithms = [
        ("Ford-Fulkerson DFS", solver.max_flow_ford_fulkerson_dfs),
        ("Edmonds-Karp", solver.max_flow_edmonds_karp),
        ("Capacity Scaling", solver.max_flow_capacity_scaling),
        ("Push-Relabel", solver.max_flow_push_relabel_basic),
    ]
    
    print(f"\nAlgorithm performance comparison:")
    print(f"{'Algorithm':<18} | {'Flow':<4} | {'Iter':<4} | {'Paths':<5} | {'Updates':<7}")
    print("-" * 65)
    
    for alg_name, alg_func in algorithms:
        try:
            result = alg_func(graph, 0, 5)
            flow = result['max_flow']
            stats = result['statistics']
            
            iterations = stats.get('iterations', 0)
            paths = stats.get('paths_found', 0)
            updates = stats.get('residual_graph_updates', 0)
            
            print(f"{alg_name:<18} | {flow:<4} | {iterations:<4} | {paths:<5} | {updates:<7}")
            
        except Exception as e:
            print(f"{alg_name:<18} | ERROR: {str(e)[:20]}")

def analyze_flow_theory():
    """Analyze maximum flow theory and applications"""
    print("\n=== Maximum Flow Theory ===")
    
    print("Maximum Flow Problem Theory:")
    
    print("\n1. **Fundamental Concepts:**")
    print("   • Flow network: directed graph with capacities")
    print("   • Source s: vertex with no incoming edges")
    print("   • Sink t: vertex with no outgoing edges")
    print("   • Flow: function satisfying capacity and conservation constraints")
    
    print("\n2. **Key Theorems:**")
    print("   • Max-Flow Min-Cut Theorem: max flow = min cut")
    print("   • Ford-Fulkerson theorem and algorithm correctness")
    print("   • Integral flow theorem: if capacities are integers, max flow is integer")
    print("   • Augmenting path theorem: max flow ⟺ no augmenting paths")
    
    print("\n3. **Algorithm Complexity:**")
    print("   • Ford-Fulkerson: O(E * f) where f is max flow value")
    print("   • Edmonds-Karp: O(V * E^2) - polynomial time guarantee")
    print("   • Capacity scaling: O(E^2 * log(max_capacity))")
    print("   • Push-relabel: O(V^2 * E) - advanced implementations O(V^3)")
    
    print("\n4. **Practical Considerations:**")
    print("   • Edmonds-Karp preferred for general use")
    print("   • Push-relabel better for dense graphs")
    print("   • Capacity scaling good for large capacity ranges")
    print("   • Preflow-push algorithms avoid augmenting paths")
    
    print("\n5. **Applications:**")
    print("   • Network routing and bandwidth allocation")
    print("   • Bipartite matching via flow networks")
    print("   • Image segmentation and computer vision")
    print("   • Transportation and logistics optimization")

def demonstrate_residual_graph_analysis():
    """Demonstrate residual graph construction and analysis"""
    print("\n=== Residual Graph Analysis ===")
    
    print("Residual Graph Concepts:")
    
    print("\n1. **Construction Rules:**")
    print("   • Forward edge (u,v): residual capacity = original capacity - current flow")
    print("   • Backward edge (v,u): residual capacity = current flow")
    print("   • Augmenting path: path from source to sink with positive residual capacity")
    print("   • Bottleneck: minimum residual capacity along augmenting path")
    
    print("\n2. **Flow Updates:**")
    print("   • Forward edges: decrease residual capacity by flow amount")
    print("   • Backward edges: increase residual capacity by flow amount")
    print("   • Conservation: total flow into vertex = total flow out (except s,t)")
    print("   • Optimality: no augmenting paths exist in final residual graph")
    
    print("\n3. **Cut Analysis:**")
    print("   • s-t cut: partition of vertices into sets S (containing s) and T (containing t)")
    print("   • Cut capacity: sum of capacities of edges from S to T")
    print("   • Min-cut: cut with minimum capacity")
    print("   • Max-flow = min-cut capacity (fundamental theorem)")
    
    # Example demonstration
    graph = {0: {1: 10, 2: 8}, 1: {3: 5}, 2: {3: 10}, 3: {}}
    
    print(f"\nExample: {graph}")
    
    solver = MaximumFlowBasic()
    result = solver.max_flow_edmonds_karp(graph, 0, 3)
    
    print(f"Maximum flow: {result['max_flow']}")
    print(f"This equals the minimum cut capacity of 13")
    print(f"Min-cut separates {{0}} from {{1,2,3}} or {{0,1,2}} from {{3}}")

if __name__ == "__main__":
    test_maximum_flow_algorithms()
    demonstrate_flow_algorithm_steps()
    demonstrate_algorithm_comparison()
    analyze_flow_theory()
    demonstrate_residual_graph_analysis()

"""
Maximum Flow and Network Flow Theory Concepts:
1. Classic Flow Algorithms: Ford-Fulkerson, Edmonds-Karp, Push-Relabel
2. Residual Graph Construction and Augmenting Path Finding
3. Flow Network Analysis and Min-Cut Max-Flow Theorem
4. Algorithm Optimization and Complexity Analysis
5. Real-world Applications in Network Design and Optimization

Key Algorithmic Insights:
- Ford-Fulkerson provides framework, Edmonds-Karp guarantees polynomial time
- Residual graph captures remaining flow potential
- BFS finds shortest augmenting paths for better performance
- Push-relabel avoids explicit path finding
- Multiple algorithmic approaches with different trade-offs

Algorithm Strategy:
1. Build residual graph from original network
2. Find augmenting paths using BFS (Edmonds-Karp) or DFS
3. Push maximum possible flow along each path
4. Update residual capacities for forward and backward edges
5. Repeat until no augmenting paths exist

Theoretical Foundations:
- Max-Flow Min-Cut theorem and network flow duality
- Augmenting path characterization of optimality
- Polynomial-time complexity guarantees
- Integral flow theorem for integer capacities
- Flow conservation and capacity constraints

Optimization Techniques:
- Shortest path selection for fewer iterations
- Capacity scaling for better worst-case bounds
- Push-relabel for dense graph efficiency
- Multiple path algorithms for parallelization

Real-world Applications:
- Network routing and traffic engineering
- Resource allocation and supply chain optimization
- Bipartite matching and assignment problems
- Image processing and computer vision
- Transportation and logistics planning

This comprehensive implementation provides fundamental
maximum flow algorithms essential for network optimization.
"""

