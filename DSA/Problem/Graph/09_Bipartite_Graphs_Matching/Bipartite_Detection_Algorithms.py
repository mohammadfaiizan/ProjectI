"""
Bipartite Detection Algorithms - Comprehensive Implementation
Difficulty: Easy

This file provides comprehensive implementations of various bipartite detection algorithms,
including theoretical foundations, optimization techniques, and practical applications.

Key Algorithms:
1. BFS-based 2-Coloring
2. DFS-based 2-Coloring  
3. Union-Find with Complement Graph
4. Matrix-based Detection
5. Advanced Optimization Techniques
6. Parallel and Distributed Approaches
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict, deque
import random
import time

class BipartiteDetectionAlgorithms:
    """Comprehensive bipartite detection algorithm implementations"""
    
    def __init__(self):
        self.reset_statistics()
    
    def reset_statistics(self):
        """Reset performance statistics"""
        self.stats = {
            'nodes_visited': 0,
            'edges_traversed': 0,
            'components_found': 0,
            'conflicts_detected': 0
        }
    
    def detect_bipartite_bfs_standard(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 1: Standard BFS 2-Coloring
        
        Classic BFS approach with detailed analysis.
        
        Time: O(V + E)
        Space: O(V)
        """
        self.reset_statistics()
        
        vertices = set()
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
        
        if not vertices:
            return {'is_bipartite': True, 'partitions': [[], []], 'stats': self.stats}
        
        color = {}
        partition_A = []
        partition_B = []
        
        for start in vertices:
            if start not in color:
                self.stats['components_found'] += 1
                
                # BFS coloring
                queue = deque([start])
                color[start] = 0
                
                while queue:
                    node = queue.popleft()
                    self.stats['nodes_visited'] += 1
                    current_color = color[node]
                    
                    for neighbor in graph.get(node, []):
                        self.stats['edges_traversed'] += 1
                        
                        if neighbor not in color:
                            color[neighbor] = 1 - current_color
                            queue.append(neighbor)
                        elif color[neighbor] == current_color:
                            self.stats['conflicts_detected'] += 1
                            return {
                                'is_bipartite': False,
                                'conflict': (node, neighbor),
                                'stats': self.stats
                            }
        
        # Build partitions
        for vertex in vertices:
            if color.get(vertex, 0) == 0:
                partition_A.append(vertex)
            else:
                partition_B.append(vertex)
        
        return {
            'is_bipartite': True,
            'partitions': [sorted(partition_A), sorted(partition_B)],
            'coloring': color,
            'stats': self.stats
        }
    
    def detect_bipartite_dfs_recursive(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 2: Recursive DFS 2-Coloring
        
        Recursive DFS with conflict detection.
        
        Time: O(V + E)
        Space: O(V) for recursion stack
        """
        self.reset_statistics()
        
        vertices = set()
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
        
        color = {}
        conflict_info = {}
        
        def dfs(node, node_color):
            """Recursive DFS coloring"""
            self.stats['nodes_visited'] += 1
            color[node] = node_color
            
            for neighbor in graph.get(node, []):
                self.stats['edges_traversed'] += 1
                
                if neighbor not in color:
                    if not dfs(neighbor, 1 - node_color):
                        return False
                elif color[neighbor] == node_color:
                    self.stats['conflicts_detected'] += 1
                    conflict_info['conflict'] = (node, neighbor)
                    return False
            
            return True
        
        # Process each component
        for start in vertices:
            if start not in color:
                self.stats['components_found'] += 1
                if not dfs(start, 0):
                    return {
                        'is_bipartite': False,
                        'conflict': conflict_info.get('conflict'),
                        'stats': self.stats
                    }
        
        # Build result
        partition_A = [v for v in vertices if color.get(v, 0) == 0]
        partition_B = [v for v in vertices if color.get(v, 0) == 1]
        
        return {
            'is_bipartite': True,
            'partitions': [sorted(partition_A), sorted(partition_B)],
            'coloring': color,
            'stats': self.stats
        }
    
    def detect_bipartite_union_find(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 3: Union-Find with Complement Graph
        
        Use Union-Find to detect bipartiteness through complement relationships.
        
        Time: O(E * α(V))
        Space: O(V)
        """
        self.reset_statistics()
        
        class UnionFind:
            def __init__(self, vertices):
                self.parent = {}
                self.rank = {}
                
                for v in vertices:
                    self.parent[v] = v
                    self.parent[v + 'complement'] = v + 'complement'
                    self.rank[v] = 0
                    self.rank[v + 'complement'] = 0
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return
                
                if self.rank[px] < self.rank[py]:
                    px, py = py, px
                
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
            
            def connected(self, x, y):
                return self.find(x) == self.find(y)
        
        vertices = set()
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
        
        if not vertices:
            return {'is_bipartite': True, 'partitions': [[], []], 'stats': self.stats}
        
        uf = UnionFind(vertices)
        
        # Process edges
        for u in graph:
            for v in graph[u]:
                self.stats['edges_traversed'] += 1
                
                # Check if u and v are already in same partition
                if uf.connected(u, v):
                    self.stats['conflicts_detected'] += 1
                    return {
                        'is_bipartite': False,
                        'conflict': (u, v),
                        'stats': self.stats
                    }
                
                # Union u with v's complement, v with u's complement
                uf.union(u, str(v) + 'complement')
                uf.union(v, str(u) + 'complement')
        
        # Build partitions based on Union-Find structure
        partition_map = {}
        for v in vertices:
            root_v = uf.find(v)
            root_complement = uf.find(str(v) + 'complement')
            
            # Assign to partition based on root comparison
            if root_v < root_complement:
                partition_map[v] = 0
            else:
                partition_map[v] = 1
        
        partition_A = [v for v in vertices if partition_map[v] == 0]
        partition_B = [v for v in vertices if partition_map[v] == 1]
        
        self.stats['components_found'] = len(set(uf.find(v) for v in vertices))
        
        return {
            'is_bipartite': True,
            'partitions': [sorted(partition_A), sorted(partition_B)],
            'coloring': partition_map,
            'stats': self.stats
        }
    
    def detect_bipartite_matrix_based(self, adjacency_matrix: List[List[int]]) -> Dict:
        """
        Approach 4: Matrix-based Bipartite Detection
        
        Use adjacency matrix for bipartite detection.
        
        Time: O(V^2)
        Space: O(V^2)
        """
        self.reset_statistics()
        
        n = len(adjacency_matrix)
        if n == 0:
            return {'is_bipartite': True, 'partitions': [[], []], 'stats': self.stats}
        
        color = [-1] * n
        
        for start in range(n):
            if color[start] == -1:
                self.stats['components_found'] += 1
                
                # BFS using matrix
                queue = deque([start])
                color[start] = 0
                
                while queue:
                    node = queue.popleft()
                    self.stats['nodes_visited'] += 1
                    current_color = color[node]
                    
                    for neighbor in range(n):
                        if adjacency_matrix[node][neighbor] == 1:
                            self.stats['edges_traversed'] += 1
                            
                            if color[neighbor] == -1:
                                color[neighbor] = 1 - current_color
                                queue.append(neighbor)
                            elif color[neighbor] == current_color:
                                self.stats['conflicts_detected'] += 1
                                return {
                                    'is_bipartite': False,
                                    'conflict': (node, neighbor),
                                    'stats': self.stats
                                }
        
        # Build partitions
        partition_A = [i for i in range(n) if color[i] == 0]
        partition_B = [i for i in range(n) if color[i] == 1]
        
        return {
            'is_bipartite': True,
            'partitions': [partition_A, partition_B],
            'coloring': {i: color[i] for i in range(n)},
            'stats': self.stats
        }
    
    def detect_bipartite_optimized_bfs(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 5: Optimized BFS with Early Termination
        
        Enhanced BFS with various optimizations.
        
        Time: O(V + E)
        Space: O(V)
        """
        self.reset_statistics()
        
        vertices = set()
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
        
        if not vertices:
            return {'is_bipartite': True, 'partitions': [[], []], 'stats': self.stats}
        
        color = {}
        components = []
        
        # Sort vertices by degree (heuristic optimization)
        vertex_degrees = {v: len(graph.get(v, [])) for v in vertices}
        sorted_vertices = sorted(vertices, key=lambda x: vertex_degrees[x], reverse=True)
        
        for start in sorted_vertices:
            if start not in color:
                component_partition_A = []
                component_partition_B = []
                
                # Optimized BFS
                queue = deque([start])
                color[start] = 0
                component_partition_A.append(start)
                
                while queue:
                    node = queue.popleft()
                    self.stats['nodes_visited'] += 1
                    current_color = color[node]
                    
                    # Early termination check
                    if self.stats['conflicts_detected'] > 0:
                        break
                    
                    neighbors = graph.get(node, [])
                    # Sort neighbors for consistent processing
                    neighbors.sort()
                    
                    for neighbor in neighbors:
                        self.stats['edges_traversed'] += 1
                        
                        if neighbor not in color:
                            next_color = 1 - current_color
                            color[neighbor] = next_color
                            
                            if next_color == 0:
                                component_partition_A.append(neighbor)
                            else:
                                component_partition_B.append(neighbor)
                            
                            queue.append(neighbor)
                        elif color[neighbor] == current_color:
                            self.stats['conflicts_detected'] += 1
                            return {
                                'is_bipartite': False,
                                'conflict': (node, neighbor),
                                'component_analyzed': len(components),
                                'stats': self.stats
                            }
                
                components.append({
                    'partition_A': component_partition_A,
                    'partition_B': component_partition_B,
                    'size': len(component_partition_A) + len(component_partition_B)
                })
                self.stats['components_found'] += 1
        
        # Combine all partitions
        total_partition_A = []
        total_partition_B = []
        
        for component in components:
            total_partition_A.extend(component['partition_A'])
            total_partition_B.extend(component['partition_B'])
        
        return {
            'is_bipartite': True,
            'partitions': [sorted(total_partition_A), sorted(total_partition_B)],
            'components': components,
            'coloring': color,
            'stats': self.stats
        }
    
    def detect_bipartite_parallel_simulation(self, graph: Dict[int, List[int]], num_threads: int = 4) -> Dict:
        """
        Approach 6: Parallel Bipartite Detection Simulation
        
        Simulate parallel processing for large graphs.
        
        Time: O((V + E) / P) ideally
        Space: O(V + E)
        """
        self.reset_statistics()
        
        vertices = set()
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
        
        if not vertices:
            return {'is_bipartite': True, 'partitions': [[], []], 'stats': self.stats}
        
        # Simulate work distribution
        vertex_list = list(vertices)
        chunk_size = len(vertex_list) // num_threads
        
        thread_work = []
        global_color = {}
        
        # Simulate parallel component detection
        for thread_id in range(num_threads):
            start_idx = thread_id * chunk_size
            end_idx = start_idx + chunk_size if thread_id < num_threads - 1 else len(vertex_list)
            
            thread_vertices = vertex_list[start_idx:end_idx]
            local_components = []
            
            for vertex in thread_vertices:
                if vertex not in global_color:
                    # Simulate BFS for this component
                    component = []
                    queue = deque([vertex])
                    local_color = {vertex: 0}
                    
                    while queue:
                        node = queue.popleft()
                        component.append(node)
                        
                        for neighbor in graph.get(node, []):
                            if neighbor not in local_color:
                                local_color[neighbor] = 1 - local_color[node]
                                queue.append(neighbor)
                                
                                if neighbor in thread_vertices:  # Only if in this thread's range
                                    continue
                            elif local_color[neighbor] == local_color[node]:
                                return {
                                    'is_bipartite': False,
                                    'conflict': (node, neighbor),
                                    'thread_id': thread_id,
                                    'stats': self.stats
                                }
                    
                    local_components.append(component)
                    global_color.update(local_color)
            
            thread_work.append({
                'thread_id': thread_id,
                'vertices_processed': thread_vertices,
                'components_found': local_components
            })
        
        # Simulate coordination phase
        # In real implementation, threads would synchronize here
        
        # Build final partitions
        partition_A = [v for v in vertices if global_color.get(v, 0) == 0]
        partition_B = [v for v in vertices if global_color.get(v, 0) == 1]
        
        return {
            'is_bipartite': True,
            'partitions': [sorted(partition_A), sorted(partition_B)],
            'parallel_info': {
                'num_threads': num_threads,
                'thread_work': thread_work,
                'coordination_overhead': 'simulated'
            },
            'coloring': global_color,
            'stats': self.stats
        }

def test_bipartite_algorithms():
    """Test all bipartite detection algorithms"""
    detector = BipartiteDetectionAlgorithms()
    
    print("=== Testing Bipartite Detection Algorithms ===")
    
    test_graphs = [
        # Bipartite graphs
        {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]},  # Square
        {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]},        # Star
        {0: [2, 4], 1: [2, 4], 2: [0, 1], 4: [0, 1]},  # Complete bipartite
        
        # Non-bipartite graphs
        {0: [1, 2], 1: [0, 2], 2: [0, 1]},             # Triangle
        {0: [1, 4], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 0]},  # 5-cycle
        
        # Edge cases
        {},                                              # Empty
        {0: []},                                        # Single vertex
        {0: [1], 1: [0]},                              # Single edge
    ]
    
    algorithms = [
        ("Standard BFS", detector.detect_bipartite_bfs_standard),
        ("Recursive DFS", detector.detect_bipartite_dfs_recursive),
        ("Union-Find", detector.detect_bipartite_union_find),
        ("Optimized BFS", detector.detect_bipartite_optimized_bfs),
        ("Parallel Sim", detector.detect_bipartite_parallel_simulation),
    ]
    
    for i, graph in enumerate(test_graphs):
        print(f"\n--- Test Graph {i+1}: {graph} ---")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(graph)
                is_bipartite = result['is_bipartite']
                stats = result.get('stats', {})
                
                print(f"{alg_name:15} | Bipartite: {is_bipartite:5} | "
                      f"Nodes: {stats.get('nodes_visited', 0):3} | "
                      f"Edges: {stats.get('edges_traversed', 0):3}")
                
            except Exception as e:
                print(f"{alg_name:15} | Error: {str(e)[:30]}")

def demonstrate_algorithmic_differences():
    """Demonstrate differences between algorithms"""
    print("\n=== Algorithmic Differences Demo ===")
    
    detector = BipartiteDetectionAlgorithms()
    test_graph = {0: [1, 2, 3], 1: [0, 4], 2: [0, 5], 3: [0, 6], 4: [1], 5: [2], 6: [3]}
    
    print(f"Test graph: {test_graph}")
    print(f"Structure: Star with 3 arms, each arm has 2 nodes")
    
    # BFS approach
    print(f"\n1. BFS Approach:")
    result_bfs = detector.detect_bipartite_bfs_standard(test_graph)
    print(f"   Partitions: {result_bfs['partitions']}")
    print(f"   Components: {result_bfs['stats']['components_found']}")
    print(f"   Nodes visited: {result_bfs['stats']['nodes_visited']}")
    
    # DFS approach
    print(f"\n2. DFS Approach:")
    result_dfs = detector.detect_bipartite_dfs_recursive(test_graph)
    print(f"   Partitions: {result_dfs['partitions']}")
    print(f"   Components: {result_dfs['stats']['components_found']}")
    print(f"   Nodes visited: {result_dfs['stats']['nodes_visited']}")
    
    # Union-Find approach
    print(f"\n3. Union-Find Approach:")
    result_uf = detector.detect_bipartite_union_find(test_graph)
    print(f"   Partitions: {result_uf['partitions']}")
    print(f"   Edges processed: {result_uf['stats']['edges_traversed']}")

def analyze_performance_characteristics():
    """Analyze performance characteristics of different algorithms"""
    print("\n=== Performance Characteristics Analysis ===")
    
    print("Algorithm Performance Comparison:")
    
    print("\n1. **Time Complexity:**")
    print("   • BFS/DFS: O(V + E) - optimal for graph traversal")
    print("   • Union-Find: O(E × α(V)) - good for sparse graphs")
    print("   • Matrix-based: O(V²) - suitable for dense graphs")
    print("   • Parallel: O((V + E) / P) - for large-scale processing")
    
    print("\n2. **Space Complexity:**")
    print("   • BFS: O(V) queue + O(V) coloring")
    print("   • DFS: O(V) recursion + O(V) coloring")
    print("   • Union-Find: O(V) parent array")
    print("   • Matrix: O(V²) adjacency matrix")
    
    print("\n3. **Practical Considerations:**")
    print("   • **Small graphs (V < 100):** Any algorithm works")
    print("   • **Large sparse graphs:** BFS preferred")
    print("   • **Dense graphs:** Matrix-based efficient")
    print("   • **Dynamic graphs:** Union-Find advantageous")
    
    print("\n4. **Memory Access Patterns:**")
    print("   • BFS: Good cache locality with queue")
    print("   • DFS: Stack-friendly, potential stack overflow")
    print("   • Union-Find: Random access pattern")
    print("   • Matrix: Excellent cache locality")
    
    print("\n5. **Parallelization Potential:**")
    print("   • BFS: Moderate (level synchronization)")
    print("   • DFS: Limited (sequential nature)")
    print("   • Union-Find: Good (parallel union-find exists)")
    print("   • Matrix: Excellent (row/column parallelism)")

def demonstrate_optimization_techniques():
    """Demonstrate various optimization techniques"""
    print("\n=== Optimization Techniques Demo ===")
    
    print("Bipartite Detection Optimizations:")
    
    print("\n1. **Early Termination:**")
    print("   • Stop immediately on first conflict")
    print("   • No need to explore entire graph")
    print("   • Particularly effective for non-bipartite graphs")
    
    print("\n2. **Degree-based Ordering:**")
    print("   • Process high-degree vertices first")
    print("   • Higher chance of early conflict detection")
    print("   • Better component discovery")
    
    print("\n3. **Component Isolation:**")
    print("   • Process connected components independently")
    print("   • Parallel processing opportunities")
    print("   • Memory access optimization")
    
    print("\n4. **Cache Optimization:**")
    print("   • Locality-aware vertex processing")
    print("   • Compact data structures")
    print("   • Memory prefetching strategies")
    
    print("\n5. **Algorithm Selection:**")
    print("   • Choose algorithm based on graph properties")
    print("   • Sparse vs dense graph considerations")
    print("   • Dynamic vs static graph scenarios")

def demonstrate_real_world_applications():
    """Demonstrate real-world applications of bipartite detection"""
    print("\n=== Real-World Applications ===")
    
    print("Bipartite Detection Applications:")
    
    print("\n1. **Social Network Analysis:**")
    print("   • User-interest bipartite graphs")
    print("   • Community detection preparation")
    print("   • Recommendation system foundations")
    
    print("\n2. **Scheduling and Assignment:**")
    print("   • Task-resource assignment graphs")
    print("   • Course-student enrollment")
    print("   • Job-worker matching preparation")
    
    print("\n3. **Biological Networks:**")
    print("   • Gene-disease association networks")
    print("   • Protein interaction analysis")
    print("   • Drug-target interaction modeling")
    
    print("\n4. **Web and Information Systems:**")
    print("   • Web page categorization")
    print("   • Document-keyword relationships")
    print("   • Query-result bipartite structures")
    
    print("\n5. **Economic and Market Analysis:**")
    print("   • Buyer-seller market graphs")
    print("   • Product-consumer relationships")
    print("   • Supply chain bipartite modeling")

def demonstrate_matrix_conversion():
    """Demonstrate conversion between graph representations"""
    print("\n=== Graph Representation Conversion ===")
    
    detector = BipartiteDetectionAlgorithms()
    
    # Example graph
    adj_list = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}
    
    print(f"Adjacency List: {adj_list}")
    
    # Convert to matrix
    vertices = sorted(set(v for v in adj_list.keys()) | 
                     set(v for neighbors in adj_list.values() for v in neighbors))
    n = len(vertices)
    vertex_to_index = {v: i for i, v in enumerate(vertices)}
    
    adj_matrix = [[0] * n for _ in range(n)]
    for u in adj_list:
        for v in adj_list[u]:
            i, j = vertex_to_index[u], vertex_to_index[v]
            adj_matrix[i][j] = 1
    
    print(f"\nAdjacency Matrix:")
    for i, row in enumerate(adj_matrix):
        print(f"   {vertices[i]}: {row}")
    
    # Test both representations
    result_list = detector.detect_bipartite_bfs_standard(adj_list)
    result_matrix = detector.detect_bipartite_matrix_based(adj_matrix)
    
    print(f"\nList result: {result_list['is_bipartite']}")
    print(f"Matrix result: {result_matrix['is_bipartite']}")
    print(f"Results match: {result_list['is_bipartite'] == result_matrix['is_bipartite']}")

if __name__ == "__main__":
    test_bipartite_algorithms()
    demonstrate_algorithmic_differences()
    analyze_performance_characteristics()
    demonstrate_optimization_techniques()
    demonstrate_real_world_applications()
    demonstrate_matrix_conversion()

"""
Bipartite Detection Algorithms and Optimization Concepts:
1. Comprehensive Graph 2-Coloring Algorithm Implementations
2. Performance Analysis and Algorithm Selection Strategies
3. Optimization Techniques for Large-Scale Graph Processing
4. Parallel and Distributed Bipartite Detection Methods
5. Real-world Applications and Practical Considerations

Key Algorithmic Insights:
- Multiple approaches with different trade-offs
- BFS/DFS optimal for most scenarios
- Union-Find excellent for dynamic graphs
- Matrix-based efficient for dense graphs
- Parallel processing enables scalability

Performance Characteristics:
- Time complexity: O(V + E) for traversal-based methods
- Space complexity varies by approach and graph density
- Cache efficiency important for large graphs
- Early termination critical for non-bipartite detection

Optimization Strategies:
- Algorithm selection based on graph properties
- Early conflict detection and termination
- Component-wise processing for efficiency
- Memory access pattern optimization
- Parallel processing for scalability

Implementation Considerations:
- Robust error handling and edge cases
- Performance monitoring and statistics
- Flexible input format support
- Scalability for large datasets
- Integration with matching algorithms

Real-world Impact:
- Foundation for matching and assignment problems
- Essential for social network analysis
- Critical for scheduling and resource allocation
- Enables recommendation system development
- Supports biological network analysis

This comprehensive implementation provides production-ready
bipartite detection suitable for various application domains.
"""
