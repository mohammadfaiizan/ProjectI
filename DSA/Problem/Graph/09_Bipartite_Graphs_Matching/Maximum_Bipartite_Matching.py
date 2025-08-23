"""
Maximum Bipartite Matching - Comprehensive Algorithm Implementation
Difficulty: Medium

This file provides comprehensive implementations of maximum bipartite matching algorithms,
including classical approaches, optimizations, and theoretical foundations.

Key Algorithms:
1. Ford-Fulkerson based approaches
2. Hopcroft-Karp algorithm
3. Hungarian algorithm for weighted matching
4. Push-relabel algorithms
5. Approximation algorithms
6. Online and dynamic matching algorithms
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict, deque
import heapq
import random

class MaximumBipartiteMatching:
    """Comprehensive maximum bipartite matching algorithm implementations"""
    
    def __init__(self):
        self.reset_statistics()
    
    def reset_statistics(self):
        """Reset algorithm performance statistics"""
        self.stats = {
            'augmenting_paths_found': 0,
            'dfs_calls': 0,
            'bfs_calls': 0,
            'iterations': 0,
            'edges_explored': 0
        }
    
    def maximum_matching_ford_fulkerson_dfs(self, left_vertices: int, right_vertices: int, edges: List[Tuple[int, int]]) -> Dict:
        """
        Approach 1: Ford-Fulkerson with DFS (Classic Implementation)
        
        Use DFS to find augmenting paths for maximum bipartite matching.
        
        Time: O(V * E)
        Space: O(V + E)
        """
        self.reset_statistics()
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
        
        # Matching arrays
        match_left = [-1] * left_vertices
        match_right = [-1] * right_vertices
        
        def dfs_augmenting_path(u, visited):
            """Find augmenting path using DFS"""
            self.stats['dfs_calls'] += 1
            
            for v in graph[u]:
                self.stats['edges_explored'] += 1
                
                if v in visited:
                    continue
                
                visited.add(v)
                
                # If v is unmatched or we can find augmenting path from match_right[v]
                if match_right[v] == -1 or dfs_augmenting_path(match_right[v], visited):
                    match_left[u] = v
                    match_right[v] = u
                    return True
            
            return False
        
        matching_size = 0
        
        # Try to find augmenting path for each left vertex
        for u in range(left_vertices):
            visited = set()
            if dfs_augmenting_path(u, visited):
                matching_size += 1
                self.stats['augmenting_paths_found'] += 1
            self.stats['iterations'] += 1
        
        return {
            'matching_size': matching_size,
            'match_left': match_left,
            'match_right': match_right,
            'statistics': self.stats.copy()
        }
    
    def maximum_matching_hopcroft_karp(self, left_vertices: int, right_vertices: int, edges: List[Tuple[int, int]]) -> Dict:
        """
        Approach 2: Hopcroft-Karp Algorithm (Optimal for Unweighted)
        
        Use level-wise BFS and DFS for optimal bipartite matching.
        
        Time: O(E * sqrt(V))
        Space: O(V + E)
        """
        self.reset_statistics()
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
        
        # Matching arrays
        pair_u = [-1] * left_vertices
        pair_v = [-1] * right_vertices
        dist = [0] * (left_vertices + 1)  # Distance array with NIL vertex
        NIL = left_vertices  # Special NIL vertex
        
        def bfs():
            """BFS to build level graph"""
            self.stats['bfs_calls'] += 1
            queue = deque()
            
            # Initialize distances
            for u in range(left_vertices):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            
            dist[NIL] = float('inf')
            
            # BFS traversal
            while queue:
                u = queue.popleft()
                
                if dist[u] < dist[NIL]:
                    for v in graph[u]:
                        self.stats['edges_explored'] += 1
                        
                        if pair_v[v] == -1:
                            if dist[NIL] == float('inf'):
                                dist[NIL] = dist[u] + 1
                        elif dist[pair_v[v]] == float('inf'):
                            dist[pair_v[v]] = dist[u] + 1
                            queue.append(pair_v[v])
            
            return dist[NIL] != float('inf')
        
        def dfs(u):
            """DFS to find augmenting paths"""
            if u == -1:
                return True
            
            self.stats['dfs_calls'] += 1
            
            for v in graph[u]:
                if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                    pair_v[v] = u
                    pair_u[u] = v
                    return True
            
            dist[u] = float('inf')
            return False
        
        matching_size = 0
        
        # Main Hopcroft-Karp loop
        while bfs():
            for u in range(left_vertices):
                if pair_u[u] == -1 and dfs(u):
                    matching_size += 1
                    self.stats['augmenting_paths_found'] += 1
            self.stats['iterations'] += 1
        
        return {
            'matching_size': matching_size,
            'match_left': pair_u,
            'match_right': pair_v,
            'statistics': self.stats.copy()
        }
    
    def maximum_matching_push_relabel(self, left_vertices: int, right_vertices: int, edges: List[Tuple[int, int]]) -> Dict:
        """
        Approach 3: Push-Relabel Algorithm for Maximum Flow
        
        Use push-relabel method to solve maximum flow formulation.
        
        Time: O(V^2 * E)
        Space: O(V^2)
        """
        self.reset_statistics()
        
        # Create flow network
        total_vertices = left_vertices + right_vertices + 2
        source = total_vertices - 2
        sink = total_vertices - 1
        
        # Build capacity matrix
        capacity = [[0] * total_vertices for _ in range(total_vertices)]
        
        # Source to left vertices
        for u in range(left_vertices):
            capacity[source][u] = 1
        
        # Right vertices to sink
        for v in range(right_vertices):
            capacity[left_vertices + v][sink] = 1
        
        # Left to right based on edges
        for u, v in edges:
            capacity[u][left_vertices + v] = 1
        
        # Push-relabel implementation
        height = [0] * total_vertices
        excess = [0] * total_vertices
        
        # Initialize preflow
        height[source] = total_vertices
        for u in range(left_vertices):
            if capacity[source][u] > 0:
                excess[u] = 1
                excess[source] -= 1
                capacity[source][u] = 0
                capacity[u][source] = 1
        
        def push(u, v):
            """Push operation"""
            delta = min(excess[u], capacity[u][v])
            excess[u] -= delta
            excess[v] += delta
            capacity[u][v] -= delta
            capacity[v][u] += delta
        
        def relabel(u):
            """Relabel operation"""
            min_height = float('inf')
            for v in range(total_vertices):
                if capacity[u][v] > 0:
                    min_height = min(min_height, height[v])
            height[u] = min_height + 1
        
        def discharge(u):
            """Discharge operation"""
            while excess[u] > 0:
                admissible_found = False
                for v in range(total_vertices):
                    if capacity[u][v] > 0 and height[u] == height[v] + 1:
                        push(u, v)
                        admissible_found = True
                        break
                
                if not admissible_found:
                    relabel(u)
                
                self.stats['iterations'] += 1
        
        # Main push-relabel loop
        changed = True
        while changed:
            changed = False
            for u in range(left_vertices):
                if excess[u] > 0:
                    discharge(u)
                    changed = True
        
        # Extract matching from flow
        match_left = [-1] * left_vertices
        match_right = [-1] * right_vertices
        matching_size = 0
        
        for u in range(left_vertices):
            for v in range(right_vertices):
                if capacity[left_vertices + v][u] == 1:  # Flow from right to left means matching
                    match_left[u] = v
                    match_right[v] = u
                    matching_size += 1
                    break
        
        return {
            'matching_size': matching_size,
            'match_left': match_left,
            'match_right': match_right,
            'statistics': self.stats.copy()
        }
    
    def maximum_weighted_matching_hungarian(self, left_vertices: int, right_vertices: int, 
                                          weighted_edges: List[Tuple[int, int, int]]) -> Dict:
        """
        Approach 4: Hungarian Algorithm for Maximum Weighted Bipartite Matching
        
        Solve maximum weight bipartite matching using Hungarian algorithm.
        
        Time: O(V^3)
        Space: O(V^2)
        """
        self.reset_statistics()
        
        # Create cost matrix (negate weights for maximum)
        n = max(left_vertices, right_vertices)
        cost = [[float('inf')] * n for _ in range(n)]
        
        for u, v, weight in weighted_edges:
            if u < n and v < n:
                cost[u][v] = -weight  # Negate for maximum weight
        
        # Hungarian algorithm implementation
        u = [0] * (n + 1)  # Potentials for left vertices
        v = [0] * (n + 1)  # Potentials for right vertices
        p = [-1] * (n + 1)  # Assignment
        way = [0] * (n + 1)  # Path reconstruction
        
        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [float('inf')] * (n + 1)
            used = [False] * (n + 1)
            
            while p[j0]:
                used[j0] = True
                i0 = p[j0]
                delta = float('inf')
                j1 = 0
                
                for j in range(1, n + 1):
                    if not used[j]:
                        cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                
                for j in range(n + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                
                j0 = j1
                self.stats['iterations'] += 1
            
            while j0:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
        
        # Extract result
        match_left = [-1] * left_vertices
        match_right = [-1] * right_vertices
        total_weight = 0
        matching_size = 0
        
        for j in range(1, n + 1):
            if p[j] > 0 and p[j] <= left_vertices and j - 1 < right_vertices:
                match_left[p[j] - 1] = j - 1
                match_right[j - 1] = p[j] - 1
                # Find original weight
                for u, v, weight in weighted_edges:
                    if u == p[j] - 1 and v == j - 1:
                        total_weight += weight
                        break
                matching_size += 1
        
        return {
            'matching_size': matching_size,
            'total_weight': total_weight,
            'match_left': match_left,
            'match_right': match_right,
            'statistics': self.stats.copy()
        }
    
    def approximate_matching_greedy(self, left_vertices: int, right_vertices: int, edges: List[Tuple[int, int]]) -> Dict:
        """
        Approach 5: Greedy Approximation Algorithm
        
        Use greedy approach for fast approximate matching.
        
        Time: O(E)
        Space: O(V)
        """
        self.reset_statistics()
        
        match_left = [-1] * left_vertices
        match_right = [-1] * right_vertices
        matching_size = 0
        
        # Shuffle edges for randomized greedy
        edges_copy = edges[:]
        random.shuffle(edges_copy)
        
        for u, v in edges_copy:
            self.stats['edges_explored'] += 1
            
            if match_left[u] == -1 and match_right[v] == -1:
                match_left[u] = v
                match_right[v] = u
                matching_size += 1
                self.stats['augmenting_paths_found'] += 1
        
        return {
            'matching_size': matching_size,
            'match_left': match_left,
            'match_right': match_right,
            'approximation_ratio': 0.5,  # Theoretical guarantee
            'statistics': self.stats.copy()
        }
    
    def online_matching_ranking(self, left_vertices: int, right_vertices: int, 
                               online_edges: List[List[Tuple[int, int]]]) -> Dict:
        """
        Approach 6: Online Matching with Ranking Algorithm
        
        Solve online bipartite matching where right vertices arrive dynamically.
        
        Time: O(E)
        Space: O(V)
        """
        self.reset_statistics()
        
        # Ranking algorithm: assign ranks to left vertices
        ranks = list(range(left_vertices))
        random.shuffle(ranks)
        
        match_left = [-1] * left_vertices
        match_right = [-1] * right_vertices
        matching_size = 0
        
        # Process right vertices as they arrive
        for right_vertex, connections in enumerate(online_edges):
            if right_vertex >= right_vertices:
                break
            
            # Find highest-ranked available left vertex
            best_left = -1
            best_rank = -1
            
            for left_vertex, _ in connections:
                if match_left[left_vertex] == -1 and ranks[left_vertex] > best_rank:
                    best_rank = ranks[left_vertex]
                    best_left = left_vertex
            
            # Make assignment if found
            if best_left != -1:
                match_left[best_left] = right_vertex
                match_right[right_vertex] = best_left
                matching_size += 1
            
            self.stats['iterations'] += 1
        
        return {
            'matching_size': matching_size,
            'match_left': match_left,
            'match_right': match_right,
            'competitive_ratio': 1 - 1/2.718,  # 1 - 1/e ≈ 0.632
            'statistics': self.stats.copy()
        }

def test_maximum_bipartite_matching():
    """Test all maximum bipartite matching algorithms"""
    matcher = MaximumBipartiteMatching()
    
    print("=== Testing Maximum Bipartite Matching Algorithms ===")
    
    # Test cases
    test_cases = [
        # (left_vertices, right_vertices, edges, expected_matching)
        (3, 3, [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)], 3),
        (4, 3, [(0, 0), (1, 1), (2, 1), (2, 2), (3, 2)], 3),
        (2, 4, [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3)], 2),
        (3, 2, [(0, 0), (1, 0), (2, 1)], 2),
        (1, 1, [(0, 0)], 1),
        (2, 2, [], 0),  # No edges
    ]
    
    algorithms = [
        ("Ford-Fulkerson DFS", matcher.maximum_matching_ford_fulkerson_dfs),
        ("Hopcroft-Karp", matcher.maximum_matching_hopcroft_karp),
        ("Push-Relabel", matcher.maximum_matching_push_relabel),
        ("Greedy Approx", matcher.approximate_matching_greedy),
    ]
    
    for i, (left_v, right_v, edges, expected) in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: L={left_v}, R={right_v}, E={len(edges)} ---")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(left_v, right_v, edges)
                matching_size = result['matching_size']
                stats = result['statistics']
                
                status = "✓" if matching_size == expected else "✗"
                print(f"{alg_name:20} | {status} | Size: {matching_size:2} | "
                      f"Paths: {stats.get('augmenting_paths_found', 0):2} | "
                      f"Iter: {stats.get('iterations', 0):3}")
                
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")

def demonstrate_algorithm_comparison():
    """Demonstrate performance comparison between algorithms"""
    print("\n=== Algorithm Performance Comparison ===")
    
    matcher = MaximumBipartiteMatching()
    
    # Create test graph
    left_vertices, right_vertices = 50, 50
    edges = []
    
    # Create random bipartite graph
    import random
    random.seed(42)
    for u in range(left_vertices):
        for v in range(right_vertices):
            if random.random() < 0.1:  # 10% edge probability
                edges.append((u, v))
    
    print(f"Test graph: {left_vertices} left vertices, {right_vertices} right vertices, {len(edges)} edges")
    
    algorithms = [
        ("Ford-Fulkerson DFS", matcher.maximum_matching_ford_fulkerson_dfs),
        ("Hopcroft-Karp", matcher.maximum_matching_hopcroft_karp),
        ("Greedy Approximation", matcher.approximate_matching_greedy),
    ]
    
    print(f"\nPerformance comparison:")
    print(f"{'Algorithm':<20} | {'Size':<4} | {'Paths':<5} | {'Iterations':<6} | {'Edges':<6}")
    print("-" * 65)
    
    for alg_name, alg_func in algorithms:
        try:
            result = alg_func(left_vertices, right_vertices, edges)
            stats = result['statistics']
            
            print(f"{alg_name:<20} | {result['matching_size']:<4} | "
                  f"{stats.get('augmenting_paths_found', 0):<5} | "
                  f"{stats.get('iterations', 0):<10} | "
                  f"{stats.get('edges_explored', 0):<6}")
            
        except Exception as e:
            print(f"{alg_name:<20} | ERROR: {str(e)[:20]}")

def demonstrate_hungarian_algorithm():
    """Demonstrate Hungarian algorithm for weighted matching"""
    print("\n=== Hungarian Algorithm Demo ===")
    
    matcher = MaximumBipartiteMatching()
    
    # Example weighted bipartite graph
    left_vertices, right_vertices = 3, 3
    weighted_edges = [
        (0, 0, 4), (0, 1, 2), (0, 2, 3),
        (1, 0, 2), (1, 1, 4), (1, 2, 6),
        (2, 0, 3), (2, 1, 6), (2, 2, 2)
    ]
    
    print(f"Weighted bipartite graph:")
    print(f"Left vertices: {list(range(left_vertices))}")
    print(f"Right vertices: {list(range(right_vertices))}")
    print(f"Weighted edges (left, right, weight):")
    for u, v, w in weighted_edges:
        print(f"  ({u}, {v}) → {w}")
    
    result = matcher.maximum_weighted_matching_hungarian(left_vertices, right_vertices, weighted_edges)
    
    print(f"\nHungarian algorithm result:")
    print(f"Maximum weight matching size: {result['matching_size']}")
    print(f"Total weight: {result['total_weight']}")
    print(f"Matching pairs:")
    
    for u in range(left_vertices):
        if result['match_left'][u] != -1:
            v = result['match_left'][u]
            weight = next(w for l, r, w in weighted_edges if l == u and r == v)
            print(f"  Left {u} ↔ Right {v} (weight {weight})")

def analyze_theoretical_foundations():
    """Analyze theoretical foundations of bipartite matching"""
    print("\n=== Theoretical Foundations ===")
    
    print("Maximum Bipartite Matching Theory:")
    
    print("\n1. **König's Theorem:**")
    print("   • In bipartite graphs: max matching = min vertex cover")
    print("   • Fundamental duality result")
    print("   • Constructive proof via augmenting paths")
    print("   • Applications in optimization")
    
    print("\n2. **Hall's Marriage Theorem:**")
    print("   • Perfect matching exists ⟺ Hall's condition")
    print("   • |N(S)| ≥ |S| for all S ⊆ Left vertices")
    print("   • Combinatorial characterization")
    print("   • Necessary and sufficient condition")
    
    print("\n3. **Augmenting Path Theory:**")
    print("   • Matching is maximum ⟺ no augmenting paths")
    print("   • Berge's theorem generalization")
    print("   • Algorithmic foundation")
    print("   • Complexity analysis basis")
    
    print("\n4. **Network Flow Connection:**")
    print("   • Max flow = max matching in unit capacity networks")
    print("   • Min cut = min vertex cover (König's theorem)")
    print("   • Integrality of maximum flows")
    print("   • Linear programming relaxation")
    
    print("\n5. **Complexity Results:**")
    print("   • Unweighted: O(E√V) optimal (Hopcroft-Karp)")
    print("   • Weighted: O(V³) optimal (Hungarian)")
    print("   • Online: 1-1/e competitive ratio")
    print("   • Approximation: 1/2-approximation in linear time")

def demonstrate_real_world_applications():
    """Demonstrate real-world applications and extensions"""
    print("\n=== Real-World Applications ===")
    
    print("Bipartite Matching Applications:")
    
    print("\n1. **Job Assignment Optimization:**")
    print("   • Workers ↔ Tasks with skill requirements")
    print("   • Maximize productivity or minimize cost")
    print("   • Handle preferences and constraints")
    print("   • Dynamic reassignment as conditions change")
    
    print("\n2. **Resource Allocation:**")
    print("   • Users ↔ Computing resources")
    print("   • Network bandwidth allocation")
    print("   • Memory and storage assignment")
    print("   • Load balancing in distributed systems")
    
    print("\n3. **Matching Markets:**")
    print("   • Medical residency matching")
    print("   • School choice programs")
    print("   • Kidney exchange programs")
    print("   • Online dating platforms")
    
    print("\n4. **Network Design:**")
    print("   • Communication network routing")
    print("   • Transportation network optimization")
    print("   • Supply chain management")
    print("   • Facility location problems")
    
    print("\n5. **Machine Learning:**")
    print("   • Feature selection optimization")
    print("   • Data association in tracking")
    print("   • Graph neural network optimization")
    print("   • Clustering and classification")

def demonstrate_advanced_variations():
    """Demonstrate advanced variations and extensions"""
    print("\n=== Advanced Variations ===")
    
    print("Bipartite Matching Extensions:")
    
    print("\n1. **Capacitated Matching:**")
    print("   • Vertices can handle multiple connections")
    print("   • Generalized assignment problems")
    print("   • Network flow with capacities")
    print("   • Applications in resource allocation")
    
    print("\n2. **Stable Matching:**")
    print("   • Both sides have preferences")
    print("   • No blocking pairs in solution")
    print("   • Gale-Shapley algorithm")
    print("   • Medical residency, school choice")
    
    print("\n3. **Dynamic Matching:**")
    print("   • Edges appear and disappear over time")
    print("   • Maintain matching under updates")
    print("   • Amortized analysis techniques")
    print("   • Online streaming algorithms")
    
    print("\n4. **Stochastic Matching:**")
    print("   • Uncertain edge existence/weights")
    print("   • Probabilistic constraints")
    print("   • Expected value optimization")
    print("   • Risk-aware decision making")
    
    print("\n5. **Multi-objective Matching:**")
    print("   • Optimize multiple criteria simultaneously")
    print("   • Pareto optimal solutions")
    print("   • Weighted sum approaches")
    print("   • Fairness and equity considerations")

if __name__ == "__main__":
    test_maximum_bipartite_matching()
    demonstrate_algorithm_comparison()
    demonstrate_hungarian_algorithm()
    analyze_theoretical_foundations()
    demonstrate_real_world_applications()
    demonstrate_advanced_variations()

"""
Maximum Bipartite Matching and Algorithm Theory Concepts:
1. Classical Bipartite Matching Algorithms and Complexity Analysis
2. Augmenting Path Methods and Network Flow Approaches
3. Weighted Matching and Hungarian Algorithm Implementation
4. Online and Approximation Algorithms for Dynamic Scenarios
5. Theoretical Foundations and Real-world Applications

Key Algorithmic Insights:
- Multiple approaches with different complexity trade-offs
- Ford-Fulkerson provides foundation for understanding
- Hopcroft-Karp achieves optimal O(E√V) complexity
- Hungarian algorithm solves weighted variant optimally
- Approximation and online algorithms handle practical constraints

Theoretical Foundations:
- König's theorem: max matching = min vertex cover
- Hall's marriage theorem: perfect matching characterization
- Augmenting path theory: optimality conditions
- Network flow duality and integrality
- Complexity theory and lower bounds

Algorithm Selection:
- Problem size and graph density considerations
- Weighted vs unweighted variants
- Online vs offline processing requirements
- Approximation quality vs computational efficiency
- Memory constraints and parallel processing

Implementation Techniques:
- Efficient data structures for graph representation
- Optimization of augmenting path finding
- Memory access pattern optimization
- Parallel and distributed implementations
- Integration with larger optimization frameworks

Real-world Impact:
- Foundation for assignment and allocation problems
- Critical for matching market design
- Essential for resource optimization
- Enables fair and efficient distribution systems
- Supports decision making under constraints

This comprehensive implementation provides production-ready
bipartite matching algorithms suitable for various applications.
"""
