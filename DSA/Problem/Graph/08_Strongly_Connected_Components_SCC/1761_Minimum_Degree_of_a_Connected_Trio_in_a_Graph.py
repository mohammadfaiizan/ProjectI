"""
1761. Minimum Degree of a Connected Trio in a Graph
Difficulty: Hard

Problem:
You are given an undirected graph. You are given an integer n which is the number of nodes in the graph 
and an array edges where edges[i] = [ai, bi] indicates that there is an edge between ai and bi in the graph.

A connected trio is a set of three nodes where there is an edge between every pair of them.

The degree of a connected trio is the number of edges where one endpoint is in the trio, and the other is not.

Return the minimum degree of a connected trio in the graph, or -1 if the graph has no connected trios.

Examples:
Input: n = 6, edges = [[1,2],[1,3],[3,2],[4,1],[5,2],[3,6]]
Output: 3
Explanation: There is exactly one trio, which is [1,2,3]. The edges that form its degree are bolded in the above figure.

Input: n = 7, edges = [[1,3],[4,1],[4,3],[2,5],[5,6],[6,7],[7,5],[2,6]]
Output: 0
Explanation: There are exactly three trios:
1) [1,3,4] with degree 0. All edges are internal.
2) [2,5,6] with degree 2. The edges in red are external.
3) [5,6,7] with degree 2. The edges in red are external.
The minimum is 0.

Constraints:
- 2 <= n <= 400
- edges.length <= n * (n-1) / 2
- 1 <= ai, bi <= n
- ai != bi
- There are no repeated edges.
"""

from typing import List, Set, Dict, Tuple
from collections import defaultdict

class Solution:
    def minTrioDegree_approach1_brute_force_optimized(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 1: Optimized Brute Force
        
        Check all possible triangles and calculate their external degrees.
        
        Time: O(V^3)
        Space: O(V^2)
        """
        # Build adjacency set for O(1) edge lookup
        adj_set = set()
        adj_list = defaultdict(set)
        degree = [0] * (n + 1)
        
        for u, v in edges:
            adj_set.add((min(u, v), max(u, v)))
            adj_list[u].add(v)
            adj_list[v].add(u)
            degree[u] += 1
            degree[v] += 1
        
        min_degree = float('inf')
        
        # Check all possible trios (triangles)
        for a in range(1, n + 1):
            for b in range(a + 1, n + 1):
                if b not in adj_list[a]:
                    continue
                
                for c in range(b + 1, n + 1):
                    if (c not in adj_list[a] or c not in adj_list[b]):
                        continue
                    
                    # Found triangle (a, b, c)
                    # Calculate external degree
                    trio_degree = degree[a] + degree[b] + degree[c] - 6
                    min_degree = min(min_degree, trio_degree)
        
        return min_degree if min_degree != float('inf') else -1
    
    def minTrioDegree_approach2_adjacency_matrix_triangles(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 2: Adjacency Matrix with Triangle Detection
        
        Use adjacency matrix for efficient triangle detection.
        
        Time: O(V^3)
        Space: O(V^2)
        """
        # Build adjacency matrix and degree array
        adj_matrix = [[False] * (n + 1) for _ in range(n + 1)]
        degree = [0] * (n + 1)
        
        for u, v in edges:
            adj_matrix[u][v] = True
            adj_matrix[v][u] = True
            degree[u] += 1
            degree[v] += 1
        
        min_degree = float('inf')
        
        # Find all triangles
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if not adj_matrix[i][j]:
                    continue
                
                for k in range(j + 1, n + 1):
                    if adj_matrix[i][k] and adj_matrix[j][k]:
                        # Triangle found: (i, j, k)
                        external_degree = degree[i] + degree[j] + degree[k] - 6
                        min_degree = min(min_degree, external_degree)
        
        return min_degree if min_degree != float('inf') else -1
    
    def minTrioDegree_approach3_neighbor_intersection(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 3: Neighbor Intersection for Triangle Finding
        
        Find triangles by intersecting neighbor sets.
        
        Time: O(V * E) in practice, O(V^3) worst case
        Space: O(V + E)
        """
        # Build adjacency list and degree count
        adj_list = defaultdict(set)
        degree = defaultdict(int)
        
        for u, v in edges:
            adj_list[u].add(v)
            adj_list[v].add(u)
            degree[u] += 1
            degree[v] += 1
        
        min_degree = float('inf')
        processed_pairs = set()
        
        # For each edge, find common neighbors
        for u, v in edges:
            pair = (min(u, v), max(u, v))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)
            
            # Find common neighbors of u and v
            common_neighbors = adj_list[u] & adj_list[v]
            
            for w in common_neighbors:
                # Triangle found: (u, v, w)
                triangle = tuple(sorted([u, v, w]))
                external_degree = degree[u] + degree[v] + degree[w] - 6
                min_degree = min(min_degree, external_degree)
        
        return min_degree if min_degree != float('inf') else -1
    
    def minTrioDegree_approach4_degree_optimization(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 4: Degree-based Optimization
        
        Optimize by processing vertices in degree order.
        
        Time: O(V^3) but with practical improvements
        Space: O(V^2)
        """
        # Build graph structures
        adj_set = set()
        adj_list = defaultdict(list)
        degree = [0] * (n + 1)
        
        for u, v in edges:
            adj_set.add((min(u, v), max(u, v)))
            adj_list[u].append(v)
            adj_list[v].append(u)
            degree[u] += 1
            degree[v] += 1
        
        # Sort vertices by degree (heuristic: high-degree vertices more likely in triangles)
        vertices = list(range(1, n + 1))
        vertices.sort(key=lambda x: degree[x], reverse=True)
        
        min_degree = float('inf')
        
        # Check triangles with degree-ordered vertices
        for i in range(len(vertices)):
            a = vertices[i]
            for j in range(i + 1, len(vertices)):
                b = vertices[j]
                
                # Early termination: if remaining degree sum is too large
                if degree[a] + degree[b] - 4 >= min_degree:
                    continue
                
                if (min(a, b), max(a, b)) not in adj_set:
                    continue
                
                for k in range(j + 1, len(vertices)):
                    c = vertices[k]
                    
                    # Early termination
                    if degree[a] + degree[b] + degree[c] - 6 >= min_degree:
                        continue
                    
                    if ((min(a, c), max(a, c)) in adj_set and 
                        (min(b, c), max(b, c)) in adj_set):
                        
                        trio_degree = degree[a] + degree[b] + degree[c] - 6
                        min_degree = min(min_degree, trio_degree)
        
        return min_degree if min_degree != float('inf') else -1
    
    def minTrioDegree_approach5_advanced_pruning(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 5: Advanced Pruning and Triangle Enumeration
        
        Use advanced pruning techniques for large graphs.
        
        Time: O(V^3) with significant pruning
        Space: O(V^2)
        """
        if len(edges) < 3:
            return -1
        
        # Build adjacency structures
        adj_matrix = [[False] * (n + 1) for _ in range(n + 1)]
        neighbors = [[] for _ in range(n + 1)]
        degree = [0] * (n + 1)
        
        for u, v in edges:
            adj_matrix[u][v] = adj_matrix[v][u] = True
            neighbors[u].append(v)
            neighbors[v].append(u)
            degree[u] += 1
            degree[v] += 1
        
        # Sort neighbors by degree (another heuristic)
        for i in range(1, n + 1):
            neighbors[i].sort(key=lambda x: degree[x])
        
        min_degree = float('inf')
        
        def find_triangles_from_vertex(v):
            """Find all triangles containing vertex v"""
            nonlocal min_degree
            
            # Early termination if vertex degree is too low
            if degree[v] < 2:
                return
            
            v_neighbors = neighbors[v]
            
            for i in range(len(v_neighbors)):
                u = v_neighbors[i]
                if u >= v:  # Avoid duplicate triangles
                    break
                
                for j in range(i + 1, len(v_neighbors)):
                    w = v_neighbors[j]
                    if w >= v:  # Avoid duplicate triangles
                        break
                    
                    if adj_matrix[u][w]:
                        # Triangle found: (u, v, w)
                        trio_degree = degree[u] + degree[v] + degree[w] - 6
                        min_degree = min(min_degree, trio_degree)
                        
                        # Early global termination
                        if min_degree == 0:
                            return
        
        # Process vertices in degree order
        vertex_list = list(range(1, n + 1))
        vertex_list.sort(key=lambda x: degree[x], reverse=True)
        
        for vertex in vertex_list:
            find_triangles_from_vertex(vertex)
            if min_degree == 0:
                break
        
        return min_degree if min_degree != float('inf') else -1

def test_minimum_trio_degree():
    """Test all approaches with various test cases"""
    solution = Solution()
    
    test_cases = [
        # (n, edges, expected)
        (6, [[1,2],[1,3],[3,2],[4,1],[5,2],[3,6]], 3),
        (7, [[1,3],[4,1],[4,3],[2,5],[5,6],[6,7],[7,5],[2,6]], 0),
        (4, [[1,2],[1,3],[2,3]], 0),
        (5, [[1,2],[2,3],[1,3],[1,4],[2,5]], 2),
        (3, [[1,2]], -1),  # No triangle
        (4, [[1,2],[2,3],[3,4],[4,1]], -1),  # Cycle but no triangle
    ]
    
    approaches = [
        ("Brute Force Optimized", solution.minTrioDegree_approach1_brute_force_optimized),
        ("Adjacency Matrix", solution.minTrioDegree_approach2_adjacency_matrix_triangles),
        ("Neighbor Intersection", solution.minTrioDegree_approach3_neighbor_intersection),
        ("Degree Optimization", solution.minTrioDegree_approach4_degree_optimization),
        ("Advanced Pruning", solution.minTrioDegree_approach5_advanced_pruning),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, edges, expected) in enumerate(test_cases):
            result = func(n, edges[:])  # Copy edges
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} n={n}, expected={expected}, got={result}")

def demonstrate_trio_analysis():
    """Demonstrate connected trio analysis"""
    print("\n=== Connected Trio Analysis Demo ===")
    
    n = 6
    edges = [[1,2],[1,3],[3,2],[4,1],[5,2],[3,6]]
    
    print(f"Graph: n={n}, edges={edges}")
    
    # Build adjacency for visualization
    adj = defaultdict(list)
    degree = defaultdict(int)
    
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1
    
    print(f"\nAdjacency list:")
    for vertex in range(1, n + 1):
        print(f"  {vertex}: {sorted(adj[vertex])}")
    
    print(f"\nVertex degrees:")
    for vertex in range(1, n + 1):
        print(f"  {vertex}: degree {degree[vertex]}")
    
    print(f"\nTrio analysis:")
    # Find triangles manually
    triangles = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if b in adj[a]:
                for c in range(b + 1, n + 1):
                    if c in adj[a] and c in adj[b]:
                        external_degree = degree[a] + degree[b] + degree[c] - 6
                        triangles.append((a, b, c, external_degree))
    
    if triangles:
        print(f"  Found triangles:")
        for a, b, c, ext_deg in triangles:
            print(f"    Triangle {(a, b, c)}: external degree = {ext_deg}")
            print(f"      Total degree: {degree[a]} + {degree[b]} + {degree[c]} = {degree[a] + degree[b] + degree[c]}")
            print(f"      Internal edges: 3 pairs × 2 = 6")
            print(f"      External degree: {degree[a] + degree[b] + degree[c]} - 6 = {ext_deg}")
    else:
        print(f"  No triangles found")
    
    solution = Solution()
    result = solution.minTrioDegree_approach1_brute_force_optimized(n, edges)
    print(f"\nMinimum trio degree: {result}")

def demonstrate_triangle_enumeration():
    """Demonstrate triangle enumeration techniques"""
    print("\n=== Triangle Enumeration Techniques ===")
    
    print("Triangle Detection Methods:")
    
    print("\n1. **Brute Force (O(V³)):**")
    print("   • Check all vertex triplets")
    print("   • Verify edge existence for each pair")
    print("   • Simple but inefficient for large graphs")
    print("   • Good baseline for correctness verification")
    
    print("\n2. **Edge-based Enumeration (O(E√E)):**")
    print("   • For each edge, find common neighbors")
    print("   • Intersection of adjacency lists")
    print("   • More efficient for sparse graphs")
    print("   • Practical for many real-world networks")
    
    print("\n3. **Degree-ordered Processing:**")
    print("   • Process vertices by degree order")
    print("   • High-degree vertices more likely in triangles")
    print("   • Enables early pruning optimizations")
    print("   • Effective for scale-free networks")
    
    print("\n4. **Matrix Multiplication (O(V^ω)):**")
    print("   • A³ matrix gives triangle counts")
    print("   • ω ≈ 2.373 with advanced algorithms")
    print("   • Theoretical interest, complex implementation")
    print("   • Good for dense graphs")
    
    print("\n5. **Compact Forward Algorithm:**")
    print("   • Node ordering by degree or other criteria")
    print("   • Only check forward neighbors")
    print("   • Avoids duplicate triangle counting")
    print("   • Practical for implementation")

def analyze_graph_clustering_applications():
    """Analyze applications in graph clustering and community detection"""
    print("\n=== Graph Clustering Applications ===")
    
    print("Connected Trio Applications in Graph Analysis:")
    
    print("\n1. **Social Network Analysis:**")
    print("   • Friend group identification")
    print("   • Social circle cohesion measurement")
    print("   • Influence network analysis")
    print("   • Community boundary detection")
    
    print("\n2. **Biological Networks:**")
    print("   • Protein complex identification")
    print("   • Gene interaction triangles")
    print("   • Metabolic pathway clustering")
    print("   • Functional module detection")
    
    print("\n3. **Recommendation Systems:**")
    print("   • Collaborative filtering triangles")
    print("   • User similarity clustering")
    print("   • Item relationship analysis")
    print("   • Trust network formation")
    
    print("\n4. **Network Security:**")
    print("   • Suspicious activity pattern detection")
    print("   • Collusion group identification")
    print("   • Anomaly detection in communication")
    print("   • Fraud prevention triangulation")
    
    print("\n5. **Transportation Networks:**")
    print("   • Route optimization triangles")
    print("   • Hub identification")
    print("   • Network efficiency analysis")
    print("   • Traffic pattern clustering")
    
    print("\nClustering Metrics:")
    print("• **Clustering coefficient**: Local triangle density")
    print("• **Transitivity**: Global triangle probability")
    print("• **Triangle participation ratio**: Nodes in triangles")
    print("• **Triangle connectivity**: Inter-cluster triangles")

def demonstrate_optimization_techniques():
    """Demonstrate optimization techniques for large graphs"""
    print("\n=== Optimization Techniques ===")
    
    print("Large Graph Optimization Strategies:")
    
    print("\n1. **Early Termination:**")
    print("   • Stop when minimum possible degree found")
    print("   • Degree-based bounds for pruning")
    print("   • Progressive minimum updates")
    print("   • Target value optimization")
    
    print("\n2. **Vertex Ordering:**")
    print("   • Process high-degree vertices first")
    print("   • Degeneracy ordering for efficiency")
    print("   • Core-based decomposition")
    print("   • Adaptive ordering strategies")
    
    print("\n3. **Data Structure Optimization:**")
    print("   • Adjacency list vs matrix trade-offs")
    print("   • Compressed sparse representations")
    print("   • Cache-friendly memory layouts")
    print("   • Parallel processing structures")
    
    print("\n4. **Algorithmic Improvements:**")
    print("   • Neighbor intersection algorithms")
    print("   • Forward/backward edge processing")
    print("   • Incremental triangle updates")
    print("   • Approximation algorithms")
    
    print("\n5. **Practical Considerations:**")
    print("   • Memory usage optimization")
    print("   • I/O efficiency for large graphs")
    print("   • Distributed computation")
    print("   • Streaming algorithms")

def analyze_complexity_and_bounds():
    """Analyze complexity and theoretical bounds"""
    print("\n=== Complexity Analysis ===")
    
    print("Triangle Enumeration Complexity:")
    
    print("\n1. **Time Complexity Bounds:**")
    print("   • Brute force: O(V³)")
    print("   • Edge iteration: O(E^1.5) for sparse graphs")
    print("   • Matrix multiplication: O(V^ω) where ω ≈ 2.373")
    print("   • Practical algorithms: O(m^1.5) where m = |E|")
    
    print("\n2. **Space Complexity:**")
    print("   • Adjacency list: O(V + E)")
    print("   • Adjacency matrix: O(V²)")
    print("   • Compressed representations: O(E)")
    print("   • Streaming: O(V) space")
    
    print("\n3. **Output Sensitivity:**")
    print("   • O(T) where T = number of triangles")
    print("   • Dense graphs: T = O(V³)")
    print("   • Sparse graphs: T = O(E^1.5)")
    print("   • Real networks: often T = O(E)")
    
    print("\n4. **Lower Bounds:**")
    print("   • Matrix multiplication reduction")
    print("   • 3SUM hardness connections")
    print("   • Communication complexity bounds")
    print("   • Conditional lower bounds")
    
    print("\n5. **Practical Performance:**")
    print("   • Real-world graphs often sparse")
    print("   • Power-law degree distributions")
    print("   • Small-world properties")
    print("   • Clustering structure exploitation")

if __name__ == "__main__":
    test_minimum_trio_degree()
    demonstrate_trio_analysis()
    demonstrate_triangle_enumeration()
    analyze_graph_clustering_applications()
    demonstrate_optimization_techniques()
    analyze_complexity_and_bounds()

"""
Connected Trio and Triangle Analysis Concepts:
1. Triangle Enumeration Algorithms and Optimization Techniques
2. Graph Clustering and Community Detection Applications
3. Degree-based Analysis and Network Metrics
4. Social Network Analysis and Influence Modeling
5. Algorithmic Optimization for Large-scale Graph Processing

Key Problem Insights:
- Connected trios are triangles with external degree calculation
- External degree = total degree - internal edges (6 for triangle)
- Triangle enumeration fundamental to graph clustering
- Multiple algorithmic approaches with different trade-offs

Algorithm Strategy:
1. Enumerate all triangles in the graph
2. Calculate external degree for each triangle
3. Return minimum external degree found
4. Use optimization techniques for large graphs

Triangle Detection Methods:
- Brute force: Check all vertex triplets
- Edge-based: Find common neighbors of each edge
- Matrix-based: Use adjacency matrix properties
- Degree-ordered: Process by vertex degree priorities

Optimization Techniques:
- Early termination with bounds
- Vertex ordering by degree or other metrics
- Data structure optimization for cache efficiency
- Algorithmic improvements with neighbor intersection

Real-world Applications:
- Social network community detection
- Biological network module identification
- Recommendation system clustering
- Network security pattern detection
- Transportation network optimization

This problem demonstrates triangle enumeration techniques
essential for graph clustering and community analysis.
"""
