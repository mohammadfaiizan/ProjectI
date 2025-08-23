"""
1584. Min Cost to Connect All Points
Difficulty: Medium

Problem:
You are given an array points representing integer coordinates of some points on a 2D-plane, 
where points[i] = [xi, yi].

The cost of connecting two points [xi, yi] and [xj, yj] is the manhattan distance between them: 
|xi - xj| + |yi - yj|.

Return the minimum cost to make all points connected. All points are connected if there is 
exactly one simple path between any two points.

Examples:
Input: points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
Output: 20

Input: points = [[3,12],[-2,5],[-4,1]]
Output: 18

Input: points = [[0,0],[1,1],[1,0],[-1,1]]
Output: 4

Input: points = [[-1000000,-1000000],[1000000,1000000]]
Output: 4000000

Constraints:
- 1 <= points.length <= 1000
- -10^6 <= xi, yi <= 10^6
- All pairs (xi, yi) are distinct
"""

from typing import List
import heapq

class Solution:
    def minCostConnectPoints_approach1_prims_algorithm(self, points: List[List[int]]) -> int:
        """
        Approach 1: Prim's Algorithm (Optimal for dense graphs)
        
        Build MST by starting from one vertex and greedily adding minimum edges.
        
        Time: O(V^2) or O(E log V) with heap
        Space: O(V)
        """
        if len(points) <= 1:
            return 0
        
        n = len(points)
        
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # Prim's algorithm with heap
        visited = [False] * n
        min_heap = [(0, 0)]  # (cost, vertex)
        total_cost = 0
        edges_added = 0
        
        while min_heap and edges_added < n:
            cost, u = heapq.heappop(min_heap)
            
            if visited[u]:
                continue
            
            # Add vertex to MST
            visited[u] = True
            total_cost += cost
            edges_added += 1
            
            # Add all edges from u to unvisited vertices
            for v in range(n):
                if not visited[v]:
                    edge_cost = manhattan_distance(points[u], points[v])
                    heapq.heappush(min_heap, (edge_cost, v))
        
        return total_cost
    
    def minCostConnectPoints_approach2_kruskals_algorithm(self, points: List[List[int]]) -> int:
        """
        Approach 2: Kruskal's Algorithm with Union-Find
        
        Sort all edges and greedily add edges that don't create cycles.
        
        Time: O(E log E) = O(V^2 log V)
        Space: O(E) = O(V^2)
        """
        if len(points) <= 1:
            return 0
        
        n = len(points)
        
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # Union-Find data structure
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
        
        # Generate all edges
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                cost = manhattan_distance(points[i], points[j])
                edges.append((cost, i, j))
        
        # Sort edges by cost
        edges.sort()
        
        # Kruskal's algorithm
        total_cost = 0
        edges_added = 0
        
        for cost, u, v in edges:
            if union(u, v):
                total_cost += cost
                edges_added += 1
                
                if edges_added == n - 1:
                    break
        
        return total_cost
    
    def minCostConnectPoints_approach3_prims_adjacency_matrix(self, points: List[List[int]]) -> int:
        """
        Approach 3: Prim's with Adjacency Matrix (Classic Implementation)
        
        Use adjacency matrix representation for Prim's algorithm.
        
        Time: O(V^2)
        Space: O(V^2)
        """
        if len(points) <= 1:
            return 0
        
        n = len(points)
        
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # Build adjacency matrix
        adj_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    adj_matrix[i][j] = manhattan_distance(points[i], points[j])
        
        # Prim's algorithm
        visited = [False] * n
        min_cost = [float('inf')] * n
        min_cost[0] = 0
        total_cost = 0
        
        for _ in range(n):
            # Find minimum cost unvisited vertex
            u = -1
            for v in range(n):
                if not visited[v] and (u == -1 or min_cost[v] < min_cost[u]):
                    u = v
            
            # Add to MST
            visited[u] = True
            total_cost += min_cost[u]
            
            # Update costs to neighbors
            for v in range(n):
                if not visited[v] and adj_matrix[u][v] < min_cost[v]:
                    min_cost[v] = adj_matrix[u][v]
        
        return total_cost
    
    def minCostConnectPoints_approach4_boruvkas_algorithm(self, points: List[List[int]]) -> int:
        """
        Approach 4: Borůvka's Algorithm
        
        Parallel MST algorithm that adds multiple edges in each iteration.
        
        Time: O(E log V)
        Space: O(V + E)
        """
        if len(points) <= 1:
            return 0
        
        n = len(points)
        
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # Union-Find
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
        
        total_cost = 0
        
        while True:
            # Find cheapest edge for each component
            cheapest = [-1] * n
            cheapest_cost = [float('inf')] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    root_i, root_j = find(i), find(j)
                    
                    if root_i != root_j:
                        cost = manhattan_distance(points[i], points[j])
                        
                        if cost < cheapest_cost[root_i]:
                            cheapest_cost[root_i] = cost
                            cheapest[root_i] = j
                        
                        if cost < cheapest_cost[root_j]:
                            cheapest_cost[root_j] = cost
                            cheapest[root_j] = i
            
            # Add cheapest edges
            edges_added = 0
            for i in range(n):
                if cheapest[i] != -1:
                    root_i = find(i)
                    root_j = find(cheapest[i])
                    
                    if root_i != root_j and union(i, cheapest[i]):
                        total_cost += cheapest_cost[i]
                        edges_added += 1
            
            if edges_added == 0:
                break
        
        return total_cost
    
    def minCostConnectPoints_approach5_optimized_prims(self, points: List[List[int]]) -> int:
        """
        Approach 5: Optimized Prim's with Early Termination
        
        Optimized version with various heuristics for better performance.
        
        Time: O(V^2) average case with optimizations
        Space: O(V)
        """
        if len(points) <= 1:
            return 0
        
        n = len(points)
        
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # Use a more efficient representation
        visited = [False] * n
        min_cost = [float('inf')] * n
        parent = [-1] * n
        
        # Start from point 0
        min_cost[0] = 0
        total_cost = 0
        
        for _ in range(n):
            # Find minimum cost unvisited vertex (can be optimized with heap)
            u = -1
            for v in range(n):
                if not visited[v] and (u == -1 or min_cost[v] < min_cost[u]):
                    u = v
            
            # Add to MST
            visited[u] = True
            total_cost += min_cost[u]
            
            # Update costs to unvisited neighbors
            for v in range(n):
                if not visited[v]:
                    cost = manhattan_distance(points[u], points[v])
                    if cost < min_cost[v]:
                        min_cost[v] = cost
                        parent[v] = u
        
        return total_cost

def test_min_cost_connect_points():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (points, expected)
        ([[0,0],[2,2],[3,10],[5,2],[7,0]], 20),
        ([[3,12],[-2,5],[-4,1]], 18),
        ([[0,0],[1,1],[1,0],[-1,1]], 4),
        ([[-1000000,-1000000],[1000000,1000000]], 4000000),
        ([[0,0]], 0),
        ([[0,0],[1,0]], 1),
        ([[0,0],[1,0],[2,0]], 2),
    ]
    
    approaches = [
        ("Prim's Algorithm", solution.minCostConnectPoints_approach1_prims_algorithm),
        ("Kruskal's Algorithm", solution.minCostConnectPoints_approach2_kruskals_algorithm),
        ("Prim's Adjacency Matrix", solution.minCostConnectPoints_approach3_prims_adjacency_matrix),
        ("Borůvka's Algorithm", solution.minCostConnectPoints_approach4_boruvkas_algorithm),
        ("Optimized Prim's", solution.minCostConnectPoints_approach5_optimized_prims),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (points, expected) in enumerate(test_cases):
            result = func(points[:])  # Deep copy
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Expected: {expected}, Got: {result}")

def demonstrate_mst_algorithms():
    """Demonstrate MST algorithm concepts"""
    print("\n=== MST Algorithms Demo ===")
    
    points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
    
    print(f"Points: {points}")
    print(f"Goal: Connect all points with minimum total Manhattan distance")
    
    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    # Show all possible edges
    n = len(points)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            cost = manhattan_distance(points[i], points[j])
            edges.append((cost, i, j))
    
    edges.sort()
    
    print(f"\nAll possible edges (sorted by cost):")
    for cost, i, j in edges:
        print(f"  ({i},{j}): {points[i]} ↔ {points[j]} = {cost}")
    
    print(f"\nMST requires exactly {n-1} edges to connect {n} vertices")
    print(f"Total possible edges: {len(edges)}")

def demonstrate_prims_algorithm():
    """Demonstrate Prim's algorithm step by step"""
    print("\n=== Prim's Algorithm Demo ===")
    
    points = [[0,0],[2,2],[3,10],[5,2]]
    n = len(points)
    
    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    print(f"Points: {points}")
    print(f"Starting Prim's algorithm from vertex 0")
    
    visited = [False] * n
    min_cost = [float('inf')] * n
    parent = [-1] * n
    min_cost[0] = 0
    total_cost = 0
    
    for step in range(n):
        # Find minimum cost unvisited vertex
        u = -1
        for v in range(n):
            if not visited[v] and (u == -1 or min_cost[v] < min_cost[u]):
                u = v
        
        print(f"\nStep {step + 1}: Adding vertex {u} (cost: {min_cost[u]})")
        visited[u] = True
        total_cost += min_cost[u]
        
        if parent[u] != -1:
            print(f"  Edge: {parent[u]} → {u}")
        
        print(f"  Current MST cost: {total_cost}")
        
        # Update costs to unvisited neighbors
        print(f"  Updating costs to neighbors:")
        for v in range(n):
            if not visited[v]:
                cost = manhattan_distance(points[u], points[v])
                if cost < min_cost[v]:
                    old_cost = min_cost[v] if min_cost[v] != float('inf') else '∞'
                    min_cost[v] = cost
                    parent[v] = u
                    print(f"    Vertex {v}: {old_cost} → {cost}")
    
    print(f"\nFinal MST cost: {total_cost}")

def demonstrate_kruskals_algorithm():
    """Demonstrate Kruskal's algorithm step by step"""
    print("\n=== Kruskal's Algorithm Demo ===")
    
    points = [[0,0],[2,2],[3,10],[5,2]]
    n = len(points)
    
    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    # Union-Find
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
    
    # Generate and sort all edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            cost = manhattan_distance(points[i], points[j])
            edges.append((cost, i, j))
    
    edges.sort()
    
    print(f"Points: {points}")
    print(f"All edges sorted by cost: {edges}")
    
    total_cost = 0
    mst_edges = []
    
    print(f"\nKruskal's algorithm:")
    for i, (cost, u, v) in enumerate(edges):
        root_u, root_v = find(u), find(v)
        
        print(f"Step {i + 1}: Consider edge ({u},{v}) with cost {cost}")
        print(f"  Components: {u} in {root_u}, {v} in {root_v}")
        
        if union(u, v):
            total_cost += cost
            mst_edges.append((u, v, cost))
            print(f"  ✓ Added to MST (no cycle)")
            print(f"  Current MST cost: {total_cost}")
        else:
            print(f"  ✗ Rejected (would create cycle)")
        
        if len(mst_edges) == n - 1:
            print(f"  MST complete!")
            break
    
    print(f"\nFinal MST edges: {mst_edges}")
    print(f"Final MST cost: {total_cost}")

def analyze_mst_algorithms():
    """Analyze different MST algorithms"""
    print("\n=== MST Algorithms Analysis ===")
    
    print("Minimum Spanning Tree Problem:")
    print("• Given: Connected, weighted graph")
    print("• Goal: Find tree that connects all vertices with minimum total weight")
    print("• Properties: Exactly V-1 edges, no cycles, connected")
    
    print("\nMST Algorithms Comparison:")
    
    print("\n1. **Prim's Algorithm:**")
    print("   • Strategy: Grow tree by adding minimum weight edge to current tree")
    print("   • Time: O(V²) with array, O(E log V) with heap")
    print("   • Space: O(V)")
    print("   • Best for: Dense graphs (E ≈ V²)")
    print("   • Implementation: Start from arbitrary vertex, expand greedily")
    
    print("\n2. **Kruskal's Algorithm:**")
    print("   • Strategy: Sort edges, add minimum edges that don't create cycles")
    print("   • Time: O(E log E) = O(E log V)")
    print("   • Space: O(V) for Union-Find")
    print("   • Best for: Sparse graphs (E << V²)")
    print("   • Implementation: Union-Find for cycle detection")
    
    print("\n3. **Borůvka's Algorithm:**")
    print("   • Strategy: Parallel algorithm, add multiple edges per iteration")
    print("   • Time: O(E log V)")
    print("   • Space: O(V)")
    print("   • Best for: Parallel processing")
    print("   • Implementation: Each component finds cheapest outgoing edge")
    
    print("\nChoosing the Right Algorithm:")
    print("• **Dense graphs (E ≈ V²):** Prim's with adjacency matrix")
    print("• **Sparse graphs (E << V²):** Kruskal's or Prim's with heap")
    print("• **Parallel processing:** Borůvka's algorithm")
    print("• **Online algorithms:** Prim's (easier to make incremental)")

def analyze_mst_properties():
    """Analyze MST properties and theorems"""
    print("\n=== MST Properties and Theorems ===")
    
    print("Fundamental MST Properties:")
    
    print("\n1. **Cut Property:**")
    print("   • For any cut of the graph, the minimum weight edge crossing the cut")
    print("   • is in some MST (if edge weights are distinct, it's in every MST)")
    print("   • Foundation for correctness of Prim's algorithm")
    
    print("\n2. **Cycle Property:**")
    print("   • For any cycle in the graph, the maximum weight edge in the cycle")
    print("   • is not in any MST (if edge weights are distinct)")
    print("   • Foundation for correctness of Kruskal's algorithm")
    
    print("\n3. **Uniqueness:**")
    print("   • If all edge weights are distinct, MST is unique")
    print("   • Otherwise, multiple MSTs may exist with same total weight")
    
    print("\n4. **MST vs Shortest Path:**")
    print("   • MST minimizes total edge weight")
    print("   • Shortest path tree minimizes distances from source")
    print("   • Different objectives, different trees")
    
    print("\nMST Applications:")
    print("• **Network Design:** Minimum cost to connect all nodes")
    print("• **Clustering:** Remove heaviest edges to create clusters")
    print("• **Approximation:** For Steiner tree, traveling salesman")
    print("• **Image Segmentation:** Minimum spanning tree clustering")
    print("• **Circuit Design:** Minimum wire length connections")

def compare_real_world_applications():
    """Compare real-world applications of MST"""
    print("\n=== Real-World MST Applications ===")
    
    print("1. **Network Infrastructure:**")
    print("   • Telecommunication network design")
    print("   • Computer network topology")
    print("   • Power grid connections")
    print("   • Water/gas pipeline networks")
    
    print("\n2. **Transportation:**")
    print("   • Road network planning")
    print("   • Railway line construction")
    print("   • Airline route optimization")
    print("   • Public transit system design")
    
    print("\n3. **Circuit Design:**")
    print("   • VLSI circuit layout")
    print("   • Printed circuit board routing")
    print("   • Minimizing wire length")
    print("   • Reducing signal interference")
    
    print("\n4. **Data Analysis:**")
    print("   • Cluster analysis")
    print("   • Image segmentation")
    print("   • Social network analysis")
    print("   • Phylogenetic tree construction")
    
    print("\n5. **Approximation Algorithms:**")
    print("   • TSP 2-approximation")
    print("   • Steiner tree approximation")
    print("   • Facility location problems")
    print("   • Network design problems")
    
    print("\nPractical Considerations:")
    print("• **Scale:** Algorithm choice depends on graph size")
    print("• **Dynamic updates:** Adding/removing vertices or edges")
    print("• **Distributed computation:** Parallel MST algorithms")
    print("• **Constraints:** Additional requirements beyond minimum weight")
    print("• **Approximation:** Trade-offs between quality and speed")

if __name__ == "__main__":
    test_min_cost_connect_points()
    demonstrate_mst_algorithms()
    demonstrate_prims_algorithm()
    demonstrate_kruskals_algorithm()
    analyze_mst_algorithms()
    analyze_mst_properties()
    compare_real_world_applications()

"""
Minimum Spanning Tree (MST) Concepts:
1. Prim's Algorithm - Vertex-based greedy approach
2. Kruskal's Algorithm - Edge-based with Union-Find
3. Borůvka's Algorithm - Parallel MST construction
4. Cut Property and Cycle Property
5. MST Applications and Real-world Usage

Key Problem Insights:
- Connect all points with minimum total cost
- Manhattan distance as edge weight function
- Multiple MST algorithms with different trade-offs
- Choice depends on graph density and requirements

Algorithm Strategy:
1. Prim's: Start from vertex, grow tree greedily
2. Kruskal's: Sort edges, add minimum without cycles
3. Both guarantee optimal MST solution
4. Different time complexities for different graph types

Prim's Algorithm:
- Vertex-centric approach
- Maintains partial MST at each step
- Adds minimum edge to expand tree
- O(V²) with array, O(E log V) with heap

Kruskal's Algorithm:
- Edge-centric approach  
- Sorts all edges by weight
- Uses Union-Find for cycle detection
- O(E log E) time complexity

MST Properties:
- Exactly V-1 edges for V vertices
- No cycles (tree property)
- Connects all vertices (spanning property)
- Minimum total weight among all spanning trees

Real-world Applications:
- Network infrastructure design
- Transportation system planning
- Circuit layout optimization
- Clustering and data analysis
- Approximation algorithms

This problem demonstrates fundamental MST algorithms
and their applications in optimization problems.
"""
