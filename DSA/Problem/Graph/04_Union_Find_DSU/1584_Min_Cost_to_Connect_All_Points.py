"""
1584. Min Cost to Connect All Points
Difficulty: Medium

Problem:
You are given an array points representing integer coordinates of some points on a 2D-plane, 
where points[i] = [xi, yi].

The cost of connecting two points [xi, yi] and [xj, yj] is the manhattan distance between 
them: |xi - xj| + |yi - yj|.

Return the minimum cost to make all points connected. All points are connected if there is 
exactly one simple path between any two points.

Examples:
Input: points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
Output: 20

Input: points = [[3,12],[-2,5],[-4,1]]
Output: 18

Input: points = [[0,0],[1,1],[1,0],[-1,1]]
Output: 4

Constraints:
- 1 <= points.length <= 1000
- -10^6 <= xi, yi <= 10^6
- All pairs (xi, yi) are distinct
"""

from typing import List
import heapq

class UnionFind:
    """Union-Find with path compression and union by rank"""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
    def find(self, x):
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank, returns True if union performed"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.components -= 1
        return True
    
    def is_connected(self):
        """Check if all points are connected (1 component)"""
        return self.components == 1

class Solution:
    def minCostConnectPoints_approach1_kruskals(self, points: List[List[int]]) -> int:
        """
        Approach 1: Kruskal's Algorithm with Union-Find (Optimal)
        
        Generate all edges, sort by weight, use Union-Find for MST.
        
        Time: O(N^2 log N) - N^2 edges, sort dominates
        Space: O(N^2) - store all edges
        """
        n = len(points)
        if n <= 1:
            return 0
        
        def manhattan_distance(p1, p2):
            """Calculate Manhattan distance between two points"""
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # Generate all edges with weights
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = manhattan_distance(points[i], points[j])
                edges.append((dist, i, j))
        
        # Sort edges by weight (Kruskal's requirement)
        edges.sort()
        
        # Kruskal's algorithm using Union-Find
        uf = UnionFind(n)
        mst_cost = 0
        edges_used = 0
        
        for weight, u, v in edges:
            if uf.union(u, v):
                mst_cost += weight
                edges_used += 1
                
                # MST has exactly n-1 edges
                if edges_used == n - 1:
                    break
        
        return mst_cost
    
    def minCostConnectPoints_approach2_prims_heap(self, points: List[List[int]]) -> int:
        """
        Approach 2: Prim's Algorithm with Priority Queue
        
        Start from arbitrary point, grow MST by adding minimum edge.
        
        Time: O(N^2 log N)
        Space: O(N^2)
        """
        n = len(points)
        if n <= 1:
            return 0
        
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # Start with point 0
        visited = [False] * n
        min_heap = [(0, 0)]  # (cost, point_index)
        mst_cost = 0
        
        while min_heap:
            cost, u = heapq.heappop(min_heap)
            
            if visited[u]:
                continue
            
            visited[u] = True
            mst_cost += cost
            
            # Add all edges from u to unvisited points
            for v in range(n):
                if not visited[v]:
                    dist = manhattan_distance(points[u], points[v])
                    heapq.heappush(min_heap, (dist, v))
        
        return mst_cost
    
    def minCostConnectPoints_approach3_prims_array(self, points: List[List[int]]) -> int:
        """
        Approach 3: Prim's Algorithm with Array
        
        Use array to track minimum distances, more space efficient.
        
        Time: O(N^2)
        Space: O(N)
        """
        n = len(points)
        if n <= 1:
            return 0
        
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # Track minimum cost to connect each point to MST
        min_cost = [float('inf')] * n
        visited = [False] * n
        min_cost[0] = 0
        
        mst_cost = 0
        
        for _ in range(n):
            # Find unvisited point with minimum cost
            u = -1
            for i in range(n):
                if not visited[i] and (u == -1 or min_cost[i] < min_cost[u]):
                    u = i
            
            visited[u] = True
            mst_cost += min_cost[u]
            
            # Update minimum costs for neighbors
            for v in range(n):
                if not visited[v]:
                    dist = manhattan_distance(points[u], points[v])
                    min_cost[v] = min(min_cost[v], dist)
        
        return mst_cost
    
    def minCostConnectPoints_approach4_optimized_kruskals(self, points: List[List[int]]) -> int:
        """
        Approach 4: Optimized Kruskal's with Early Termination
        
        Add optimizations for better average case performance.
        
        Time: O(N^2 log N)
        Space: O(N^2)
        """
        n = len(points)
        if n <= 1:
            return 0
        
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # Generate and sort edges
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = manhattan_distance(points[i], points[j])
                edges.append((dist, i, j))
        
        edges.sort()
        
        # Union-Find with optimizations
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            root_x, root_y = find(x), find(y)
            if root_x == root_y:
                return False
            
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1
            
            return True
        
        mst_cost = 0
        edges_used = 0
        
        for weight, u, v in edges:
            if union(u, v):
                mst_cost += weight
                edges_used += 1
                
                if edges_used == n - 1:
                    break
        
        return mst_cost

def test_min_cost_connect_points():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (points, expected)
        ([[0,0],[2,2],[3,10],[5,2],[7,0]], 20),
        ([[3,12],[-2,5],[-4,1]], 18),
        ([[0,0],[1,1],[1,0],[-1,1]], 4),
        ([[0,0]], 0),
        ([[0,0],[1,0]], 1),
        ([[2,-3],[-17,-8],[13,8],[-17,-15]], 53),
    ]
    
    approaches = [
        ("Kruskal's Algorithm", solution.minCostConnectPoints_approach1_kruskals),
        ("Prim's with Heap", solution.minCostConnectPoints_approach2_prims_heap),
        ("Prim's with Array", solution.minCostConnectPoints_approach3_prims_array),
        ("Optimized Kruskal's", solution.minCostConnectPoints_approach4_optimized_kruskals),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (points, expected) in enumerate(test_cases):
            result = func(points[:])  # Copy to avoid modification
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Points: {len(points)}, Expected: {expected}, Got: {result}")

def demonstrate_kruskals_algorithm():
    """Demonstrate Kruskal's algorithm process"""
    print("\n=== Kruskal's Algorithm Demo ===")
    
    points = [[0,0],[2,2],[3,10],[5,2]]
    print(f"Points: {points}")
    
    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    # Generate all edges
    n = len(points)
    edges = []
    
    print(f"\nGenerating all edges:")
    for i in range(n):
        for j in range(i + 1, n):
            dist = manhattan_distance(points[i], points[j])
            edges.append((dist, i, j))
            print(f"  Edge ({i}, {j}): distance = {dist}")
    
    # Sort edges
    edges.sort()
    print(f"\nSorted edges by weight:")
    for weight, i, j in edges:
        print(f"  ({i}, {j}): {weight}")
    
    # Kruskal's algorithm
    print(f"\nKruskal's algorithm execution:")
    
    uf = UnionFind(n)
    mst_cost = 0
    mst_edges = []
    
    for weight, u, v in edges:
        print(f"\nConsidering edge ({u}, {v}) with weight {weight}")
        
        if uf.find(u) == uf.find(v):
            print(f"  ❌ Rejected: Would create cycle (same component)")
        else:
            print(f"  ✅ Accepted: Connects different components")
            uf.union(u, v)
            mst_cost += weight
            mst_edges.append((u, v, weight))
            
            # Show current components
            components = {}
            for point in range(n):
                root = uf.find(point)
                if root not in components:
                    components[root] = []
                components[root].append(point)
            
            print(f"  Current components: {list(components.values())}")
            print(f"  MST cost so far: {mst_cost}")
            
            if len(mst_edges) == n - 1:
                print(f"  🎯 MST complete! ({n-1} edges added)")
                break
    
    print(f"\nFinal MST:")
    print(f"  Edges: {mst_edges}")
    print(f"  Total cost: {mst_cost}")

def demonstrate_prims_algorithm():
    """Demonstrate Prim's algorithm process"""
    print("\n=== Prim's Algorithm Demo ===")
    
    points = [[0,0],[2,2],[3,10]]
    print(f"Points: {points}")
    
    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    n = len(points)
    visited = [False] * n
    min_heap = [(0, 0)]  # Start from point 0
    mst_cost = 0
    mst_edges = []
    
    print(f"\nPrim's algorithm execution:")
    print(f"Starting from point 0: {points[0]}")
    
    step = 0
    while min_heap:
        step += 1
        cost, u = heapq.heappop(min_heap)
        
        if visited[u]:
            print(f"\nStep {step}: Point {u} already visited, skipping")
            continue
        
        print(f"\nStep {step}: Adding point {u} to MST (cost: {cost})")
        visited[u] = True
        mst_cost += cost
        
        if cost > 0:  # Don't record the starting point
            mst_edges.append((u, cost))
        
        print(f"  Current MST cost: {mst_cost}")
        print(f"  Visited points: {[i for i in range(n) if visited[i]]}")
        
        # Add edges to unvisited points
        new_edges = []
        for v in range(n):
            if not visited[v]:
                dist = manhattan_distance(points[u], points[v])
                heapq.heappush(min_heap, (dist, v))
                new_edges.append((v, dist))
        
        if new_edges:
            print(f"  Added to heap: {new_edges}")
        
        # Show current heap state
        heap_contents = sorted(min_heap)
        unvisited_heap = [(cost, point) for cost, point in heap_contents if not visited[point]]
        if unvisited_heap:
            print(f"  Current heap (unvisited): {unvisited_heap}")
    
    print(f"\nFinal MST cost: {mst_cost}")

def analyze_mst_algorithms():
    """Analyze different MST algorithms"""
    print("\n=== MST Algorithms Analysis ===")
    
    print("1. **Kruskal's Algorithm:**")
    print("   • Edge-based approach")
    print("   • Sort all edges by weight")
    print("   • Use Union-Find to detect cycles")
    print("   • Time: O(E log E) = O(N² log N) for complete graph")
    print("   • Space: O(E) = O(N²) for edge storage")
    print("   • Best for sparse graphs")
    
    print("\n2. **Prim's Algorithm (Heap):**")
    print("   • Vertex-based approach")
    print("   • Grow MST from arbitrary starting vertex")
    print("   • Use priority queue for minimum edge selection")
    print("   • Time: O(E log V) = O(N² log N)")
    print("   • Space: O(V) = O(N) for heap")
    print("   • Good for dense graphs")
    
    print("\n3. **Prim's Algorithm (Array):**")
    print("   • Linear search for minimum edge")
    print("   • No heap overhead")
    print("   • Time: O(V²) = O(N²)")
    print("   • Space: O(V) = O(N)")
    print("   • Optimal for dense graphs")
    
    print("\nManhattan Distance Properties:")
    print("• Distance = |x₁ - x₂| + |y₁ - y₂|")
    print("• Also known as L₁ distance or taxicab distance")
    print("• Forms diamond-shaped distance contours")
    print("• Commonly used in grid-based problems")
    print("• Different from Euclidean distance")
    
    print("\nProblem-Specific Insights:")
    print("• Complete graph: all points connected to all others")
    print("• N points → N(N-1)/2 edges = O(N²) edges")
    print("• MST always has exactly N-1 edges")
    print("• Unique MST when all edge weights are distinct")
    print("• Multiple MSTs possible with equal weights")

def compare_algorithm_performance():
    """Compare performance characteristics of different algorithms"""
    print("\n=== Algorithm Performance Comparison ===")
    
    print("Time Complexity Analysis:")
    print("• **Kruskal's:** O(N² log N)")
    print("  - Generate N² edges: O(N²)")
    print("  - Sort edges: O(N² log N)")
    print("  - Union-Find operations: O(N² α(N)) ≈ O(N²)")
    print("  - Dominated by sorting")
    
    print("\n• **Prim's (Heap):** O(N² log N)")
    print("  - Each vertex added once: O(N)")
    print("  - Each edge considered once: O(N²)")
    print("  - Heap operations: O(log N)")
    print("  - Total: O(N² log N)")
    
    print("\n• **Prim's (Array):** O(N²)")
    print("  - Each vertex added once: O(N)")
    print("  - Linear scan for minimum: O(N)")
    print("  - Total: O(N²)")
    print("  - Most efficient for dense graphs")
    
    print("\nSpace Complexity:")
    print("• **Kruskal's:** O(N²) - store all edges")
    print("• **Prim's (Heap):** O(N²) - heap can contain O(N²) edges")
    print("• **Prim's (Array):** O(N) - only distance array")
    
    print("\nWhen to Use Each:")
    print("• **Kruskal's:** Sparse graphs, need edge-based MST")
    print("• **Prim's (Heap):** General purpose, moderate density")
    print("• **Prim's (Array):** Dense graphs, memory constraints")
    
    print("\nReal-world Applications:")
    print("• **Network Design:** Minimum cost to connect all nodes")
    print("• **Circuit Design:** Minimum wire length connections")
    print("• **Clustering:** Connect similar data points")
    print("• **Image Processing:** Region connectivity")
    print("• **Game Development:** Procedural map generation")

if __name__ == "__main__":
    test_min_cost_connect_points()
    demonstrate_kruskals_algorithm()
    demonstrate_prims_algorithm()
    analyze_mst_algorithms()
    compare_algorithm_performance()

"""
Union-Find Concepts:
1. Minimum Spanning Tree (MST) Construction
2. Kruskal's Algorithm Implementation
3. Cycle Detection in MST Building
4. Manhattan Distance in 2D Space

Key Problem Insights:
- Connect all points with minimum total cost
- Manhattan distance: |x₁-x₂| + |y₁-y₂|
- MST has exactly N-1 edges for N points
- Kruskal's + Union-Find prevents cycles

Algorithm Strategy:
1. Generate all possible edges with weights
2. Sort edges by Manhattan distance
3. Use Union-Find to detect cycles
4. Add edges that don't create cycles

Kruskal's Algorithm Steps:
1. Create edge list with distances
2. Sort edges by weight (ascending)
3. Process edges in order
4. Use Union-Find to check connectivity
5. Add edge if it connects different components

Real-world Applications:
- Network infrastructure design
- Circuit board layout optimization
- Transportation route planning
- Facility location problems
- Clustering and data analysis

This problem demonstrates Union-Find in
Minimum Spanning Tree construction algorithms.
"""
