"""
1135. Connecting Cities With Minimum Cost
Difficulty: Medium

Problem:
There are n cities numbered from 1 to n.

You are given connections, where each connections[i] = [city1, city2, cost] represents the 
cost to connect city1 and city2 together. (A connection is bidirectional: connecting city1 
and city2 is the same as connecting city2 and city1.)

Return the minimum cost so that for every pair of cities, there exists a path of connections 
(possibly of length 1) directly or indirectly connecting the two cities. If the task is 
impossible, return -1.

The cost of connecting the cities is the sum of the connection costs used. Please note that 
there can be multiple connections between the same two cities.

Examples:
Input: n = 3, connections = [[1,2,5],[1,3,6],[2,3,1]]
Output: 6

Input: n = 4, connections = [[1,2,3],[3,4,4]]
Output: -1

Constraints:
- 1 <= n <= 10000
- 1 <= connections.length <= 10000
- 1 <= connections[i][0], connections[i][1] <= n
- 0 <= connections[i][2] <= 10^5
- connections[i][0] != connections[i][1]
"""

from typing import List
import heapq

class Solution:
    def minimumCost_approach1_kruskals_union_find(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 1: Kruskal's Algorithm with Union-Find (Optimal)
        
        Sort edges by cost and greedily add edges that don't create cycles.
        
        Time: O(E log E + E α(V)) where α is inverse Ackermann
        Space: O(V)
        """
        if len(connections) < n - 1:
            return -1  # Not enough edges to connect all cities
        
        # Union-Find data structure
        parent = list(range(n + 1))
        rank = [0] * (n + 1)
        
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
        
        # Sort edges by cost
        connections.sort(key=lambda x: x[2])
        
        total_cost = 0
        edges_used = 0
        
        for city1, city2, cost in connections:
            if union(city1, city2):
                total_cost += cost
                edges_used += 1
                
                if edges_used == n - 1:
                    break
        
        return total_cost if edges_used == n - 1 else -1
    
    def minimumCost_approach2_prims_algorithm(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 2: Prim's Algorithm with Min-Heap
        
        Start from city 1 and greedily add minimum cost edges to expand MST.
        
        Time: O(E log V)
        Space: O(V + E)
        """
        if len(connections) < n - 1:
            return -1
        
        # Build adjacency list
        graph = [[] for _ in range(n + 1)]
        for city1, city2, cost in connections:
            graph[city1].append((city2, cost))
            graph[city2].append((city1, cost))
        
        # Prim's algorithm starting from city 1
        visited = [False] * (n + 1)
        min_heap = [(0, 1)]  # (cost, city)
        total_cost = 0
        cities_connected = 0
        
        while min_heap and cities_connected < n:
            cost, city = heapq.heappop(min_heap)
            
            if visited[city]:
                continue
            
            # Add city to MST
            visited[city] = True
            total_cost += cost
            cities_connected += 1
            
            # Add edges to unvisited neighbors
            for neighbor, edge_cost in graph[city]:
                if not visited[neighbor]:
                    heapq.heappush(min_heap, (edge_cost, neighbor))
        
        return total_cost if cities_connected == n else -1
    
    def minimumCost_approach3_prims_with_key_array(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 3: Prim's Algorithm with Key Array
        
        Classic Prim's implementation using key array for minimum edge tracking.
        
        Time: O(V^2) or O(E log V) with heap optimization
        Space: O(V + E)
        """
        if len(connections) < n - 1:
            return -1
        
        # Build adjacency list
        graph = [[] for _ in range(n + 1)]
        for city1, city2, cost in connections:
            graph[city1].append((city2, cost))
            graph[city2].append((city1, cost))
        
        # Check if graph is connected (city 1 can reach all others)
        if not self._is_connected(n, graph):
            return -1
        
        # Prim's algorithm with key array
        visited = [False] * (n + 1)
        key = [float('inf')] * (n + 1)
        parent = [-1] * (n + 1)
        key[1] = 0
        
        total_cost = 0
        
        for _ in range(n):
            # Find minimum key unvisited vertex
            u = -1
            for v in range(1, n + 1):
                if not visited[v] and (u == -1 or key[v] < key[u]):
                    u = v
            
            # Add to MST
            visited[u] = True
            total_cost += key[u]
            
            # Update keys of adjacent vertices
            for neighbor, cost in graph[u]:
                if not visited[neighbor] and cost < key[neighbor]:
                    key[neighbor] = cost
                    parent[neighbor] = u
        
        return total_cost
    
    def minimumCost_approach4_optimized_kruskals(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 4: Optimized Kruskal's with Path Compression and Union by Rank
        
        Enhanced Union-Find with optimizations for better performance.
        
        Time: O(E log E + E α(V))
        Space: O(V)
        """
        if len(connections) < n - 1:
            return -1
        
        # Optimized Union-Find
        parent = list(range(n + 1))
        rank = [0] * (n + 1)
        components = n
        
        def find(x):
            # Path compression
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            nonlocal components
            px, py = find(x), find(y)
            
            if px == py:
                return False
            
            # Union by rank
            if rank[px] < rank[py]:
                px, py = py, px
            
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            
            components -= 1
            return True
        
        # Sort edges by cost
        connections.sort(key=lambda x: x[2])
        
        total_cost = 0
        edges_used = 0
        
        for city1, city2, cost in connections:
            if union(city1, city2):
                total_cost += cost
                edges_used += 1
                
                # Early termination
                if edges_used == n - 1:
                    break
        
        return total_cost if components == 1 else -1
    
    def minimumCost_approach5_boruvkas_algorithm(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 5: Borůvka's Algorithm (Educational)
        
        Parallel MST algorithm that finds minimum edges for each component.
        
        Time: O(E log V)
        Space: O(V + E)
        """
        if len(connections) < n - 1:
            return -1
        
        # Union-Find
        parent = list(range(n + 1))
        rank = [0] * (n + 1)
        
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
            cheapest = {}
            
            for city1, city2, cost in connections:
                root1, root2 = find(city1), find(city2)
                
                if root1 != root2:
                    # Update cheapest edge for component 1
                    if root1 not in cheapest or cost < cheapest[root1][2]:
                        cheapest[root1] = (city1, city2, cost)
                    
                    # Update cheapest edge for component 2
                    if root2 not in cheapest or cost < cheapest[root2][2]:
                        cheapest[root2] = (city1, city2, cost)
            
            # Add cheapest edges
            edges_added = 0
            added_edges = set()
            
            for city1, city2, cost in cheapest.values():
                edge = tuple(sorted([city1, city2]))
                if edge not in added_edges and union(city1, city2):
                    total_cost += cost
                    edges_added += 1
                    added_edges.add(edge)
            
            if edges_added == 0:
                break
        
        # Check if all cities are connected
        root = find(1)
        for city in range(2, n + 1):
            if find(city) != root:
                return -1
        
        return total_cost
    
    def _is_connected(self, n: int, graph: List[List[tuple]]) -> bool:
        """Helper function to check if graph is connected"""
        visited = [False] * (n + 1)
        
        def dfs(city):
            visited[city] = True
            for neighbor, _ in graph[city]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        dfs(1)
        return all(visited[1:])

def test_connecting_cities():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (n, connections, expected)
        (3, [[1,2,5],[1,3,6],[2,3,1]], 6),
        (4, [[1,2,3],[3,4,4]], -1),
        (4, [[1,2,3],[2,3,4],[3,4,5],[1,4,6]], 12),
        (5, [[1,2,1],[2,3,2],[3,4,3],[4,5,4]], 10),
        (3, [[1,2,5],[2,3,6]], 11),
        (2, [[1,2,10]], 10),
        (1, [], 0),
    ]
    
    approaches = [
        ("Kruskal's Union-Find", solution.minimumCost_approach1_kruskals_union_find),
        ("Prim's Algorithm", solution.minimumCost_approach2_prims_algorithm),
        ("Prim's Key Array", solution.minimumCost_approach3_prims_with_key_array),
        ("Optimized Kruskal's", solution.minimumCost_approach4_optimized_kruskals),
        ("Borůvka's Algorithm", solution.minimumCost_approach5_boruvkas_algorithm),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, connections, expected) in enumerate(test_cases):
            result = func(n, connections[:])  # Deep copy
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} n={n}, expected={expected}, got={result}")

def demonstrate_mst_construction():
    """Demonstrate MST construction step by step"""
    print("\n=== MST Construction Demo ===")
    
    n = 4
    connections = [[1,2,1],[2,3,2],[3,4,3],[1,4,4],[2,4,5]]
    
    print(f"Cities: {n}")
    print(f"Connections: {connections}")
    print(f"Goal: Connect all cities with minimum total cost")
    
    # Sort connections by cost
    sorted_connections = sorted(connections, key=lambda x: x[2])
    print(f"\nConnections sorted by cost: {sorted_connections}")
    
    # Demonstrate Kruskal's algorithm
    print(f"\nKruskal's Algorithm Simulation:")
    
    parent = list(range(n + 1))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        parent[py] = px
        return True
    
    total_cost = 0
    mst_edges = []
    
    for i, (city1, city2, cost) in enumerate(sorted_connections):
        root1, root2 = find(city1), find(city2)
        
        print(f"\nStep {i+1}: Consider edge ({city1}, {city2}) with cost {cost}")
        print(f"  City {city1} in component {root1}, City {city2} in component {root2}")
        
        if union(city1, city2):
            total_cost += cost
            mst_edges.append((city1, city2, cost))
            print(f"  ✓ Added to MST (components merged)")
            print(f"  Current total cost: {total_cost}")
        else:
            print(f"  ✗ Rejected (would create cycle)")
        
        if len(mst_edges) == n - 1:
            print(f"  MST complete with {n-1} edges!")
            break
    
    print(f"\nFinal MST:")
    print(f"  Edges: {mst_edges}")
    print(f"  Total cost: {total_cost}")

def demonstrate_connectivity_check():
    """Demonstrate graph connectivity verification"""
    print("\n=== Connectivity Check Demo ===")
    
    test_cases = [
        (4, [[1,2,1],[2,3,2],[3,4,3]], "Connected"),
        (4, [[1,2,1],[3,4,2]], "Disconnected"),
        (3, [[1,2,1],[1,3,2]], "Connected"),
        (5, [[1,2,1],[2,3,2],[4,5,3]], "Disconnected"),
    ]
    
    for n, connections, expected in test_cases:
        print(f"\nTest: n={n}, connections={connections}")
        
        # Build adjacency list
        graph = [[] for _ in range(n + 1)]
        for city1, city2, cost in connections:
            graph[city1].append((city2, cost))
            graph[city2].append((city1, cost))
        
        # DFS to check connectivity
        visited = [False] * (n + 1)
        
        def dfs(city):
            visited[city] = True
            for neighbor, _ in graph[city]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        dfs(1)
        is_connected = all(visited[1:])
        
        print(f"  Expected: {expected}")
        print(f"  Result: {'Connected' if is_connected else 'Disconnected'}")
        print(f"  Visited cities: {[i for i in range(1, n+1) if visited[i]]}")

def analyze_mst_complexity():
    """Analyze complexity of different MST algorithms"""
    print("\n=== MST Algorithm Complexity Analysis ===")
    
    print("Time Complexity Comparison:")
    
    print("\n1. **Kruskal's Algorithm:**")
    print("   • Sorting edges: O(E log E)")
    print("   • Union-Find operations: O(E α(V))")
    print("   • Total: O(E log E)")
    print("   • Best for sparse graphs")
    
    print("\n2. **Prim's with Binary Heap:**")
    print("   • Each vertex added once: O(V log V)")
    print("   • Each edge relaxed once: O(E log V)")
    print("   • Total: O(E log V)")
    print("   • Good balance for most graphs")
    
    print("\n3. **Prim's with Adjacency Matrix:**")
    print("   • Find minimum key: O(V) per iteration")
    print("   • V iterations: O(V²)")
    print("   • Total: O(V²)")
    print("   • Best for dense graphs")
    
    print("\n4. **Borůvka's Algorithm:**")
    print("   • Find cheapest edges: O(E) per iteration")
    print("   • O(log V) iterations")
    print("   • Total: O(E log V)")
    print("   • Good for parallel processing")
    
    print("\nSpace Complexity:")
    print("• **Kruskal's:** O(V) for Union-Find")
    print("• **Prim's heap:** O(V + E) for graph + heap")
    print("• **Prim's matrix:** O(V²) for adjacency matrix")
    print("• **Borůvka's:** O(V + E) for graph + Union-Find")
    
    print("\nPractical Guidelines:")
    print("• **Sparse graphs (E = O(V)):** Kruskal's")
    print("• **Dense graphs (E = O(V²)):** Prim's with matrix")
    print("• **Balanced graphs:** Prim's with heap")
    print("• **Parallel processing:** Borůvka's")

def analyze_real_world_scenarios():
    """Analyze real-world applications of city connection problems"""
    print("\n=== Real-World Applications ===")
    
    print("City Connection Problem Applications:")
    
    print("\n1. **Transportation Networks:**")
    print("   • Road construction planning")
    print("   • Railway line development")
    print("   • Public transit optimization")
    print("   • Minimize infrastructure cost")
    
    print("\n2. **Utility Networks:**")
    print("   • Power grid design")
    print("   • Water distribution systems")
    print("   • Natural gas pipelines")
    print("   • Telecommunications infrastructure")
    
    print("\n3. **Supply Chain:**")
    print("   • Distribution center connections")
    print("   • Manufacturing plant networks")
    print("   • Logistics hub optimization")
    print("   • Minimize transportation costs")
    
    print("\n4. **Computer Networks:**")
    print("   • Network topology design")
    print("   • Data center interconnections")
    print("   • Internet backbone planning")
    print("   • Minimize latency and cost")
    
    print("\n5. **Social Networks:**")
    print("   • Community connection analysis")
    print("   • Influence propagation paths")
    print("   • Information dissemination")
    print("   • Minimize communication barriers")
    
    print("\nPractical Considerations:")
    print("• **Budget constraints:** Hard cost limits")
    print("• **Geographic factors:** Terrain, distance limitations")
    print("• **Redundancy requirements:** Backup connections")
    print("• **Scalability:** Future expansion planning")
    print("• **Maintenance costs:** Long-term operational expenses")

if __name__ == "__main__":
    test_connecting_cities()
    demonstrate_mst_construction()
    demonstrate_connectivity_check()
    analyze_mst_complexity()
    analyze_real_world_scenarios()

"""
Minimum Spanning Tree (MST) Concepts:
1. City Connection as Classic MST Problem
2. Kruskal's Algorithm with Union-Find Optimization
3. Prim's Algorithm with Multiple Implementations
4. Graph Connectivity Verification
5. MST Algorithm Selection and Performance Analysis

Key Problem Insights:
- Connect all cities with minimum total cost
- Classic MST problem with connectivity constraint
- Multiple valid approaches with different trade-offs
- Connectivity verification essential for feasibility

Algorithm Strategy:
1. Kruskal's: Sort edges, add minimum without cycles
2. Prim's: Grow tree from single city, expand greedily
3. Borůvka's: Parallel approach finding multiple edges
4. All guarantee optimal solution when graph is connected

Implementation Considerations:
- Union-Find optimization with path compression
- Heap-based priority queue for efficient minimum finding
- Early termination when MST is complete
- Connectivity check before/after MST construction

Performance Analysis:
- Sparse graphs: Kruskal's O(E log E) preferred
- Dense graphs: Prim's O(V²) with matrix preferred
- Balanced graphs: Prim's O(E log V) with heap
- Parallel processing: Borůvka's O(E log V)

Connectivity Requirements:
- Must be able to reach all cities
- Requires at least n-1 edges for n cities
- DFS/BFS can verify connectivity
- Return -1 if connection impossible

Real-world Applications:
- Transportation network design
- Utility infrastructure planning
- Computer network topology
- Supply chain optimization
- Social network analysis

This problem demonstrates practical MST applications
in infrastructure and network design scenarios.
"""
