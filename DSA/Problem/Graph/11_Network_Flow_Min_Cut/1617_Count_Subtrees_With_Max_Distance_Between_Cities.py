"""
1617. Count Subtrees With Max Distance Between Cities - Multiple Approaches
Difficulty: Hard

There are n cities numbered from 1 to n. You are given the array edges where 
edges[i] = [ui, vi] represents a bidirectional edge between cities ui and vi. 
There exists a unique path between each pair of cities. In other words, the 
cities form a tree.

A subtree is a subset of cities where every city is reachable from every other 
city in the subset, where the path between each pair passes through only the 
cities in the subset. Two subtrees are different if there is a city in one 
subtree and not in the other.

For each d from 1 to n-1, find the number of subtrees in which the maximum 
distance between any two cities in the subtree is equal to d.

Return an array of size n-1 where the answer[i] is the number of subtrees in 
which the maximum distance between any two cities is equal to i+1.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import itertools

class CountSubtreesMaxDistance:
    """Multiple approaches to count subtrees with maximum distance"""
    
    def countSubgraphsForEachDiameter_brute_force(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 1: Brute Force with All Subsets
        
        Check all possible subsets and verify if they form connected subtrees.
        
        Time: O(2^n * n^2)
        Space: O(n^2)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def is_connected_subtree(cities: Set[int]) -> bool:
            """Check if cities form a connected subtree"""
            if len(cities) <= 1:
                return True
            
            # BFS to check connectivity within subset
            start = next(iter(cities))
            visited = {start}
            queue = deque([start])
            
            while queue:
                current = queue.popleft()
                for neighbor in graph[current]:
                    if neighbor in cities and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            return len(visited) == len(cities)
        
        def get_max_distance(cities: Set[int]) -> int:
            """Get maximum distance between any two cities in subset"""
            if len(cities) <= 1:
                return 0
            
            max_dist = 0
            cities_list = list(cities)
            
            for i in range(len(cities_list)):
                # BFS from each city to find distances
                distances = {cities_list[i]: 0}
                queue = deque([cities_list[i]])
                
                while queue:
                    current = queue.popleft()
                    for neighbor in graph[current]:
                        if neighbor in cities and neighbor not in distances:
                            distances[neighbor] = distances[current] + 1
                            queue.append(neighbor)
                
                # Update max distance
                for city in cities_list:
                    if city in distances:
                        max_dist = max(max_dist, distances[city])
            
            return max_dist
        
        result = [0] * (n - 1)
        
        # Check all possible subsets of cities
        for size in range(2, n + 1):
            for subset in itertools.combinations(range(1, n + 1), size):
                cities = set(subset)
                
                if is_connected_subtree(cities):
                    max_dist = get_max_distance(cities)
                    if 1 <= max_dist <= n - 1:
                        result[max_dist - 1] += 1
        
        return result
    
    def countSubgraphsForEachDiameter_optimized_bfs(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 2: Optimized BFS with Distance Calculation
        
        More efficient distance calculation using single BFS per subset.
        
        Time: O(2^n * n^2)
        Space: O(n^2)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def get_diameter_and_check_connectivity(cities: Set[int]) -> int:
            """Get diameter and check connectivity in one pass"""
            if len(cities) <= 1:
                return 0 if len(cities) == 1 else -1
            
            cities_list = list(cities)
            max_diameter = 0
            
            # Try each city as starting point
            for start in cities_list:
                distances = {start: 0}
                queue = deque([start])
                max_dist_from_start = 0
                
                while queue:
                    current = queue.popleft()
                    for neighbor in graph[current]:
                        if neighbor in cities and neighbor not in distances:
                            distances[neighbor] = distances[current] + 1
                            max_dist_from_start = max(max_dist_from_start, distances[neighbor])
                            queue.append(neighbor)
                
                # Check if all cities are reachable
                if len(distances) != len(cities):
                    return -1  # Not connected
                
                max_diameter = max(max_diameter, max_dist_from_start)
            
            return max_diameter
        
        result = [0] * (n - 1)
        
        # Check all possible subsets
        for mask in range(1, 1 << n):
            cities = set()
            for i in range(n):
                if mask & (1 << i):
                    cities.add(i + 1)
            
            if len(cities) >= 2:
                diameter = get_diameter_and_check_connectivity(cities)
                if 1 <= diameter <= n - 1:
                    result[diameter - 1] += 1
        
        return result
    
    def countSubgraphsForEachDiameter_tree_diameter(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 3: Tree Diameter Algorithm
        
        Use standard tree diameter algorithm (two BFS) for each subset.
        
        Time: O(2^n * n)
        Space: O(n^2)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def find_diameter(cities: Set[int]) -> int:
            """Find diameter using two BFS approach"""
            if len(cities) <= 1:
                return 0 if len(cities) == 1 else -1
            
            def bfs_farthest(start: int) -> Tuple[int, int]:
                """BFS to find farthest node and distance"""
                distances = {start: 0}
                queue = deque([start])
                farthest_node = start
                max_distance = 0
                
                while queue:
                    current = queue.popleft()
                    for neighbor in graph[current]:
                        if neighbor in cities and neighbor not in distances:
                            distances[neighbor] = distances[current] + 1
                            if distances[neighbor] > max_distance:
                                max_distance = distances[neighbor]
                                farthest_node = neighbor
                            queue.append(neighbor)
                
                return farthest_node, max_distance
            
            # First BFS from arbitrary node
            start = next(iter(cities))
            endpoint1, _ = bfs_farthest(start)
            
            # Second BFS from farthest node found
            endpoint2, diameter = bfs_farthest(endpoint1)
            
            # Check connectivity
            distances = {endpoint1: 0}
            queue = deque([endpoint1])
            
            while queue:
                current = queue.popleft()
                for neighbor in graph[current]:
                    if neighbor in cities and neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
            
            if len(distances) != len(cities):
                return -1  # Not connected
            
            return diameter
        
        result = [0] * (n - 1)
        
        # Check all possible subsets
        for mask in range(1, 1 << n):
            cities = set()
            for i in range(n):
                if mask & (1 << i):
                    cities.add(i + 1)
            
            if len(cities) >= 2:
                diameter = find_diameter(cities)
                if 1 <= diameter <= n - 1:
                    result[diameter - 1] += 1
        
        return result
    
    def countSubgraphsForEachDiameter_dp_bitmask(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 4: Dynamic Programming with Bitmask
        
        Use DP to build connected components incrementally.
        
        Time: O(3^n)
        Space: O(2^n)
        """
        # Build adjacency list and adjacency matrix
        graph = defaultdict(list)
        adj_matrix = [[False] * (n + 1) for _ in range(n + 1)]
        
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
            adj_matrix[u][v] = adj_matrix[v][u] = True
        
        # DP: dp[mask] = True if mask represents a connected subtree
        dp = [False] * (1 << n)
        dp[0] = True
        
        # Single nodes are connected
        for i in range(n):
            dp[1 << i] = True
        
        def get_diameter(mask: int) -> int:
            """Calculate diameter of connected component represented by mask"""
            cities = []
            for i in range(n):
                if mask & (1 << i):
                    cities.append(i + 1)
            
            if len(cities) <= 1:
                return 0
            
            # Use Floyd-Warshall on subset
            dist = [[float('inf')] * len(cities) for _ in range(len(cities))]
            
            for i in range(len(cities)):
                dist[i][i] = 0
                for j in range(len(cities)):
                    if adj_matrix[cities[i]][cities[j]]:
                        dist[i][j] = 1
            
            for k in range(len(cities)):
                for i in range(len(cities)):
                    for j in range(len(cities)):
                        if dist[i][k] + dist[k][j] < dist[i][j]:
                            dist[i][j] = dist[i][k] + dist[k][j]
            
            max_dist = 0
            for i in range(len(cities)):
                for j in range(len(cities)):
                    if dist[i][j] != float('inf'):
                        max_dist = max(max_dist, dist[i][j])
            
            return max_dist
        
        # Build connected components
        for mask in range(1 << n):
            if not dp[mask]:
                continue
            
            # Try adding each adjacent node
            for i in range(n):
                if mask & (1 << i):
                    for neighbor in graph[i + 1]:
                        neighbor_bit = 1 << (neighbor - 1)
                        if not (mask & neighbor_bit):
                            new_mask = mask | neighbor_bit
                            dp[new_mask] = True
        
        result = [0] * (n - 1)
        
        # Count subtrees by diameter
        for mask in range(1, 1 << n):
            if dp[mask] and bin(mask).count('1') >= 2:
                diameter = get_diameter(mask)
                if 1 <= diameter <= n - 1:
                    result[diameter - 1] += 1
        
        return result
    
    def countSubgraphsForEachDiameter_optimized_connectivity(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 5: Optimized Connectivity Check
        
        Efficient connectivity checking with early termination.
        
        Time: O(2^n * n^2)
        Space: O(n^2)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def is_connected_and_get_diameter(cities: List[int]) -> int:
            """Check connectivity and get diameter efficiently"""
            if len(cities) <= 1:
                return 0 if len(cities) == 1 else -1
            
            city_set = set(cities)
            
            # Check if subset forms a connected component
            visited = set()
            queue = deque([cities[0]])
            visited.add(cities[0])
            
            while queue:
                current = queue.popleft()
                for neighbor in graph[current]:
                    if neighbor in city_set and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            if len(visited) != len(cities):
                return -1  # Not connected
            
            # Find diameter using two BFS
            def bfs_distances(start: int) -> Dict[int, int]:
                distances = {start: 0}
                queue = deque([start])
                
                while queue:
                    current = queue.popleft()
                    for neighbor in graph[current]:
                        if neighbor in city_set and neighbor not in distances:
                            distances[neighbor] = distances[current] + 1
                            queue.append(neighbor)
                
                return distances
            
            # First BFS to find one end of diameter
            distances1 = bfs_distances(cities[0])
            farthest1 = max(distances1.keys(), key=lambda x: distances1[x])
            
            # Second BFS to find actual diameter
            distances2 = bfs_distances(farthest1)
            diameter = max(distances2.values())
            
            return diameter
        
        result = [0] * (n - 1)
        
        # Check all subsets
        for mask in range(1, 1 << n):
            cities = []
            for i in range(n):
                if mask & (1 << i):
                    cities.append(i + 1)
            
            if len(cities) >= 2:
                diameter = is_connected_and_get_diameter(cities)
                if 1 <= diameter <= n - 1:
                    result[diameter - 1] += 1
        
        return result

def test_count_subtrees_max_distance():
    """Test all approaches with various test cases"""
    solver = CountSubtreesMaxDistance()
    
    test_cases = [
        # (n, edges, expected, description)
        (4, [[1,2],[2,3],[2,4]], [3,4,0], "Star with 4 nodes"),
        (2, [[1,2]], [1], "Simple edge"),
        (3, [[1,2],[2,3]], [2,1], "Path of 3 nodes"),
        (6, [[1,2],[1,3],[2,4],[3,5],[3,6]], [6,4,1,0,0], "Tree with 6 nodes"),
    ]
    
    approaches = [
        ("Brute Force", solver.countSubgraphsForEachDiameter_brute_force),
        ("Optimized BFS", solver.countSubgraphsForEachDiameter_optimized_bfs),
        ("Tree Diameter", solver.countSubgraphsForEachDiameter_tree_diameter),
        ("DP Bitmask", solver.countSubgraphsForEachDiameter_dp_bitmask),
        ("Optimized Connectivity", solver.countSubgraphsForEachDiameter_optimized_connectivity),
    ]
    
    print("=== Testing Count Subtrees With Max Distance ===")
    
    for n, edges, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"n={n}, edges={edges}")
        print(f"Expected: {expected}")
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(n, edges)
                status = "✓" if result == expected else "✗"
                print(f"{approach_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{approach_name:20} | ERROR: {str(e)[:30]}")

def demonstrate_subtree_analysis():
    """Demonstrate subtree counting and analysis"""
    print("\n=== Subtree Analysis Demo ===")
    
    # Example tree
    n = 4
    edges = [[1,2],[2,3],[2,4]]
    
    print(f"Tree structure: n={n}, edges={edges}")
    print("Tree visualization:")
    print("    1")
    print("    |")
    print("    2")
    print("   / \\")
    print("  3   4")
    
    solver = CountSubtreesMaxDistance()
    result = solver.countSubgraphsForEachDiameter_tree_diameter(n, edges)
    
    print(f"\nSubtree count by diameter: {result}")
    
    print(f"\nAnalysis:")
    print(f"• Diameter 1: {result[0]} subtrees (adjacent pairs)")
    print(f"• Diameter 2: {result[1]} subtrees (paths of length 2)")
    print(f"• Diameter 3: {result[2]} subtrees (paths of length 3)")
    
    print(f"\nSubtree enumeration:")
    print(f"Diameter 1: {{1,2}}, {{2,3}}, {{2,4}} = 3 subtrees")
    print(f"Diameter 2: {{1,2,3}}, {{1,2,4}}, {{2,3,4}}, {{1,2,3,4}} = 4 subtrees")
    print(f"Diameter 3: None (maximum possible is 3)")

def analyze_algorithm_complexity():
    """Analyze complexity of different approaches"""
    print("\n=== Algorithm Complexity Analysis ===")
    
    print("Approach Comparison:")
    
    print("\n1. **Brute Force:**")
    print("   • Time: O(2^n * n^2) - check all subsets")
    print("   • Space: O(n^2) - adjacency list and BFS")
    print("   • Pros: Simple and straightforward")
    print("   • Cons: Exponential time complexity")
    
    print("\n2. **Optimized BFS:**")
    print("   • Time: O(2^n * n^2) - same complexity")
    print("   • Space: O(n^2)")
    print("   • Pros: Better constant factors")
    print("   • Cons: Still exponential")
    
    print("\n3. **Tree Diameter:**")
    print("   • Time: O(2^n * n) - two BFS per subset")
    print("   • Space: O(n^2)")
    print("   • Pros: Optimal diameter calculation")
    print("   • Cons: Still exponential in n")
    
    print("\n4. **DP Bitmask:**")
    print("   • Time: O(3^n) - better for dense connections")
    print("   • Space: O(2^n) - DP table")
    print("   • Pros: Avoids redundant connectivity checks")
    print("   • Cons: High space complexity")
    
    print("\n5. **Optimized Connectivity:**")
    print("   • Time: O(2^n * n^2)")
    print("   • Space: O(n^2)")
    print("   • Pros: Early termination optimizations")
    print("   • Cons: Worst-case still exponential")
    
    print("\nPractical Considerations:")
    print("• n ≤ 15: All approaches feasible")
    print("• n > 15: Need approximation or pruning")
    print("• Tree structure: Can enable optimizations")
    print("• Sparse vs dense: Affects algorithm choice")

def demonstrate_tree_properties():
    """Demonstrate tree properties relevant to the problem"""
    print("\n=== Tree Properties Demo ===")
    
    print("Key Tree Properties:")
    
    print("\n1. **Connectivity:**")
    print("   • Tree has exactly n-1 edges")
    print("   • Unique path between any two vertices")
    print("   • Removing any edge disconnects the tree")
    print("   • Adding any edge creates exactly one cycle")
    
    print("\n2. **Diameter Properties:**")
    print("   • Diameter = longest path in tree")
    print("   • Can be found using two BFS calls")
    print("   • Subtree diameter ≤ original tree diameter")
    print("   • Diameter endpoints are always leaves")
    
    print("\n3. **Subtree Characteristics:**")
    print("   • Subtree must be connected subset")
    print("   • Induced subgraph must be a tree")
    print("   • Number of subtrees = 2^n - 1 (excluding empty)")
    print("   • Connected subtrees ≤ total subtrees")
    
    print("\n4. **Counting Strategy:**")
    print("   • Enumerate all possible subsets")
    print("   • Check connectivity for each subset")
    print("   • Calculate diameter of connected subtrees")
    print("   • Group by diameter value")
    
    print("\n5. **Optimization Opportunities:**")
    print("   • Prune disconnected subsets early")
    print("   • Use tree structure for efficient diameter calculation")
    print("   • Dynamic programming for overlapping subproblems")
    print("   • Bit manipulation for efficient subset enumeration")

if __name__ == "__main__":
    test_count_subtrees_max_distance()
    demonstrate_subtree_analysis()
    analyze_algorithm_complexity()
    demonstrate_tree_properties()

"""
Count Subtrees With Max Distance - Key Insights:

1. **Problem Structure:**
   - Tree with n cities and n-1 edges
   - Find subtrees with specific diameter values
   - Subtree must be connected subset of vertices
   - Diameter = maximum distance between any two vertices

2. **Algorithm Categories:**
   - Brute Force: Check all subsets explicitly
   - Optimized: Better connectivity and diameter calculation
   - Dynamic Programming: Build connected components incrementally
   - Tree-specific: Leverage tree properties for efficiency

3. **Key Challenges:**
   - Exponential number of subsets (2^n)
   - Connectivity checking for each subset
   - Efficient diameter calculation
   - Avoiding redundant computations

4. **Optimization Techniques:**
   - Two-BFS diameter algorithm for trees
   - Early termination for disconnected subsets
   - Bit manipulation for subset enumeration
   - Dynamic programming for connected components

5. **Complexity Analysis:**
   - Time: O(2^n * n^2) for most approaches
   - Space: O(n^2) for adjacency representation
   - Practical limit: n ≤ 15 for exact algorithms
   - Approximation needed for larger instances

The problem combines tree algorithms with combinatorial
enumeration, requiring efficient subset processing.
"""
