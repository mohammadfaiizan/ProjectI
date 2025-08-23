"""
1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance
Difficulty: Medium

Problem:
There are n cities numbered from 0 to n-1. Given the array edges where edges[i] = [fromi, toi, weighti] 
represents a bidirectional and weighted edge between cities fromi and toi, and given the integer 
distanceThreshold.

Return the city with the smallest number of cities that are reachable through some path and whose 
distance is at most distanceThreshold, If there are multiple such cities, return the city with 
the greatest number.

Examples:
Input: n = 4, edges = [[0,1,3],[1,2,1],[1,3,4],[2,3,1]], distanceThreshold = 4
Output: 3

Input: n = 5, edges = [[0,1,2],[0,4,8],[1,2,3],[1,4,2],[2,3,1],[3,4,1]], distanceThreshold = 2
Output: 0

Constraints:
- 2 <= n <= 100
- 1 <= edges.length <= n * (n - 1) / 2
- edges[i].length == 3
- 0 <= fromi < toi < n
- 1 <= weighti, distanceThreshold <= 10^4
- All pairs (fromi, toi) are distinct
"""

from typing import List
import heapq
from collections import defaultdict

class Solution:
    def findTheCity_approach1_floyd_warshall(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
        """
        Approach 1: Floyd-Warshall All-Pairs Shortest Path (Optimal)
        
        Use Floyd-Warshall to find shortest distances between all pairs of cities.
        
        Time: O(V^3)
        Space: O(V^2)
        """
        # Initialize distance matrix
        INF = float('inf')
        dist = [[INF] * n for _ in range(n)]
        
        # Distance from city to itself is 0
        for i in range(n):
            dist[i][i] = 0
        
        # Fill in direct edges
        for u, v, w in edges:
            dist[u][v] = w
            dist[v][u] = w  # Bidirectional
        
        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        # Count reachable cities for each city
        min_reachable = float('inf')
        result_city = -1
        
        for i in range(n):
            reachable_count = 0
            for j in range(n):
                if i != j and dist[i][j] <= distanceThreshold:
                    reachable_count += 1
            
            # Update result (prefer city with smaller count, then larger number)
            if reachable_count < min_reachable or (reachable_count == min_reachable and i > result_city):
                min_reachable = reachable_count
                result_city = i
        
        return result_city
    
    def findTheCity_approach2_dijkstra_from_each_city(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
        """
        Approach 2: Dijkstra from Each City
        
        Run Dijkstra's algorithm from each city to find shortest distances.
        
        Time: O(V * E log V)
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        def dijkstra(start):
            """Run Dijkstra from start city"""
            distances = [float('inf')] * n
            distances[start] = 0
            pq = [(0, start)]
            
            while pq:
                dist, u = heapq.heappop(pq)
                
                if dist > distances[u]:
                    continue
                
                for v, weight in graph[u]:
                    new_dist = dist + weight
                    if new_dist < distances[v]:
                        distances[v] = new_dist
                        heapq.heappush(pq, (new_dist, v))
            
            return distances
        
        # Find city with minimum reachable neighbors
        min_reachable = float('inf')
        result_city = -1
        
        for city in range(n):
            distances = dijkstra(city)
            reachable_count = sum(1 for i in range(n) if i != city and distances[i] <= distanceThreshold)
            
            if reachable_count < min_reachable or (reachable_count == min_reachable and city > result_city):
                min_reachable = reachable_count
                result_city = city
        
        return result_city
    
    def findTheCity_approach3_bellman_ford_from_each_city(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
        """
        Approach 3: Bellman-Ford from Each City
        
        Run Bellman-Ford algorithm from each city.
        
        Time: O(V^2 * E)
        Space: O(V + E)
        """
        def bellman_ford(start):
            """Run Bellman-Ford from start city"""
            distances = [float('inf')] * n
            distances[start] = 0
            
            # Relax edges n-1 times
            for _ in range(n - 1):
                updated = False
                for u, v, w in edges:
                    # Check both directions (bidirectional)
                    if distances[u] != float('inf') and distances[u] + w < distances[v]:
                        distances[v] = distances[u] + w
                        updated = True
                    if distances[v] != float('inf') and distances[v] + w < distances[u]:
                        distances[u] = distances[v] + w
                        updated = True
                
                if not updated:
                    break
            
            return distances
        
        # Find city with minimum reachable neighbors
        min_reachable = float('inf')
        result_city = -1
        
        for city in range(n):
            distances = bellman_ford(city)
            reachable_count = sum(1 for i in range(n) if i != city and distances[i] <= distanceThreshold)
            
            if reachable_count < min_reachable or (reachable_count == min_reachable and city > result_city):
                min_reachable = reachable_count
                result_city = city
        
        return result_city
    
    def findTheCity_approach4_spfa_from_each_city(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
        """
        Approach 4: SPFA from Each City
        
        Use SPFA (Shortest Path Faster Algorithm) from each city.
        
        Time: O(V * E) average case
        Space: O(V + E)
        """
        from collections import deque
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        def spfa(start):
            """Run SPFA from start city"""
            distances = [float('inf')] * n
            distances[start] = 0
            queue = deque([start])
            in_queue = [False] * n
            in_queue[start] = True
            
            while queue:
                u = queue.popleft()
                in_queue[u] = False
                
                for v, weight in graph[u]:
                    new_dist = distances[u] + weight
                    if new_dist < distances[v]:
                        distances[v] = new_dist
                        if not in_queue[v]:
                            queue.append(v)
                            in_queue[v] = True
            
            return distances
        
        # Find city with minimum reachable neighbors
        min_reachable = float('inf')
        result_city = -1
        
        for city in range(n):
            distances = spfa(city)
            reachable_count = sum(1 for i in range(n) if i != city and distances[i] <= distanceThreshold)
            
            if reachable_count < min_reachable or (reachable_count == min_reachable and city > result_city):
                min_reachable = reachable_count
                result_city = city
        
        return result_city

def test_find_the_city():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (n, edges, distanceThreshold, expected)
        (4, [[0,1,3],[1,2,1],[1,3,4],[2,3,1]], 4, 3),
        (5, [[0,1,2],[0,4,8],[1,2,3],[1,4,2],[2,3,1],[3,4,1]], 2, 0),
        (6, [[0,1,10],[0,2,1],[2,3,1],[1,3,1],[1,4,1],[4,5,10]], 20, 5),
        (3, [[0,1,1],[1,2,1]], 1, 2),
    ]
    
    approaches = [
        ("Floyd-Warshall", solution.findTheCity_approach1_floyd_warshall),
        ("Dijkstra from Each", solution.findTheCity_approach2_dijkstra_from_each_city),
        ("Bellman-Ford from Each", solution.findTheCity_approach3_bellman_ford_from_each_city),
        ("SPFA from Each", solution.findTheCity_approach4_spfa_from_each_city),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, edges, distanceThreshold, expected) in enumerate(test_cases):
            result = func(n, edges[:], distanceThreshold)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} n={n}, threshold={distanceThreshold}, expected={expected}, got={result}")

def demonstrate_floyd_warshall():
    """Demonstrate Floyd-Warshall algorithm step by step"""
    print("\n=== Floyd-Warshall Demo ===")
    
    n = 4
    edges = [[0,1,3],[1,2,1],[1,3,4],[2,3,1]]
    distanceThreshold = 4
    
    print(f"Graph: {n} cities, edges: {edges}")
    print(f"Distance threshold: {distanceThreshold}")
    
    # Initialize distance matrix
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
    
    for u, v, w in edges:
        dist[u][v] = w
        dist[v][u] = w
    
    print(f"\nInitial distance matrix:")
    for i in range(n):
        row = ['∞' if dist[i][j] == INF else str(dist[i][j]) for j in range(n)]
        print(f"  {i}: {row}")
    
    # Floyd-Warshall algorithm with step-by-step output
    for k in range(n):
        print(f"\nUsing intermediate vertex {k}:")
        
        updated = False
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    old_dist = dist[i][j]
                    dist[i][j] = dist[i][k] + dist[k][j]
                    print(f"  Update dist[{i}][{j}]: {old_dist} -> {dist[i][j]} (via {k})")
                    updated = True
        
        if not updated:
            print(f"  No updates in this iteration")
        
        print(f"  Distance matrix after k={k}:")
        for i in range(n):
            row = ['∞' if dist[i][j] == INF else str(dist[i][j]) for j in range(n)]
            print(f"    {i}: {row}")
    
    # Count reachable cities
    print(f"\nCounting reachable cities (threshold = {distanceThreshold}):")
    
    min_reachable = float('inf')
    result_city = -1
    
    for i in range(n):
        reachable = []
        for j in range(n):
            if i != j and dist[i][j] <= distanceThreshold:
                reachable.append(j)
        
        reachable_count = len(reachable)
        print(f"  City {i}: reachable cities {reachable}, count = {reachable_count}")
        
        if reachable_count < min_reachable or (reachable_count == min_reachable and i > result_city):
            min_reachable = reachable_count
            result_city = i
    
    print(f"\nResult: City {result_city} (minimum reachable: {min_reachable})")

def analyze_all_pairs_vs_single_source():
    """Analyze all-pairs vs single-source shortest path approaches"""
    print("\n=== All-Pairs vs Single-Source Analysis ===")
    
    print("Problem Requirements:")
    print("• Need shortest distances from each city to all other cities")
    print("• Count reachable cities within threshold for each source")
    print("• Find city with minimum reachable neighbors")
    
    print("\nApproach Comparison:")
    
    print("\n1. **Floyd-Warshall (All-Pairs):**")
    print("   • Time: O(V³)")
    print("   • Space: O(V²)")
    print("   • Computes all distances in one pass")
    print("   • Best for dense graphs or multiple queries")
    print("   • No dependency on edge list structure")
    
    print("\n2. **Dijkstra from Each City:**")
    print("   • Time: O(V × E log V)")
    print("   • Space: O(V + E)")
    print("   • V separate single-source computations")
    print("   • Best for sparse graphs")
    print("   • Can early terminate if only counting threshold distances")
    
    print("\n3. **Bellman-Ford from Each City:**")
    print("   • Time: O(V² × E)")
    print("   • Space: O(V + E)")
    print("   • Handles negative weights (not needed here)")
    print("   • Generally slower than Dijkstra for this problem")
    
    print("\n4. **SPFA from Each City:**")
    print("   • Time: O(V × E) average, O(V² × E) worst")
    print("   • Space: O(V + E)")
    print("   • Often faster than Bellman-Ford in practice")
    print("   • Good for graphs with few negative edges")
    
    print("\nOptimal Choice Analysis:")
    print("• **Small n (≤ 100):** Floyd-Warshall preferred")
    print("  - Simple implementation")
    print("  - O(V³) manageable for small V")
    print("  - No repeated graph traversals")
    
    print("\n• **Large n, sparse graph:** Dijkstra from each city")
    print("  - Better time complexity for sparse graphs")
    print("  - Can optimize with early termination")
    
    print("\n• **Implementation simplicity:** Floyd-Warshall wins")
    print("  - Single algorithm, no repeated calls")
    print("  - Easy to understand and debug")

def demonstrate_neighbor_counting():
    """Demonstrate the neighbor counting and city selection process"""
    print("\n=== Neighbor Counting Demo ===")
    
    n = 5
    edges = [[0,1,2],[0,4,8],[1,2,3],[1,4,2],[2,3,1],[3,4,1]]
    distanceThreshold = 2
    
    print(f"Graph with {n} cities, threshold = {distanceThreshold}")
    print(f"Edges: {edges}")
    
    # Use Floyd-Warshall for simplicity
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
    
    for u, v, w in edges:
        dist[u][v] = w
        dist[v][u] = w
    
    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    print(f"\nFinal distance matrix:")
    print("    ", end="")
    for j in range(n):
        print(f"{j:4}", end="")
    print()
    
    for i in range(n):
        print(f"{i}: ", end="")
        for j in range(n):
            if dist[i][j] == INF:
                print("  ∞", end="")
            else:
                print(f"{dist[i][j]:4}", end="")
        print()
    
    print(f"\nReachable neighbors analysis:")
    
    city_data = []
    for i in range(n):
        reachable = []
        for j in range(n):
            if i != j and dist[i][j] <= distanceThreshold:
                reachable.append(j)
        
        city_data.append((i, len(reachable), reachable))
        print(f"City {i}: {len(reachable)} neighbors within threshold {distanceThreshold}")
        print(f"         Reachable cities: {reachable}")
    
    # Find result with tie-breaking
    print(f"\nCity selection (prefer fewer neighbors, then higher city number):")
    
    city_data.sort(key=lambda x: (x[1], -x[0]))  # Sort by count, then by negative city number
    
    for city, count, neighbors in city_data:
        print(f"City {city}: {count} neighbors")
    
    result = city_data[0][0]
    print(f"\nSelected city: {result}")

def compare_algorithm_efficiency():
    """Compare efficiency of different algorithms for this specific problem"""
    print("\n=== Algorithm Efficiency Comparison ===")
    
    print("For the City Selection Problem:")
    
    print("\n**Graph Size Analysis:**")
    print("• Constraint: n ≤ 100")
    print("• Floyd-Warshall: 100³ = 1,000,000 operations")
    print("• Dijkstra V times: 100 × E log 100 ≈ 100 × E × 7")
    print("• Crossover point: E ≈ 14,000 (nearly complete graph)")
    
    print("\n**Practical Considerations:**")
    print("• Real graphs typically sparse (E << V²)")
    print("• Floyd-Warshall has better constant factors")
    print("• Implementation simplicity favors Floyd-Warshall")
    print("• No early termination benefits for this problem")
    
    print("\n**Memory Usage:**")
    print("• Floyd-Warshall: O(V²) = 10,000 integers")
    print("• Dijkstra: O(V + E) per call, but temporary")
    print("• For n=100, memory differences negligible")
    
    print("\n**Code Complexity:**")
    print("• Floyd-Warshall: 3 nested loops")
    print("• Dijkstra: Priority queue + graph traversal × V")
    print("• Debugging and maintenance easier with Floyd-Warshall")
    
    print("\n**Recommendation for this problem:**")
    print("✅ **Floyd-Warshall** for:")
    print("   - n ≤ 100 (given constraint)")
    print("   - Need all-pairs distances anyway")
    print("   - Simple implementation")
    print("   - Predictable performance")
    
    print("\n🔄 **Dijkstra** might be better for:")
    print("   - Very sparse graphs (E << V²)")
    print("   - Larger graphs (if constraint relaxed)")
    print("   - Memory-constrained environments")
    
    print("\nReal-world Applications:")
    print("• **Urban planning:** Find central locations")
    print("• **Supply chain:** Minimize distribution points")
    print("• **Network design:** Reduce connection complexity")
    print("• **Emergency services:** Optimize coverage areas")
    print("• **Social networks:** Find influential nodes")

if __name__ == "__main__":
    test_find_the_city()
    demonstrate_floyd_warshall()
    analyze_all_pairs_vs_single_source()
    demonstrate_neighbor_counting()
    compare_algorithm_efficiency()

"""
Shortest Path Concepts:
1. All-Pairs Shortest Path with Floyd-Warshall
2. Single-Source Shortest Path from Multiple Sources
3. Threshold-based Reachability Analysis
4. City Selection with Tie-breaking Rules
5. Dense vs Sparse Graph Algorithm Selection

Key Problem Insights:
- Need shortest distances from each city to all others
- Count reachable cities within distance threshold
- Select city with minimum reachable neighbors
- Tie-breaking: prefer higher-numbered city

Algorithm Strategy:
1. Compute all-pairs shortest distances
2. For each city, count neighbors within threshold
3. Find city with minimum neighbor count
4. Apply tie-breaking rule for equal counts

Floyd-Warshall Algorithm:
- Time: O(V³) for all-pairs shortest paths
- Space: O(V²) for distance matrix
- Three nested loops with intermediate vertices
- Handles bidirectional edges naturally

Alternative Approaches:
- Dijkstra from each city: O(V × E log V)
- Bellman-Ford from each city: O(V² × E)
- SPFA from each city: O(V × E) average

Performance Analysis:
- Small n (≤100): Floyd-Warshall optimal
- Implementation simplicity favors Floyd-Warshall
- Dense graphs: Floyd-Warshall preferred
- Sparse graphs: Consider Dijkstra alternative

Real-world Applications:
- Urban planning and facility location
- Network topology optimization
- Supply chain distribution centers
- Emergency service coverage
- Social network influence analysis

This problem demonstrates all-pairs shortest path
algorithms for centrality and reachability analysis.
"""
