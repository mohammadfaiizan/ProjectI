"""
787. Cheapest Flights Within K Stops
Difficulty: Medium

Problem:
There are n cities connected by some number of flights. You are given an array flights 
where flights[i] = [fromi, toi, pricei] indicates that there is a flight from city fromi 
to city toi with cost pricei.

You are also given three integers src, dst, and k, return the cheapest price from src 
to dst with at most k stops. If there is no such route, return -1.

Examples:
Input: n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], src = 0, dst = 3, k = 1
Output: 700

Input: n = 3, flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 1
Output: 200

Input: n = 3, flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 0
Output: 500

Constraints:
- 1 <= n <= 100
- 0 <= flights.length <= (n * (n - 1) / 2)
- flights[i].length == 3
- 0 <= fromi, toi < n
- fromi != toi
- 1 <= pricei <= 10^4
- There will not be any multiple flights between two cities
- 0 <= src, dst, k < n
- src != dst
"""

from typing import List
import heapq
from collections import defaultdict, deque

class Solution:
    def findCheapestPrice_approach1_modified_dijkstra(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        """
        Approach 1: Modified Dijkstra with Stop Constraint (Optimal)
        
        Use Dijkstra's algorithm but track stops in addition to cost.
        
        Time: O(E * K * log(V * K))
        Space: O(V * K)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, price in flights:
            graph[u].append((v, price))
        
        # Priority queue: (cost, city, stops_used)
        pq = [(0, src, 0)]
        
        # Track minimum cost to reach each city with each number of stops
        # visited[city][stops] = minimum cost
        visited = {}
        
        while pq:
            cost, city, stops = heapq.heappop(pq)
            
            if city == dst:
                return cost
            
            # Skip if we've already found a better path to this city with same or fewer stops
            if (city, stops) in visited:
                continue
            
            visited[(city, stops)] = cost
            
            # If we've used all allowed stops, can't continue
            if stops > k:
                continue
            
            # Explore neighbors
            for next_city, price in graph[city]:
                new_cost = cost + price
                new_stops = stops + 1
                
                # Only continue if we haven't exceeded stop limit
                if new_stops <= k + 1:  # k stops means k+1 cities
                    heapq.heappush(pq, (new_cost, next_city, new_stops))
        
        return -1
    
    def findCheapestPrice_approach2_bellman_ford_k_iterations(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        """
        Approach 2: Bellman-Ford with K+1 Iterations
        
        Use Bellman-Ford but limit to k+1 iterations (k stops = k+1 edges).
        
        Time: O(K * E)
        Space: O(V)
        """
        # Initialize distances
        dist = [float('inf')] * n
        dist[src] = 0
        
        # Perform k+1 iterations (k stops means k+1 edges)
        for _ in range(k + 1):
            # Create a copy to avoid using updated values in same iteration
            temp_dist = dist[:]
            
            for u, v, price in flights:
                if dist[u] != float('inf'):
                    temp_dist[v] = min(temp_dist[v], dist[u] + price)
            
            dist = temp_dist
        
        return dist[dst] if dist[dst] != float('inf') else -1
    
    def findCheapestPrice_approach3_dfs_with_memoization(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        """
        Approach 3: DFS with Memoization
        
        Use DFS to explore all paths with memoization for efficiency.
        
        Time: O(V * K + E * K)
        Space: O(V * K)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, price in flights:
            graph[u].append((v, price))
        
        # Memoization: memo[city][stops_remaining] = min_cost
        memo = {}
        
        def dfs(city, stops_remaining):
            """DFS to find minimum cost from city to dst with stops_remaining"""
            if city == dst:
                return 0
            
            if stops_remaining < 0:
                return float('inf')
            
            if (city, stops_remaining) in memo:
                return memo[(city, stops_remaining)]
            
            min_cost = float('inf')
            
            for next_city, price in graph[city]:
                cost = price + dfs(next_city, stops_remaining - 1)
                min_cost = min(min_cost, cost)
            
            memo[(city, stops_remaining)] = min_cost
            return min_cost
        
        result = dfs(src, k)
        return result if result != float('inf') else -1
    
    def findCheapestPrice_approach4_bfs_level_by_level(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        """
        Approach 4: BFS Level by Level
        
        Use BFS to explore paths level by level (each level = one stop).
        
        Time: O(V * K + E * K)
        Space: O(V)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, price in flights:
            graph[u].append((v, price))
        
        # BFS with level tracking
        queue = deque([(src, 0)])  # (city, cost)
        min_cost = [float('inf')] * n
        min_cost[src] = 0
        
        for stops in range(k + 2):  # k stops = k+1 flights
            next_queue = deque()
            temp_min_cost = min_cost[:]
            
            while queue:
                city, cost = queue.popleft()
                
                if cost > min_cost[city]:
                    continue
                
                for next_city, price in graph[city]:
                    new_cost = cost + price
                    
                    if new_cost < temp_min_cost[next_city]:
                        temp_min_cost[next_city] = new_cost
                        next_queue.append((next_city, new_cost))
            
            queue = next_queue
            min_cost = temp_min_cost
            
            if not queue:
                break
        
        return min_cost[dst] if min_cost[dst] != float('inf') else -1
    
    def findCheapestPrice_approach5_dynamic_programming(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        """
        Approach 5: Dynamic Programming
        
        Use DP where dp[i][j] = minimum cost to reach city j with exactly i stops.
        
        Time: O(K * E)
        Space: O(K * V)
        """
        # dp[stops][city] = minimum cost to reach city with exactly 'stops' stops
        INF = float('inf')
        dp = [[INF] * n for _ in range(k + 2)]
        
        # Base case: 0 stops to reach source city
        dp[0][src] = 0
        
        # Fill DP table
        for i in range(k + 1):
            for u, v, price in flights:
                if dp[i][u] != INF:
                    dp[i + 1][v] = min(dp[i + 1][v], dp[i][u] + price)
        
        # Find minimum cost to reach destination with at most k stops
        result = min(dp[i][dst] for i in range(k + 2))
        return result if result != INF else -1

def test_cheapest_flights():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (n, flights, src, dst, k, expected)
        (4, [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], 0, 3, 1, 700),
        (3, [[0,1,100],[1,2,100],[0,2,500]], 0, 2, 1, 200),
        (3, [[0,1,100],[1,2,100],[0,2,500]], 0, 2, 0, 500),
        (5, [[0,1,5],[1,2,5],[0,3,2],[3,1,2],[1,4,1],[4,2,1]], 0, 2, 2, 7),
        (3, [[0,1,100],[1,2,100]], 0, 2, 0, -1),
    ]
    
    approaches = [
        ("Modified Dijkstra", solution.findCheapestPrice_approach1_modified_dijkstra),
        ("Bellman-Ford K Iterations", solution.findCheapestPrice_approach2_bellman_ford_k_iterations),
        ("DFS Memoization", solution.findCheapestPrice_approach3_dfs_with_memoization),
        ("BFS Level by Level", solution.findCheapestPrice_approach4_bfs_level_by_level),
        ("Dynamic Programming", solution.findCheapestPrice_approach5_dynamic_programming),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, flights, src, dst, k, expected) in enumerate(test_cases):
            result = func(n, flights[:], src, dst, k)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} n={n}, src={src}, dst={dst}, k={k}, expected={expected}, got={result}")

def demonstrate_constrained_shortest_path():
    """Demonstrate constrained shortest path problem"""
    print("\n=== Constrained Shortest Path Demo ===")
    
    n = 4
    flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]]
    src = 0
    dst = 3
    k = 1
    
    print(f"Flight network: {n} cities")
    print(f"Flights: {flights}")
    print(f"Find cheapest path from {src} to {dst} with at most {k} stops")
    
    # Build adjacency list for visualization
    graph = defaultdict(list)
    for u, v, price in flights:
        graph[u].append((v, price))
    
    print(f"\nAdjacency list: {dict(graph)}")
    
    print(f"\nPossible paths from {src} to {dst}:")
    
    def find_all_paths(current, target, path, cost, stops):
        if current == target:
            print(f"  Path: {' -> '.join(map(str, path))}, Cost: {cost}, Stops: {stops}")
            return [(path[:], cost, stops)]
        
        if stops > k:
            return []
        
        all_paths = []
        for next_city, price in graph[current]:
            if next_city not in path:  # Avoid cycles
                path.append(next_city)
                all_paths.extend(find_all_paths(next_city, target, path, cost + price, stops + 1))
                path.pop()
        
        return all_paths
    
    all_paths = find_all_paths(src, dst, [src], 0, 0)
    
    # Filter paths with at most k stops
    valid_paths = [p for p in all_paths if p[2] <= k]
    
    if valid_paths:
        best_path = min(valid_paths, key=lambda x: x[1])
        print(f"\nBest path: {' -> '.join(map(str, best_path[0]))}")
        print(f"Cost: {best_path[1]}, Stops: {best_path[2]}")
    else:
        print(f"\nNo valid path found with at most {k} stops")

def demonstrate_bellman_ford_variant():
    """Demonstrate Bellman-Ford variant for k-stop constraint"""
    print("\n=== Bellman-Ford K-Stop Demo ===")
    
    n = 3
    flights = [[0,1,100],[1,2,100],[0,2,500]]
    src = 0
    dst = 2
    k = 1
    
    print(f"Graph: {flights}")
    print(f"Find path from {src} to {dst} with at most {k} stops")
    
    # Initialize distances
    dist = [float('inf')] * n
    dist[src] = 0
    
    print(f"\nBellman-Ford with {k+1} iterations:")
    print(f"Initial distances: {['∞' if d == float('inf') else d for d in dist]}")
    
    for iteration in range(k + 1):
        print(f"\nIteration {iteration + 1}:")
        temp_dist = dist[:]
        
        for u, v, price in flights:
            if dist[u] != float('inf'):
                old_dist = temp_dist[v]
                new_dist = dist[u] + price
                
                if new_dist < temp_dist[v]:
                    temp_dist[v] = new_dist
                    print(f"  Update: dist[{v}] = min({old_dist}, {dist[u]} + {price}) = {new_dist}")
        
        dist = temp_dist
        print(f"  Distances: {['∞' if d == float('inf') else d for d in dist]}")
    
    result = dist[dst] if dist[dst] != float('inf') else -1
    print(f"\nFinal result: {result}")

def analyze_stop_constraint_algorithms():
    """Analyze algorithms for stop-constrained shortest path"""
    print("\n=== Stop-Constrained Algorithms Analysis ===")
    
    print("Problem Characteristics:")
    print("• Standard shortest path + stop count constraint")
    print("• Cannot use pure Dijkstra (may exceed stop limit)")
    print("• Need to track both cost and stop count")
    print("• Trade-off between cost and path length")
    
    print("\nAlgorithm Comparison:")
    
    print("\n1. **Modified Dijkstra:**")
    print("   • Time: O(E * K * log(V * K))")
    print("   • Space: O(V * K)")
    print("   • Pros: Optimal for sparse graphs, early termination")
    print("   • Cons: Complex state management")
    
    print("\n2. **Bellman-Ford K Iterations:**")
    print("   • Time: O(K * E)")
    print("   • Space: O(V)")
    print("   • Pros: Simple, naturally handles k-constraint")
    print("   • Cons: No early termination")
    
    print("\n3. **DFS + Memoization:**")
    print("   • Time: O(V * K + E * K)")
    print("   • Space: O(V * K)")
    print("   • Pros: Intuitive recursive approach")
    print("   • Cons: May explore unnecessary paths")
    
    print("\n4. **BFS Level-by-Level:**")
    print("   • Time: O(V * K + E * K)")
    print("   • Space: O(V)")
    print("   • Pros: Natural level exploration")
    print("   • Cons: May revisit same states")
    
    print("\n5. **Dynamic Programming:**")
    print("   • Time: O(K * E)")
    print("   • Space: O(K * V)")
    print("   • Pros: Clean state definition")
    print("   • Cons: Higher space complexity")
    
    print("\nBest Choice:")
    print("• **Small K:** Bellman-Ford K iterations (simple & efficient)")
    print("• **Large K:** Modified Dijkstra (better for large search spaces)")
    print("• **Dense graphs:** DP approach")
    print("• **Educational:** DFS with memoization")

def demonstrate_dp_approach():
    """Demonstrate dynamic programming approach"""
    print("\n=== Dynamic Programming Demo ===")
    
    n = 4
    flights = [[0,1,100],[1,2,100],[1,3,600],[2,3,200]]
    src = 0
    dst = 3
    k = 1
    
    print(f"Find path from {src} to {dst} with at most {k} stops")
    print(f"Flights: {flights}")
    
    # DP table: dp[stops][city] = minimum cost
    INF = float('inf')
    dp = [[INF] * n for _ in range(k + 2)]
    
    # Base case
    dp[0][src] = 0
    
    print(f"\nDP table initialization:")
    print(f"dp[0][{src}] = 0 (source city with 0 stops)")
    
    for i in range(k + 1):
        print(f"\nProcessing {i} -> {i+1} stops:")
        
        for u, v, price in flights:
            if dp[i][u] != INF:
                old_cost = dp[i + 1][v]
                new_cost = dp[i][u] + price
                
                if new_cost < dp[i + 1][v]:
                    dp[i + 1][v] = new_cost
                    print(f"  Flight {u}->{v} (${price}): dp[{i+1}][{v}] = min({old_cost}, {dp[i][u]} + {price}) = {new_cost}")
        
        print(f"  DP state after {i+1} stops: {['∞' if x == INF else x for x in dp[i+1]]}")
    
    # Find result
    result = min(dp[i][dst] for i in range(k + 2))
    result = result if result != INF else -1
    
    print(f"\nResult: min(dp[0][{dst}], dp[1][{dst}], ...) = {result}")

def compare_constraint_handling():
    """Compare different constraint handling techniques"""
    print("\n=== Constraint Handling Techniques ===")
    
    print("Handling Stop Constraints in Shortest Path:")
    
    print("\n1. **State Space Expansion:**")
    print("   • Original state: (city)")
    print("   • Extended state: (city, stops_used)")
    print("   • Pros: Natural constraint integration")
    print("   • Cons: Exponential state space growth")
    
    print("\n2. **Algorithm Modification:**")
    print("   • Modify standard algorithms to respect constraints")
    print("   • Dijkstra: Track stops in priority queue")
    print("   • Bellman-Ford: Limit iterations to k+1")
    print("   • Pros: Leverages known algorithms")
    print("   • Cons: May require significant modifications")
    
    print("\n3. **Layer-based Processing:**")
    print("   • Process graph layer by layer (BFS-style)")
    print("   • Each layer represents one additional stop")
    print("   • Pros: Natural constraint boundary")
    print("   • Cons: May revisit same states multiple times")
    
    print("\n4. **Dynamic Programming:**")
    print("   • Define states with constraint dimensions")
    print("   • dp[constraint_value][node] = optimal_cost")
    print("   • Pros: Clear state definition and transitions")
    print("   • Cons: Space complexity grows with constraints")
    
    print("\nGeneral Principles:")
    print("• **Constraint Integration:** Embed constraints in state representation")
    print("• **Pruning:** Early termination when constraints violated")
    print("• **Trade-offs:** Time vs space vs implementation complexity")
    print("• **Optimality:** Ensure constraint satisfaction doesn't break optimality")
    
    print("\nReal-world Applications:")
    print("• **Flight booking:** Price vs number of connections")
    print("• **Supply chain:** Cost vs delivery time constraints")
    print("• **Network routing:** Bandwidth vs latency limits")
    print("• **Game AI:** Resource constraints in pathfinding")
    print("• **Project management:** Budget vs time constraints")

if __name__ == "__main__":
    test_cheapest_flights()
    demonstrate_constrained_shortest_path()
    demonstrate_bellman_ford_variant()
    analyze_stop_constraint_algorithms()
    demonstrate_dp_approach()
    compare_constraint_handling()

"""
Shortest Path Concepts:
1. Constrained Shortest Path Problems
2. Modified Dijkstra with Additional State
3. Bellman-Ford with Limited Iterations
4. Dynamic Programming with Constraint Dimensions
5. Trade-offs Between Cost and Path Length

Key Problem Insights:
- Standard shortest path + stop count constraint
- Cannot use pure Dijkstra (may exceed stop limit)
- Need state space extension: (city, stops_used)
- Multiple valid approaches with different trade-offs

Algorithm Strategy:
1. Extend state space to include constraint information
2. Modify shortest path algorithms to respect constraints
3. Track both cost and constraint usage
4. Return optimal solution satisfying all constraints

Bellman-Ford Adaptation:
- Limit iterations to k+1 (k stops = k+1 flights)
- Each iteration represents one additional stop
- Natural constraint enforcement through iteration limit
- Simple and effective for small k values

State Space Considerations:
- Original: O(V) states
- With k-constraint: O(V * K) states
- Trade-off: more states vs constraint satisfaction
- Pruning crucial for large state spaces

Real-world Applications:
- Flight booking with connection limits
- Network routing with hop constraints
- Supply chain optimization
- Resource-constrained pathfinding
- Multi-objective optimization problems

This problem demonstrates constrained optimization
in shortest path algorithms with practical applications.
"""
