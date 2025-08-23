"""
1928. Minimum Cost to Reach Destination in Time
Difficulty: Hard

Problem:
There is a country of n cities numbered from 0 to n-1 where all the roads are bidirectional. 
The country has a robust highway system represented by the 2D array edges where edges[i] = [xi, yi, timei] 
means that there is a bidirectional road between cities xi and yi that takes timei minutes to travel.

There may be multiple roads between the same two cities.

Now, you want to travel from city 0 to city n-1 with the minimum cost. The cost of the trip is the 
sum of passing fees of all the cities you visit (including both starting and ending city).

Given the 2D array edges representing the road system, the array passingFees representing the 
passing fees of each city, and an integer maxTime representing the maximum time to reach the 
destination, return the minimum cost to reach city n-1 from city 0 within maxTime minutes. 
If you cannot reach city n-1 within maxTime minutes, return -1.

Examples:
Input: maxTime = 30, edges = [[0,1,10],[1,2,10],[2,5,10],[0,3,1],[3,4,10],[4,5,15]], passingFees = [5,1,2,20,20,3]
Output: 11

Input: maxTime = 29, edges = [[0,1,10],[1,2,10],[2,5,10],[0,3,1],[3,4,10],[4,5,15]], passingFees = [5,1,2,20,20,3]
Output: 48

Input: maxTime = 25, edges = [[0,1,10],[1,2,10],[2,5,10],[0,3,1],[3,4,10],[4,5,15]], passingFees = [5,1,2,20,20,3]
Output: -1

Constraints:
- 1 <= maxTime <= 1000
- n == passingFees.length
- 2 <= n <= 1000
- 1 <= edges.length <= 1000
- 0 <= xi, yi <= n-1
- 1 <= timei <= 1000
- 1 <= passingFees[i] <= 1000
"""

from typing import List
import heapq
from collections import defaultdict

class Solution:
    def minCost_approach1_dijkstra_time_constrained(self, maxTime: int, edges: List[List[int]], passingFees: List[int]) -> int:
        """
        Approach 1: Modified Dijkstra with Time Constraint (Optimal)
        
        Use Dijkstra but track both cost and time, with time as constraint.
        
        Time: O(E log V * T) where T = maxTime
        Space: O(V * T)
        """
        n = len(passingFees)
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, time in edges:
            graph[u].append((v, time))
            graph[v].append((u, time))
        
        # State: (cost, time, city)
        # min_cost[city][time] = minimum cost to reach city with exactly time
        min_cost = [[float('inf')] * (maxTime + 1) for _ in range(n)]
        
        # Priority queue: (cost, time, city)
        pq = [(passingFees[0], 0, 0)]
        min_cost[0][0] = passingFees[0]
        
        while pq:
            cost, time, city = heapq.heappop(pq)
            
            # If we reached destination
            if city == n - 1:
                return cost
            
            # Skip if we found better path
            if cost > min_cost[city][time]:
                continue
            
            # Explore neighbors
            for next_city, travel_time in graph[city]:
                new_time = time + travel_time
                
                if new_time <= maxTime:
                    new_cost = cost + passingFees[next_city]
                    
                    if new_cost < min_cost[next_city][new_time]:
                        min_cost[next_city][new_time] = new_cost
                        heapq.heappush(pq, (new_cost, new_time, next_city))
        
        # Find minimum cost among all valid times
        result = min(min_cost[n-1])
        return result if result != float('inf') else -1
    
    def minCost_approach2_dp_bellman_ford_style(self, maxTime: int, edges: List[List[int]], passingFees: List[int]) -> int:
        """
        Approach 2: Dynamic Programming (Bellman-Ford Style)
        
        Use DP where dp[t][city] = minimum cost to reach city in exactly t time.
        
        Time: O(T * E)
        Space: O(T * V)
        """
        n = len(passingFees)
        
        # dp[time][city] = minimum cost to reach city in exactly time
        INF = float('inf')
        dp = [[INF] * n for _ in range(maxTime + 1)]
        
        # Base case: start at city 0 with time 0
        dp[0][0] = passingFees[0]
        
        # Fill DP table
        for t in range(maxTime):
            for u, v, travel_time in edges:
                new_time = t + travel_time
                
                if new_time <= maxTime:
                    # Both directions (bidirectional edges)
                    if dp[t][u] != INF:
                        dp[new_time][v] = min(dp[new_time][v], dp[t][u] + passingFees[v])
                    
                    if dp[t][v] != INF:
                        dp[new_time][u] = min(dp[new_time][u], dp[t][v] + passingFees[u])
        
        # Find minimum cost to reach destination within maxTime
        result = min(dp[t][n-1] for t in range(maxTime + 1))
        return result if result != INF else -1
    
    def minCost_approach3_dijkstra_multiple_states(self, maxTime: int, edges: List[List[int]], passingFees: List[int]) -> int:
        """
        Approach 3: Dijkstra with Multiple States per City
        
        Track multiple (cost, time) states per city using Dijkstra.
        
        Time: O(E log V * T)
        Space: O(V * T)
        """
        n = len(passingFees)
        
        # Build graph
        graph = defaultdict(list)
        for u, v, time in edges:
            graph[u].append((v, time))
            graph[v].append((u, time))
        
        # Track best cost for each (city, time) pair
        best_cost = {}
        
        # Priority queue: (cost, time, city)
        pq = [(passingFees[0], 0, 0)]
        
        while pq:
            cost, time, city = heapq.heappop(pq)
            
            # Check if we've seen this state with better cost
            if (city, time) in best_cost and best_cost[(city, time)] <= cost:
                continue
            
            best_cost[(city, time)] = cost
            
            # If reached destination
            if city == n - 1:
                return cost
            
            # Explore neighbors
            for next_city, travel_time in graph[city]:
                new_time = time + travel_time
                
                if new_time <= maxTime:
                    new_cost = cost + passingFees[next_city]
                    
                    # Only add if we haven't seen this state with better cost
                    if (next_city, new_time) not in best_cost or best_cost[(next_city, new_time)] > new_cost:
                        heapq.heappush(pq, (new_cost, new_time, next_city))
        
        return -1
    
    def minCost_approach4_dfs_memoization(self, maxTime: int, edges: List[List[int]], passingFees: List[int]) -> int:
        """
        Approach 4: DFS with Memoization
        
        Use DFS to explore paths with memoization for efficiency.
        
        Time: O(V * T + E * V * T)
        Space: O(V * T)
        """
        n = len(passingFees)
        
        # Build graph
        graph = defaultdict(list)
        for u, v, time in edges:
            graph[u].append((v, time))
            graph[v].append((u, time))
        
        # Memoization: memo[city][time] = minimum cost to reach destination from city with time remaining
        memo = {}
        
        def dfs(city, time_remaining):
            """DFS to find minimum cost from city to destination with time_remaining"""
            if city == n - 1:
                return 0  # Reached destination, no additional cost
            
            if time_remaining <= 0:
                return float('inf')  # No time left
            
            if (city, time_remaining) in memo:
                return memo[(city, time_remaining)]
            
            min_cost = float('inf')
            
            for next_city, travel_time in graph[city]:
                if travel_time <= time_remaining:
                    cost = passingFees[next_city] + dfs(next_city, time_remaining - travel_time)
                    min_cost = min(min_cost, cost)
            
            memo[(city, time_remaining)] = min_cost
            return min_cost
        
        result = passingFees[0] + dfs(0, maxTime)
        return result if result != float('inf') else -1
    
    def minCost_approach5_spfa_with_constraints(self, maxTime: int, edges: List[List[int]], passingFees: List[int]) -> int:
        """
        Approach 5: SPFA with Time Constraints
        
        Use SPFA algorithm modified for time-constrained shortest path.
        
        Time: O(V * E * T) worst case
        Space: O(V * T)
        """
        from collections import deque
        
        n = len(passingFees)
        
        # Build graph
        graph = defaultdict(list)
        for u, v, time in edges:
            graph[u].append((v, time))
            graph[v].append((u, time))
        
        # min_cost[city][time] = minimum cost to reach city with exactly time
        INF = float('inf')
        min_cost = [[INF] * (maxTime + 1) for _ in range(n)]
        
        # SPFA with time tracking
        queue = deque([(0, 0)])  # (city, time)
        min_cost[0][0] = passingFees[0]
        in_queue = [[False] * (maxTime + 1) for _ in range(n)]
        in_queue[0][0] = True
        
        while queue:
            city, time = queue.popleft()
            in_queue[city][time] = False
            
            for next_city, travel_time in graph[city]:
                new_time = time + travel_time
                
                if new_time <= maxTime:
                    new_cost = min_cost[city][time] + passingFees[next_city]
                    
                    if new_cost < min_cost[next_city][new_time]:
                        min_cost[next_city][new_time] = new_cost
                        
                        if not in_queue[next_city][new_time]:
                            queue.append((next_city, new_time))
                            in_queue[next_city][new_time] = True
        
        # Find minimum cost to reach destination
        result = min(min_cost[n-1])
        return result if result != INF else -1

def test_min_cost_reach_destination():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (maxTime, edges, passingFees, expected)
        (30, [[0,1,10],[1,2,10],[2,5,10],[0,3,1],[3,4,10],[4,5,15]], [5,1,2,20,20,3], 11),
        (29, [[0,1,10],[1,2,10],[2,5,10],[0,3,1],[3,4,10],[4,5,15]], [5,1,2,20,20,3], 48),
        (25, [[0,1,10],[1,2,10],[2,5,10],[0,3,1],[3,4,10],[4,5,15]], [5,1,2,20,20,3], -1),
        (11, [[0,1,10],[1,2,1]], [5,1,2], 8),
        (10, [[0,1,10],[1,2,1]], [5,1,2], -1),
    ]
    
    approaches = [
        ("Dijkstra Time Constrained", solution.minCost_approach1_dijkstra_time_constrained),
        ("DP Bellman-Ford Style", solution.minCost_approach2_dp_bellman_ford_style),
        ("Dijkstra Multiple States", solution.minCost_approach3_dijkstra_multiple_states),
        ("DFS Memoization", solution.minCost_approach4_dfs_memoization),
        ("SPFA with Constraints", solution.minCost_approach5_spfa_with_constraints),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (maxTime, edges, passingFees, expected) in enumerate(test_cases):
            result = func(maxTime, edges[:], passingFees[:])
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} maxTime={maxTime}, expected={expected}, got={result}")

def demonstrate_time_cost_tradeoff():
    """Demonstrate time vs cost trade-off analysis"""
    print("\n=== Time vs Cost Trade-off Demo ===")
    
    maxTime = 30
    edges = [[0,1,10],[1,2,10],[2,5,10],[0,3,1],[3,4,10],[4,5,15]]
    passingFees = [5,1,2,20,20,3]
    
    print(f"Road network:")
    for u, v, time in edges:
        print(f"  {u} ↔ {v}: {time} minutes")
    
    print(f"\nPassing fees:")
    for i, fee in enumerate(passingFees):
        print(f"  City {i}: ${fee}")
    
    print(f"\nMaximum time allowed: {maxTime} minutes")
    
    print(f"\nPossible paths from 0 to 5:")
    
    # Manual path analysis
    paths = [
        {
            "route": [0, 1, 2, 5],
            "times": [10, 10, 10],
            "total_time": 30,
            "fees": [5, 1, 2, 3],
            "total_cost": 11
        },
        {
            "route": [0, 3, 4, 5],
            "times": [1, 10, 15],
            "total_time": 26,
            "fees": [5, 20, 20, 3],
            "total_cost": 48
        }
    ]
    
    for i, path in enumerate(paths, 1):
        print(f"\nPath {i}: {' → '.join(map(str, path['route']))}")
        print(f"  Time segments: {path['times']} = {path['total_time']} minutes")
        print(f"  Fees: {path['fees']} = ${path['total_cost']}")
        print(f"  Valid: {'Yes' if path['total_time'] <= maxTime else 'No'}")
    
    print(f"\nOptimal choice: Path with minimum cost among valid paths")
    valid_paths = [p for p in paths if p['total_time'] <= maxTime]
    if valid_paths:
        optimal = min(valid_paths, key=lambda x: x['total_cost'])
        print(f"Best path: {' → '.join(map(str, optimal['route']))} with cost ${optimal['total_cost']}")

def demonstrate_state_space_dijkstra():
    """Demonstrate state space in time-constrained Dijkstra"""
    print("\n=== State Space Dijkstra Demo ===")
    
    print("Standard Dijkstra state: (cost, city)")
    print("Time-constrained state: (cost, time, city)")
    
    print("\nState space considerations:")
    print("• Each (city, time) pair is a unique state")
    print("• Multiple paths to same city with different times")
    print("• Need to track minimum cost for each (city, time)")
    print("• Early termination when destination reached")
    
    # Simulate state exploration
    maxTime = 30
    passingFees = [5,1,2,20,20,3]
    
    print(f"\nExample state exploration:")
    print(f"Starting state: (cost=5, time=0, city=0)")
    
    states = [
        (5, 0, 0),    # Start
        (6, 10, 1),   # 0→1 via edge (10 minutes)
        (25, 1, 3),   # 0→3 via edge (1 minute)
        (8, 20, 2),   # 1→2 via edge (10 minutes)
        (45, 11, 4),  # 3→4 via edge (10 minutes)
        (11, 30, 5),  # 2→5 via edge (10 minutes)
        (48, 26, 5),  # 4→5 via edge (15 minutes)
    ]
    
    for cost, time, city in states:
        print(f"  State: (cost=${cost}, time={time}, city={city})")
        if city == 5:  # Destination
            print(f"    → Reached destination! Cost=${cost}")

def analyze_dp_vs_dijkstra():
    """Analyze DP vs Dijkstra approaches for this problem"""
    print("\n=== DP vs Dijkstra Analysis ===")
    
    print("Problem Characteristics:")
    print("• Shortest path with time constraint")
    print("• Minimize cost subject to time ≤ maxTime")
    print("• Multiple valid paths with different time/cost trade-offs")
    
    print("\nDynamic Programming Approach:")
    print("• State: dp[time][city] = min cost to reach city in exactly time")
    print("• Transition: dp[t+travel_time][next_city] = min(current, dp[t][city] + fee)")
    print("• Time: O(T × E) where T = maxTime")
    print("• Space: O(T × V)")
    print("• Pros: Simple state definition, handles all paths")
    print("• Cons: Explores all time values, not goal-directed")
    
    print("\nDijkstra Approach:")
    print("• State: (cost, time, city) in priority queue")
    print("• Priority: Minimum cost (Dijkstra's greedy choice)")
    print("• Time: O(E log(V × T))")
    print("• Space: O(V × T)")
    print("• Pros: Goal-directed, early termination")
    print("• Cons: More complex state management")
    
    print("\nWhen to Use Each:")
    print("• **DP:** When need all reachable states, simple implementation")
    print("• **Dijkstra:** When graph is sparse, want early termination")
    print("• **Both viable:** For this problem size (V ≤ 1000, T ≤ 1000)")

def demonstrate_constraint_handling():
    """Demonstrate different constraint handling techniques"""
    print("\n=== Constraint Handling Techniques ===")
    
    print("Time Constraint Integration:")
    
    print("\n1. **State Space Extension:**")
    print("   • Original state: city")
    print("   • Extended state: (city, time_used)")
    print("   • Pros: Natural constraint integration")
    print("   • Cons: Exponential state space growth")
    
    print("\n2. **Constraint Checking in Search:**")
    print("   • Check time constraint before state transitions")
    print("   • Prune invalid paths early")
    print("   • Pros: Reduces search space")
    print("   • Cons: May miss optimal solutions if not careful")
    
    print("\n3. **Multi-Objective Optimization:**")
    print("   • Optimize cost subject to time constraint")
    print("   • Pareto frontier of (cost, time) solutions")
    print("   • Pros: Handles trade-offs explicitly")
    print("   • Cons: More complex implementation")
    
    print("\n4. **Layered Graph Approach:**")
    print("   • Create time layers: layer t contains all cities at time t")
    print("   • Edges between consecutive layers")
    print("   • Pros: Clear time progression")
    print("   • Cons: Large graph construction")
    
    print("\nPractical Implementation:")
    print("• **Dijkstra with extended state** (most practical)")
    print("• **DP with time dimension** (simple and reliable)")
    print("• **Constraint propagation** for advanced cases")

def compare_real_world_applications():
    """Compare real-world applications of time-cost optimization"""
    print("\n=== Real-World Applications ===")
    
    print("1. **Transportation Planning:**")
    print("   • Minimize travel cost within time budget")
    print("   • Flight connections with layover constraints")
    print("   • Public transit with schedule constraints")
    
    print("\n2. **Supply Chain Optimization:**")
    print("   • Minimize delivery cost within deadline")
    print("   • Warehouse routing with capacity constraints")
    print("   • Just-in-time manufacturing")
    
    print("\n3. **Project Management:**")
    print("   • Minimize project cost within deadline")
    print("   • Resource allocation with time constraints")
    print("   • Critical path analysis with cost factors")
    
    print("\n4. **Network Routing:**")
    print("   • Minimize bandwidth cost within latency limits")
    print("   • QoS-aware routing with delay constraints")
    print("   • Load balancing with response time SLAs")
    
    print("\n5. **Financial Trading:**")
    print("   • Minimize transaction costs within market hours")
    print("   • Portfolio rebalancing with execution time limits")
    print("   • Arbitrage opportunities with time decay")
    
    print("\nCommon Patterns:")
    print("• **Multi-objective optimization:** Cost vs time trade-offs")
    print("• **Constraint satisfaction:** Hard time limits")
    print("• **Dynamic programming:** State-dependent decisions")
    print("• **Graph algorithms:** Network-based problems")
    print("• **Real-time systems:** Time-critical applications")

if __name__ == "__main__":
    test_min_cost_reach_destination()
    demonstrate_time_cost_tradeoff()
    demonstrate_state_space_dijkstra()
    analyze_dp_vs_dijkstra()
    demonstrate_constraint_handling()
    compare_real_world_applications()

"""
Shortest Path Concepts:
1. Time-Constrained Shortest Path Problems
2. Multi-Objective Optimization (Cost vs Time)
3. Extended State Space in Graph Algorithms
4. Dynamic Programming with Time Dimensions
5. Constraint Satisfaction in Pathfinding

Key Problem Insights:
- Minimize cost subject to time constraint
- Two competing objectives: cost and time
- Extended state space: (city, time_used)
- Multiple valid paths with different trade-offs

Algorithm Strategy:
1. Extend state space to include time dimension
2. Use Dijkstra with (cost, time, city) states
3. Respect time constraint in state transitions
4. Return minimum cost among valid solutions

State Space Extension:
- Original: city → cost
- Extended: (city, time) → cost
- Allows multiple visits to same city with different times
- Essential for time-constrained optimization

Dijkstra Modifications:
- Priority queue: (cost, time, city)
- State uniqueness: (city, time) pairs
- Constraint checking: time ≤ maxTime
- Early termination when destination reached

Dynamic Programming Alternative:
- State: dp[time][city] = minimum cost
- Transition: explore all edges from each time layer
- Time complexity: O(maxTime × edges)
- Space complexity: O(maxTime × cities)

Real-world Applications:
- Transportation and logistics planning
- Supply chain optimization
- Project management with deadlines
- Network routing with QoS constraints
- Financial trading systems

This problem demonstrates multi-objective optimization
in shortest path algorithms with practical constraints.
"""
