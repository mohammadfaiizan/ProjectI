"""
1976. Number of Ways to Arrive at Destination
Difficulty: Medium

Problem:
You are in a city that consists of n intersections numbered from 0 to n - 1 with 
bidirectional roads between some intersections. The inputs are generated such that 
you can reach any intersection from any other intersection and that there is at most 
one road between any two intersections.

You are given an integer n and a 2D integer array roads where roads[i] = [ui, vi, timei] 
means that there is a road between intersections ui and vi that takes timei minutes to travel.

You want to know in how many ways you can travel from intersection 0 to intersection n - 1 
in the shortest time possible.

Return the number of ways you can arrive at your destination in the shortest time. 
Since the answer may be large, return it modulo 10^9 + 7.

Examples:
Input: n = 7, roads = [[0,6,7],[0,1,2],[1,2,3],[1,3,3],[6,3,3],[3,5,1],[6,5,1],[2,5,1],[5,4,2],[4,6,2]]
Output: 4

Input: n = 2, roads = [[1,0,10]]
Output: 1

Constraints:
- 1 <= n <= 200
- n - 1 <= roads.length <= n * (n - 1) / 2
- roads[i].length == 3
- 0 <= ui, vi <= n - 1
- 1 <= timei <= 10^9
- ui != vi
- There is at most one road between any pair of intersections
"""

from typing import List
import heapq
from collections import defaultdict

class Solution:
    def countPaths_approach1_dijkstra_path_counting(self, n: int, roads: List[List[int]]) -> int:
        """
        Approach 1: Dijkstra with Path Counting (Optimal)
        
        Use Dijkstra to find shortest distances and count paths simultaneously.
        
        Time: O(E log V)
        Space: O(V + E)
        """
        MOD = 10**9 + 7
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, time in roads:
            graph[u].append((v, time))
            graph[v].append((u, time))
        
        # Dijkstra with path counting
        distances = [float('inf')] * n
        ways = [0] * n
        distances[0] = 0
        ways[0] = 1
        
        pq = [(0, 0)]  # (distance, node)
        
        while pq:
            dist, u = heapq.heappop(pq)
            
            if dist > distances[u]:
                continue
            
            for v, time in graph[u]:
                new_dist = dist + time
                
                if new_dist < distances[v]:
                    # Found shorter path
                    distances[v] = new_dist
                    ways[v] = ways[u]
                    heapq.heappush(pq, (new_dist, v))
                elif new_dist == distances[v]:
                    # Found another shortest path
                    ways[v] = (ways[v] + ways[u]) % MOD
        
        return ways[n - 1]
    
    def countPaths_approach2_dijkstra_separate_phases(self, n: int, roads: List[List[int]]) -> int:
        """
        Approach 2: Dijkstra in Two Phases
        
        First find shortest distances, then count paths on shortest path DAG.
        
        Time: O(E log V + V + E)
        Space: O(V + E)
        """
        MOD = 10**9 + 7
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, time in roads:
            graph[u].append((v, time))
            graph[v].append((u, time))
        
        # Phase 1: Find shortest distances using Dijkstra
        distances = [float('inf')] * n
        distances[0] = 0
        
        pq = [(0, 0)]
        
        while pq:
            dist, u = heapq.heappop(pq)
            
            if dist > distances[u]:
                continue
            
            for v, time in graph[u]:
                new_dist = dist + time
                
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
        
        # Phase 2: Count paths on shortest path DAG
        ways = [0] * n
        ways[0] = 1
        
        # Create list of nodes sorted by distance
        nodes_by_dist = list(range(n))
        nodes_by_dist.sort(key=lambda x: distances[x])
        
        for u in nodes_by_dist:
            if distances[u] == float('inf'):
                continue
            
            for v, time in graph[u]:
                if distances[u] + time == distances[v]:
                    ways[v] = (ways[v] + ways[u]) % MOD
        
        return ways[n - 1]
    
    def countPaths_approach3_bellman_ford_path_counting(self, n: int, roads: List[List[int]]) -> int:
        """
        Approach 3: Bellman-Ford with Path Counting
        
        Use Bellman-Ford algorithm modified to count paths.
        
        Time: O(V * E)
        Space: O(V + E)
        """
        MOD = 10**9 + 7
        
        # Initialize distances and path counts
        distances = [float('inf')] * n
        ways = [0] * n
        distances[0] = 0
        ways[0] = 1
        
        # Bellman-Ford relaxation with path counting
        for _ in range(n - 1):
            updated = False
            
            for u, v, time in roads:
                # Check both directions (bidirectional)
                for source, dest in [(u, v), (v, u)]:
                    if distances[source] != float('inf'):
                        new_dist = distances[source] + time
                        
                        if new_dist < distances[dest]:
                            distances[dest] = new_dist
                            ways[dest] = ways[source]
                            updated = True
                        elif new_dist == distances[dest]:
                            ways[dest] = (ways[dest] + ways[source]) % MOD
            
            if not updated:
                break
        
        return ways[n - 1]
    
    def countPaths_approach4_dfs_memoization(self, n: int, roads: List[List[int]]) -> int:
        """
        Approach 4: DFS with Memoization
        
        Use DFS to explore paths with memoization for efficiency.
        
        Time: O(V + E) after memoization
        Space: O(V + E)
        """
        MOD = 10**9 + 7
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, time in roads:
            graph[u].append((v, time))
            graph[v].append((u, time))
        
        # First, find shortest distance to destination
        def dijkstra_shortest():
            distances = [float('inf')] * n
            distances[0] = 0
            pq = [(0, 0)]
            
            while pq:
                dist, u = heapq.heappop(pq)
                
                if dist > distances[u]:
                    continue
                
                for v, time in graph[u]:
                    new_dist = dist + time
                    if new_dist < distances[v]:
                        distances[v] = new_dist
                        heapq.heappush(pq, (new_dist, v))
            
            return distances
        
        distances = dijkstra_shortest()
        shortest_dist = distances[n - 1]
        
        if shortest_dist == float('inf'):
            return 0
        
        # DFS with memoization to count paths
        memo = {}
        
        def dfs(node, dist_so_far):
            """Count paths from node to destination with remaining optimal distance"""
            if node == n - 1:
                return 1 if dist_so_far == shortest_dist else 0
            
            if dist_so_far > shortest_dist:
                return 0
            
            if (node, dist_so_far) in memo:
                return memo[(node, dist_so_far)]
            
            count = 0
            for neighbor, time in graph[node]:
                new_dist = dist_so_far + time
                if new_dist <= shortest_dist and distances[neighbor] == shortest_dist - new_dist:
                    count = (count + dfs(neighbor, new_dist)) % MOD
            
            memo[(node, dist_so_far)] = count
            return count
        
        return dfs(0, 0)
    
    def countPaths_approach5_topological_sort_dag(self, n: int, roads: List[List[int]]) -> int:
        """
        Approach 5: Topological Sort on Shortest Path DAG
        
        Create DAG of shortest paths and use topological sort to count.
        
        Time: O(E log V + V + E)
        Space: O(V + E)
        """
        MOD = 10**9 + 7
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, time in roads:
            graph[u].append((v, time))
            graph[v].append((u, time))
        
        # Find shortest distances
        distances = [float('inf')] * n
        distances[0] = 0
        pq = [(0, 0)]
        
        while pq:
            dist, u = heapq.heappop(pq)
            
            if dist > distances[u]:
                continue
            
            for v, time in graph[u]:
                new_dist = dist + time
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
        
        # Build shortest path DAG
        dag = defaultdict(list)
        in_degree = [0] * n
        
        for u in range(n):
            for v, time in graph[u]:
                if distances[u] + time == distances[v]:
                    dag[u].append(v)
                    in_degree[v] += 1
        
        # Topological sort with path counting
        from collections import deque
        
        queue = deque()
        ways = [0] * n
        ways[0] = 1
        
        # Start with nodes that have no incoming edges in DAG
        for i in range(n):
            if distances[i] != float('inf') and in_degree[i] == 0:
                queue.append(i)
        
        while queue:
            u = queue.popleft()
            
            for v in dag[u]:
                ways[v] = (ways[v] + ways[u]) % MOD
                in_degree[v] -= 1
                
                if in_degree[v] == 0:
                    queue.append(v)
        
        return ways[n - 1]

def test_count_paths():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (n, roads, expected)
        (7, [[0,6,7],[0,1,2],[1,2,3],[1,3,3],[6,3,3],[3,5,1],[6,5,1],[2,5,1],[5,4,2],[4,6,2]], 4),
        (2, [[1,0,10]], 1),
        (3, [[0,1,1],[1,2,1],[0,2,2]], 1),
        (4, [[0,1,1],[1,2,1],[2,3,1],[0,3,3]], 1),
        (3, [[0,1,1],[0,2,1],[1,2,1]], 2),
    ]
    
    approaches = [
        ("Dijkstra Path Counting", solution.countPaths_approach1_dijkstra_path_counting),
        ("Dijkstra Two Phases", solution.countPaths_approach2_dijkstra_separate_phases),
        ("Bellman-Ford Path Counting", solution.countPaths_approach3_bellman_ford_path_counting),
        ("DFS Memoization", solution.countPaths_approach4_dfs_memoization),
        ("Topological Sort DAG", solution.countPaths_approach5_topological_sort_dag),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, roads, expected) in enumerate(test_cases):
            result = func(n, roads[:])
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} n={n}, expected={expected}, got={result}")

def demonstrate_path_counting_concept():
    """Demonstrate the path counting concept"""
    print("\n=== Path Counting Demo ===")
    
    n = 4
    roads = [[0,1,1],[0,2,2],[1,2,1],[1,3,2],[2,3,1]]
    
    print(f"Graph with {n} nodes:")
    for u, v, time in roads:
        print(f"  {u} ↔ {v}: {time} time")
    
    # Find shortest paths manually
    print(f"\nFinding shortest path from 0 to 3:")
    
    paths = [
        {"route": [0, 1, 3], "time": 3, "description": "0 → 1 → 3"},
        {"route": [0, 1, 2, 3], "time": 3, "description": "0 → 1 → 2 → 3"},
        {"route": [0, 2, 3], "time": 3, "description": "0 → 2 → 3"},
    ]
    
    for i, path in enumerate(paths, 1):
        print(f"  Path {i}: {path['description']}, Time: {path['time']}")
    
    shortest_time = min(path['time'] for path in paths)
    shortest_paths = [path for path in paths if path['time'] == shortest_time]
    
    print(f"\nShortest time: {shortest_time}")
    print(f"Number of shortest paths: {len(shortest_paths)}")
    
    for i, path in enumerate(shortest_paths, 1):
        print(f"  Shortest path {i}: {path['description']}")

def demonstrate_dijkstra_path_counting():
    """Demonstrate Dijkstra with path counting step by step"""
    print("\n=== Dijkstra Path Counting Demo ===")
    
    n = 4
    roads = [[0,1,1],[0,2,2],[1,2,1],[1,3,2],[2,3,1]]
    
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, time in roads:
        graph[u].append((v, time))
        graph[v].append((u, time))
    
    print(f"Adjacency list: {dict(graph)}")
    
    # Dijkstra with path counting
    MOD = 10**9 + 7
    distances = [float('inf')] * n
    ways = [0] * n
    distances[0] = 0
    ways[0] = 1
    
    pq = [(0, 0)]
    
    print(f"\nDijkstra execution:")
    print(f"Initial: distances={distances}, ways={ways}")
    
    step = 0
    while pq:
        step += 1
        dist, u = heapq.heappop(pq)
        
        print(f"\nStep {step}: Processing node {u} with distance {dist}")
        
        if dist > distances[u]:
            print(f"  Skipping (already have better distance)")
            continue
        
        for v, time in graph[u]:
            new_dist = dist + time
            
            print(f"  Edge {u}→{v} (time={time}): new_dist={new_dist}")
            print(f"    Current: distances[{v}]={distances[v]}, ways[{v}]={ways[v]}")
            
            if new_dist < distances[v]:
                distances[v] = new_dist
                ways[v] = ways[u]
                heapq.heappush(pq, (new_dist, v))
                print(f"    Updated: distances[{v}]={distances[v]}, ways[{v}]={ways[v]} (shorter path)")
            elif new_dist == distances[v]:
                ways[v] = (ways[v] + ways[u]) % MOD
                print(f"    Updated: ways[{v}]={ways[v]} (another shortest path)")
        
        print(f"  Current state: distances={distances}, ways={ways}")
    
    print(f"\nFinal result: {ways[n-1]} ways to reach destination")

def analyze_shortest_path_dag():
    """Analyze shortest path DAG construction"""
    print("\n=== Shortest Path DAG Analysis ===")
    
    print("Shortest Path DAG Construction:")
    print("1. **Find shortest distances** using Dijkstra/Bellman-Ford")
    print("2. **Create DAG edges:** u→v if dist[u] + edge_weight = dist[v]")
    print("3. **Count paths in DAG** using topological sort or DP")
    
    print("\nDAG Properties:")
    print("• **Acyclic:** No cycles by construction")
    print("• **Directed:** Respects shortest path directions")
    print("• **Subgraph:** Subset of original graph edges")
    print("• **Path preserving:** All shortest paths preserved")
    
    print("\nPath Counting Techniques:")
    
    print("\n1. **Integrated Dijkstra:**")
    print("   • Count paths during shortest distance computation")
    print("   • Update counts when finding shorter/equal paths")
    print("   • Most efficient single-pass algorithm")
    
    print("\n2. **Two-Phase Approach:**")
    print("   • Phase 1: Find shortest distances")
    print("   • Phase 2: Count paths in shortest path DAG")
    print("   • Cleaner separation of concerns")
    
    print("\n3. **Topological Sort:**")
    print("   • Build shortest path DAG explicitly")
    print("   • Use topological sort for path counting")
    print("   • Natural for DAG processing")
    
    print("\n4. **Dynamic Programming:**")
    print("   • DP state: number of paths to each node")
    print("   • Transition: sum paths from predecessors")
    print("   • Works well with memoization")

def compare_algorithmic_approaches():
    """Compare different algorithmic approaches"""
    print("\n=== Algorithmic Approaches Comparison ===")
    
    print("1. **Integrated Dijkstra (Recommended):**")
    print("   ✅ Single pass algorithm")
    print("   ✅ Optimal O(E log V) time complexity")
    print("   ✅ Natural integration of distance and counting")
    print("   ❌ Slightly more complex implementation")
    
    print("\n2. **Two-Phase Dijkstra:**")
    print("   ✅ Clear separation of concerns")
    print("   ✅ Easy to understand and debug")
    print("   ✅ Reusable distance computation")
    print("   ❌ Two passes through graph")
    
    print("\n3. **Bellman-Ford Path Counting:**")
    print("   ✅ Handles negative weights (if needed)")
    print("   ✅ Simple relaxation-based approach")
    print("   ❌ O(V × E) time complexity")
    print("   ❌ Slower for this problem")
    
    print("\n4. **DFS with Memoization:**")
    print("   ✅ Intuitive recursive approach")
    print("   ✅ Natural path exploration")
    print("   ❌ Requires pre-computing shortest distances")
    print("   ❌ Potential stack overflow for large graphs")
    
    print("\n5. **Topological Sort on DAG:**")
    print("   ✅ Explicit DAG construction")
    print("   ✅ Standard DAG algorithms")
    print("   ✅ Good for analysis and visualization")
    print("   ❌ Additional DAG construction overhead")
    
    print("\nBest Choice for Different Scenarios:")
    print("• **Production code:** Integrated Dijkstra")
    print("• **Educational:** Two-phase approach")
    print("• **Negative weights:** Bellman-Ford variant")
    print("• **DAG analysis:** Topological sort approach")
    print("• **Complex constraints:** DFS with memoization")

def analyze_real_world_applications():
    """Analyze real-world applications of shortest path counting"""
    print("\n=== Real-World Applications ===")
    
    print("1. **Transportation Networks:**")
    print("   • Count fastest routes between cities")
    print("   • Load balancing across optimal paths")
    print("   • Redundancy analysis for critical routes")
    
    print("\n2. **Communication Networks:**")
    print("   • Multiple optimal routing paths")
    print("   • Network reliability analysis")
    print("   • Bandwidth distribution strategies")
    
    print("\n3. **Social Networks:**")
    print("   • Information propagation paths")
    print("   • Influence maximization strategies")
    print("   • Social distance analysis")
    
    print("\n4. **Game Development:**")
    print("   • Multiple optimal paths for AI")
    print("   • Difficulty balancing")
    print("   • Player choice analysis")
    
    print("\n5. **Supply Chain:**")
    print("   • Optimal delivery route alternatives")
    print("   • Risk distribution across paths")
    print("   • Contingency planning")
    
    print("\n6. **Financial Systems:**")
    print("   • Arbitrage opportunity counting")
    print("   • Risk-adjusted optimal strategies")
    print("   • Portfolio optimization paths")
    
    print("\nKey Insights:")
    print("• **Redundancy:** Multiple optimal solutions provide robustness")
    print("• **Load distribution:** Spread traffic across optimal paths")
    print("• **Risk management:** Alternative paths reduce single points of failure")
    print("• **Decision support:** Quantify number of equivalent optimal choices")
    print("• **System analysis:** Understand solution space characteristics")

if __name__ == "__main__":
    test_count_paths()
    demonstrate_path_counting_concept()
    demonstrate_dijkstra_path_counting()
    analyze_shortest_path_dag()
    compare_algorithmic_approaches()
    analyze_real_world_applications()

"""
Shortest Path Concepts:
1. Path Counting in Shortest Path Problems
2. Shortest Path DAG Construction and Analysis
3. Integrated Dijkstra with Path Counting
4. Topological Sort on DAGs
5. Multi-Solution Optimization Problems

Key Problem Insights:
- Count number of shortest paths, not just find one
- Multiple optimal solutions common in real networks
- Path counting requires careful state management
- DAG structure emerges from shortest path constraints

Algorithm Strategy:
1. Find shortest distances using Dijkstra
2. Simultaneously count paths to each node
3. Handle equal distance updates by adding path counts
4. Use modular arithmetic for large numbers

Dijkstra Integration:
- Extend standard Dijkstra to track path counts
- When finding shorter path: reset count to source count
- When finding equal path: add source count to current count
- Natural integration with single-pass algorithm

DAG Construction:
- Edge u→v in DAG if dist[u] + weight = dist[v]
- All shortest paths preserved in DAG structure
- Enables topological sort for path counting
- Clear separation of distance and counting phases

Path Counting Techniques:
- Dynamic programming on shortest path DAG
- Topological sort with cumulative counting
- Recursive memoization approaches
- Integrated shortest path algorithms

Real-world Applications:
- Transportation network redundancy
- Communication system reliability
- Social network information flow
- Supply chain alternative routing
- Financial arbitrage analysis

This problem demonstrates advanced shortest path
algorithms with solution counting and analysis.
"""
