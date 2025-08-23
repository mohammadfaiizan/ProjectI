"""
743. Network Delay Time
Difficulty: Medium

Problem:
You are given a network of n nodes, labeled from 1 to n. You are also given times, 
a list of travel times as directed edges times[i] = (ui, vi, wi), where ui is the 
source node, vi is the target node, and wi is the time it takes for a signal to 
travel from source to target.

We will send a signal from a given node k. Return the minimum time it takes for all 
the n nodes to receive the signal. If it is impossible for all nodes to receive the 
signal, return -1.

Examples:
Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
Output: 2

Input: times = [[1,2,1]], n = 2, k = 1
Output: 1

Input: times = [[1,2,1]], n = 2, k = 2
Output: -1

Constraints:
- 1 <= k <= n <= 100
- 1 <= times.length <= 6000
- times[i].length == 3
- 1 <= ui, vi <= n
- ui != vi
- 0 <= wi <= 100
- All the pairs (ui, vi) are unique
"""

from typing import List
import heapq
from collections import defaultdict

class Solution:
    def networkDelayTime_approach1_dijkstra_heap(self, times: List[List[int]], n: int, k: int) -> int:
        """
        Approach 1: Dijkstra's Algorithm with Heap (Optimal)
        
        Use Dijkstra's algorithm to find shortest paths from source k to all nodes.
        
        Time: O(E log V) where E = edges, V = vertices
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, w in times:
            graph[u].append((v, w))
        
        # Dijkstra's algorithm
        distances = {}
        pq = [(0, k)]  # (distance, node)
        
        while pq:
            dist, node = heapq.heappop(pq)
            
            if node in distances:
                continue  # Already processed with shorter distance
            
            distances[node] = dist
            
            # Process neighbors
            for neighbor, weight in graph[node]:
                if neighbor not in distances:
                    heapq.heappush(pq, (dist + weight, neighbor))
        
        # Check if all nodes are reachable
        if len(distances) != n:
            return -1
        
        return max(distances.values())
    
    def networkDelayTime_approach2_bellman_ford(self, times: List[List[int]], n: int, k: int) -> int:
        """
        Approach 2: Bellman-Ford Algorithm
        
        Use Bellman-Ford for single-source shortest path (handles negative weights).
        
        Time: O(V * E)
        Space: O(V)
        """
        # Initialize distances
        distances = [float('inf')] * (n + 1)
        distances[k] = 0
        
        # Relax edges n-1 times
        for _ in range(n - 1):
            updated = False
            for u, v, w in times:
                if distances[u] != float('inf') and distances[u] + w < distances[v]:
                    distances[v] = distances[u] + w
                    updated = True
            
            if not updated:
                break  # Early termination if no updates
        
        # Find maximum distance (excluding infinity)
        max_dist = 0
        for i in range(1, n + 1):
            if distances[i] == float('inf'):
                return -1  # Node not reachable
            max_dist = max(max_dist, distances[i])
        
        return max_dist
    
    def networkDelayTime_approach3_floyd_warshall(self, times: List[List[int]], n: int, k: int) -> int:
        """
        Approach 3: Floyd-Warshall Algorithm
        
        Find all-pairs shortest paths, then extract distances from k.
        
        Time: O(V^3)
        Space: O(V^2)
        """
        # Initialize distance matrix
        INF = float('inf')
        dist = [[INF] * (n + 1) for _ in range(n + 1)]
        
        # Distance from node to itself is 0
        for i in range(1, n + 1):
            dist[i][i] = 0
        
        # Fill in given edges
        for u, v, w in times:
            dist[u][v] = w
        
        # Floyd-Warshall algorithm
        for k_node in range(1, n + 1):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if dist[i][k_node] + dist[k_node][j] < dist[i][j]:
                        dist[i][j] = dist[i][k_node] + dist[k_node][j]
        
        # Find maximum distance from source k
        max_dist = 0
        for i in range(1, n + 1):
            if dist[k][i] == INF:
                return -1  # Node not reachable
            max_dist = max(max_dist, dist[k][i])
        
        return max_dist
    
    def networkDelayTime_approach4_dfs_memoization(self, times: List[List[int]], n: int, k: int) -> int:
        """
        Approach 4: DFS with Memoization
        
        Use DFS to explore paths with memoization for optimization.
        
        Time: O(V + E) with memoization
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, w in times:
            graph[u].append((v, w))
        
        # Memoization for shortest distances
        memo = {}
        
        def dfs(node):
            """DFS to find shortest distance from k to node"""
            if node in memo:
                return memo[node]
            
            if node == k:
                return 0
            
            min_dist = float('inf')
            
            # Try all incoming edges
            for u, v, w in times:
                if v == node:
                    dist_to_u = dfs(u)
                    if dist_to_u != float('inf'):
                        min_dist = min(min_dist, dist_to_u + w)
            
            memo[node] = min_dist
            return min_dist
        
        # Find shortest distance to each node
        max_dist = 0
        for i in range(1, n + 1):
            dist = dfs(i)
            if dist == float('inf'):
                return -1
            max_dist = max(max_dist, dist)
        
        return max_dist
    
    def networkDelayTime_approach5_spfa(self, times: List[List[int]], n: int, k: int) -> int:
        """
        Approach 5: SPFA (Shortest Path Faster Algorithm)
        
        Optimized version of Bellman-Ford using queue.
        
        Time: O(V * E) worst case, often much better
        Space: O(V + E)
        """
        from collections import deque
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, w in times:
            graph[u].append((v, w))
        
        # Initialize distances
        distances = [float('inf')] * (n + 1)
        distances[k] = 0
        
        # SPFA algorithm
        queue = deque([k])
        in_queue = [False] * (n + 1)
        in_queue[k] = True
        
        while queue:
            u = queue.popleft()
            in_queue[u] = False
            
            for v, w in graph[u]:
                if distances[u] + w < distances[v]:
                    distances[v] = distances[u] + w
                    
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True
        
        # Find maximum distance
        max_dist = 0
        for i in range(1, n + 1):
            if distances[i] == float('inf'):
                return -1
            max_dist = max(max_dist, distances[i])
        
        return max_dist

def test_network_delay_time():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (times, n, k, expected)
        ([[2,1,1],[2,3,1],[3,4,1]], 4, 2, 2),
        ([[1,2,1]], 2, 1, 1),
        ([[1,2,1]], 2, 2, -1),
        ([[1,2,1],[2,3,2],[1,3,4]], 3, 1, 3),
        ([[1,2,1],[2,1,3]], 2, 2, 4),
    ]
    
    approaches = [
        ("Dijkstra's Heap", solution.networkDelayTime_approach1_dijkstra_heap),
        ("Bellman-Ford", solution.networkDelayTime_approach2_bellman_ford),
        ("Floyd-Warshall", solution.networkDelayTime_approach3_floyd_warshall),
        ("DFS Memoization", solution.networkDelayTime_approach4_dfs_memoization),
        ("SPFA", solution.networkDelayTime_approach5_spfa),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (times, n, k, expected) in enumerate(test_cases):
            result = func(times[:], n, k)  # Copy to avoid modification
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} times={times}, n={n}, k={k}, expected={expected}, got={result}")

def demonstrate_dijkstra_algorithm():
    """Demonstrate Dijkstra's algorithm step by step"""
    print("\n=== Dijkstra's Algorithm Demo ===")
    
    times = [[2,1,1],[2,3,1],[3,4,1]]
    n = 4
    k = 2
    
    print(f"Network: {times}")
    print(f"Nodes: {n}, Source: {k}")
    
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))
    
    print(f"Adjacency list: {dict(graph)}")
    
    # Dijkstra's algorithm with step-by-step output
    distances = {}
    pq = [(0, k)]
    
    print(f"\nDijkstra's algorithm steps:")
    print(f"Initial: priority_queue = {pq}")
    
    step = 0
    while pq:
        step += 1
        dist, node = heapq.heappop(pq)
        
        print(f"\nStep {step}: Processing node {node} with distance {dist}")
        
        if node in distances:
            print(f"  Already processed, skipping")
            continue
        
        distances[node] = dist
        print(f"  Setting distance[{node}] = {dist}")
        print(f"  Current distances: {distances}")
        
        # Process neighbors
        neighbors_added = []
        for neighbor, weight in graph[node]:
            if neighbor not in distances:
                new_dist = dist + weight
                heapq.heappush(pq, (new_dist, neighbor))
                neighbors_added.append((neighbor, new_dist))
        
        if neighbors_added:
            print(f"  Added to queue: {neighbors_added}")
        
        print(f"  Priority queue: {sorted(pq)}")
    
    print(f"\nFinal distances: {distances}")
    
    if len(distances) != n:
        print(f"Not all nodes reachable: {n - len(distances)} nodes missing")
        result = -1
    else:
        result = max(distances.values())
        print(f"Maximum distance (network delay): {result}")
    
    return result

def analyze_shortest_path_algorithms():
    """Analyze different shortest path algorithms"""
    print("\n=== Shortest Path Algorithms Analysis ===")
    
    print("Algorithm Comparison for Single-Source Shortest Path:")
    
    print("\n1. **Dijkstra's Algorithm:**")
    print("   • Time: O(E log V) with binary heap")
    print("   • Space: O(V + E)")
    print("   • Requirements: Non-negative edge weights")
    print("   • Best for: Dense graphs, non-negative weights")
    print("   • Greedy approach: Always picks minimum distance")
    
    print("\n2. **Bellman-Ford Algorithm:**")
    print("   • Time: O(V * E)")
    print("   • Space: O(V)")
    print("   • Requirements: Can handle negative weights")
    print("   • Best for: Graphs with negative weights, cycle detection")
    print("   • Dynamic programming approach")
    
    print("\n3. **Floyd-Warshall Algorithm:**")
    print("   • Time: O(V³)")
    print("   • Space: O(V²)")
    print("   • Purpose: All-pairs shortest paths")
    print("   • Best for: Dense graphs, small number of vertices")
    print("   • Can detect negative cycles")
    
    print("\n4. **SPFA (Shortest Path Faster Algorithm):**")
    print("   • Time: O(V * E) worst case, often O(E) average")
    print("   • Space: O(V + E)")
    print("   • Optimized Bellman-Ford with queue")
    print("   • Best for: Sparse graphs with few negative edges")
    
    print("\nFor Network Delay Time Problem:")
    print("• **Graph characteristics:** Directed, non-negative weights")
    print("• **Query type:** Single-source to all destinations")
    print("• **Optimal choice:** Dijkstra's algorithm")
    print("• **Why:** Non-negative weights, need single-source distances")
    
    print("\nReal-world Applications:")
    print("• **Network routing:** Internet packet routing")
    print("• **GPS navigation:** Road network shortest paths")
    print("• **Social networks:** Degrees of separation")
    print("• **Game AI:** Pathfinding in game worlds")
    print("• **Supply chain:** Optimal delivery routes")

def demonstrate_algorithm_evolution():
    """Demonstrate evolution from simple to optimized algorithms"""
    print("\n=== Algorithm Evolution Demo ===")
    
    print("Evolution of Shortest Path Algorithms:")
    
    print("\n1. **Naive Approach (DFS/BFS):**")
    print("   • Try all possible paths")
    print("   • Time: Exponential")
    print("   • Problem: Inefficient for large graphs")
    
    print("\n2. **Bellman-Ford (1958):**")
    print("   • Dynamic programming approach")
    print("   • Relax all edges V-1 times")
    print("   • Handles negative weights")
    print("   • Time: O(V * E)")
    
    print("\n3. **Dijkstra's Algorithm (1959):**")
    print("   • Greedy approach for non-negative weights")
    print("   • Always extend shortest known path")
    print("   • Time: O(V²) originally, O(E log V) with heaps")
    
    print("\n4. **Floyd-Warshall (1962):**")
    print("   • All-pairs shortest paths")
    print("   • Dynamic programming on path length")
    print("   • Time: O(V³)")
    
    print("\n5. **SPFA (1994):**")
    print("   • Queue-based optimization of Bellman-Ford")
    print("   • Only process nodes that might improve distances")
    print("   • Average case much better than O(V * E)")
    
    print("\nKey Insights:")
    print("• **Greedy vs DP:** Dijkstra (greedy) vs Bellman-Ford (DP)")
    print("• **Data structures matter:** Heaps improve Dijkstra significantly")
    print("• **Problem constraints:** Negative weights change algorithm choice")
    print("• **Trade-offs:** Time vs space vs implementation complexity")

def compare_implementation_complexities():
    """Compare implementation complexities"""
    print("\n=== Implementation Complexity Comparison ===")
    
    print("Implementation Difficulty (1=Easy, 5=Hard):")
    
    print("\n• **Dijkstra's with Heap:** Difficulty 3/5")
    print("  - Need to understand priority queues")
    print("  - Handle duplicate entries in heap")
    print("  - Graph representation choices")
    
    print("\n• **Bellman-Ford:** Difficulty 2/5")
    print("  - Simple nested loops")
    print("  - Edge relaxation concept")
    print("  - Early termination optimization")
    
    print("\n• **Floyd-Warshall:** Difficulty 2/5")
    print("  - Triple nested loops")
    print("  - 2D array management")
    print("  - Intermediate vertex concept")
    
    print("\n• **DFS Memoization:** Difficulty 4/5")
    print("  - Recursive thinking required")
    print("  - Memoization strategy")
    print("  - Base case handling")
    
    print("\n• **SPFA:** Difficulty 3/5")
    print("  - Queue management")
    print("  - In-queue tracking")
    print("  - Cycle detection considerations")
    
    print("\nCommon Implementation Pitfalls:")
    print("• **Off-by-one errors:** 1-indexed vs 0-indexed nodes")
    print("• **Infinity handling:** Proper infinity value management")
    print("• **Graph representation:** Adjacency list vs matrix choice")
    print("• **Duplicate processing:** Handling nodes processed multiple times")
    print("• **Memory allocation:** Proper array/list sizing")
    
    print("\nBest Practices:")
    print("• Use clear variable names (dist, prev, visited)")
    print("• Handle edge cases (empty graph, single node)")
    print("• Add input validation")
    print("• Consider using built-in data structures")
    print("• Test with various graph types (sparse, dense, disconnected)")

if __name__ == "__main__":
    test_network_delay_time()
    demonstrate_dijkstra_algorithm()
    analyze_shortest_path_algorithms()
    demonstrate_algorithm_evolution()
    compare_implementation_complexities()

"""
Shortest Path Concepts:
1. Single-Source Shortest Path (SSSP)
2. Dijkstra's Algorithm for Non-negative Weights
3. Bellman-Ford for Negative Weights
4. All-Pairs Shortest Path with Floyd-Warshall
5. Algorithm Optimization Techniques

Key Problem Insights:
- Network delay = maximum shortest path from source
- All nodes must be reachable for valid solution
- Non-negative weights make Dijkstra optimal
- Priority queue implementation crucial for efficiency

Algorithm Strategy:
1. Model network as weighted directed graph
2. Use Dijkstra's algorithm for single-source shortest paths
3. Track maximum distance among all reachable nodes
4. Return -1 if any nodes unreachable

Dijkstra's Algorithm Steps:
1. Initialize distances (source = 0, others = infinity)
2. Use priority queue with (distance, node) pairs
3. Process nodes in order of shortest distance
4. Relax all outgoing edges from current node
5. Continue until all nodes processed

Real-world Applications:
- Computer network routing protocols
- GPS navigation systems
- Social network analysis
- Game pathfinding algorithms
- Supply chain optimization

This problem demonstrates fundamental shortest path
algorithms essential for graph analysis and optimization.
"""
