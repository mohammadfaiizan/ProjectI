"""
1786. Number of Restricted Paths From First to Last Node - Multiple Approaches
Difficulty: Medium

There is an undirected weighted connected graph. You are given a positive integer n which denotes that the graph has n nodes labeled from 1 to n, and an array edges where each edges[i] = [ui, vi, weighti] denotes that there is an edge between nodes ui and vi with weight weighti.

A path from node start to node end is a sequence of nodes [start, d1, d2, ..., dk, end] where there exists an edge between each pair of consecutive nodes in the sequence.

A path is called a restricted path if it additionally satisfies that:
- The distance from the starting node to any node in this path is strictly increasing.
- The distance from any node in this path to the ending node is strictly decreasing.

Return the number of restricted paths from node 1 to node n. Since this number may be very large, return it modulo 10^9 + 7.
"""

from typing import List, Dict, Tuple
from collections import defaultdict
import heapq

class NumberOfRestrictedPaths:
    """Multiple approaches to count restricted paths"""
    
    def __init__(self):
        self.MOD = 10**9 + 7
    
    def countRestrictedPaths_dijkstra_dp(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 1: Dijkstra + DP with Memoization
        
        First compute shortest distances from node n, then use DP to count paths.
        
        Time: O(E log V + V), Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        # Compute shortest distances from node n using Dijkstra
        dist = self._dijkstra(graph, n, n)
        
        # DP with memoization to count restricted paths
        memo = {}
        
        def dp(node: int) -> int:
            if node == n:
                return 1
            
            if node in memo:
                return memo[node]
            
            count = 0
            for neighbor, _ in graph[node]:
                # Path is restricted if distance to n decreases
                if dist[neighbor] < dist[node]:
                    count = (count + dp(neighbor)) % self.MOD
            
            memo[node] = count
            return count
        
        return dp(1)
    
    def countRestrictedPaths_topological_dp(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 2: Topological Sort + DP
        
        Use topological ordering based on distances to compute DP.
        
        Time: O(E log V + V log V), Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        # Compute shortest distances from node n
        dist = self._dijkstra(graph, n, n)
        
        # Sort nodes by distance (topological order for DP)
        nodes_by_dist = sorted(range(1, n + 1), key=lambda x: dist[x])
        
        # DP array
        dp = [0] * (n + 1)
        dp[n] = 1
        
        # Process nodes in order of increasing distance from n
        for node in nodes_by_dist:
            for neighbor, _ in graph[node]:
                # If this creates a restricted path
                if dist[neighbor] < dist[node]:
                    dp[node] = (dp[node] + dp[neighbor]) % self.MOD
        
        return dp[1]
    
    def countRestrictedPaths_dijkstra_on_the_fly(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 3: Dijkstra with On-the-fly DP
        
        Combine Dijkstra with DP computation in single pass.
        
        Time: O(E log V), Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        # Dijkstra from node n to compute distances
        dist = [float('inf')] * (n + 1)
        dist[n] = 0
        pq = [(0, n)]
        
        # DP array to count paths
        dp = [0] * (n + 1)
        dp[n] = 1
        
        visited = set()
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            
            visited.add(u)
            
            # Update neighbors
            for v, w in graph[u]:
                new_dist = dist[u] + w
                
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
                    # Reset DP value since we found shorter path
                    dp[v] = 0
                
                # If this creates a restricted path (distance decreases towards n)
                if dist[v] == new_dist and dist[u] > dist[v]:
                    dp[v] = (dp[v] + dp[u]) % self.MOD
        
        return dp[1]
    
    def countRestrictedPaths_recursive_memoization(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 4: Recursive DP with Memoization
        
        Use recursive approach with memoization for cleaner code.
        
        Time: O(E log V + V), Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        # Compute distances from node n
        dist = self._dijkstra(graph, n, n)
        
        # Memoization for recursive DP
        memo = {}
        
        def count_paths(node: int) -> int:
            if node == n:
                return 1
            
            if node in memo:
                return memo[node]
            
            total = 0
            for neighbor, _ in graph[node]:
                if dist[neighbor] < dist[node]:
                    total = (total + count_paths(neighbor)) % self.MOD
            
            memo[node] = total
            return total
        
        return count_paths(1)
    
    def countRestrictedPaths_bfs_dp(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 5: BFS-based DP
        
        Use BFS to process nodes in order of distance.
        
        Time: O(E log V + V^2), Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        # Compute distances from node n
        dist = self._dijkstra(graph, n, n)
        
        # Group nodes by distance
        nodes_by_distance = defaultdict(list)
        for node in range(1, n + 1):
            nodes_by_distance[dist[node]].append(node)
        
        # DP processing in order of distance
        dp = [0] * (n + 1)
        dp[n] = 1
        
        # Process nodes in order of increasing distance from n
        for d in sorted(nodes_by_distance.keys()):
            for node in nodes_by_distance[d]:
                if node != n:
                    for neighbor, _ in graph[node]:
                        if dist[neighbor] < dist[node]:
                            dp[node] = (dp[node] + dp[neighbor]) % self.MOD
        
        return dp[1]
    
    def _dijkstra(self, graph: Dict, start: int, n: int) -> List[int]:
        """Helper function to compute shortest distances using Dijkstra"""
        dist = [float('inf')] * (n + 1)
        dist[start] = 0
        pq = [(0, start)]
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if d > dist[u]:
                continue
            
            for v, w in graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    heapq.heappush(pq, (dist[v], v))
        
        return dist

def test_restricted_paths():
    """Test restricted paths algorithms"""
    solver = NumberOfRestrictedPaths()
    
    test_cases = [
        (5, [[1,2,3],[1,3,3],[2,3,1],[1,4,2],[5,2,2],[3,5,1],[5,4,10]], 3, "Example 1"),
        (7, [[1,3,1],[4,1,2],[7,3,4],[2,5,3],[5,6,1],[6,7,2],[7,5,3],[2,6,4]], 1, "Example 2"),
        (2, [[1,2,1]], 1, "Simple case"),
    ]
    
    algorithms = [
        ("Dijkstra + DP", solver.countRestrictedPaths_dijkstra_dp),
        ("Topological DP", solver.countRestrictedPaths_topological_dp),
        ("Dijkstra On-the-fly", solver.countRestrictedPaths_dijkstra_on_the_fly),
        ("Recursive Memo", solver.countRestrictedPaths_recursive_memoization),
        ("BFS DP", solver.countRestrictedPaths_bfs_dp),
    ]
    
    print("=== Testing Number of Restricted Paths ===")
    
    for n, edges, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"n={n}, edges={edges}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(n, edges)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Paths: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_restricted_paths()

"""
Number of Restricted Paths demonstrates advanced shortest path problems
combined with dynamic programming for path counting with constraints.
"""
