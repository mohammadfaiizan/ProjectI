"""
847. Shortest Path Visiting All Nodes - Multiple Approaches
Difficulty: Medium

You have an undirected, connected graph of n nodes labeled from 0 to n - 1. 
You are given an array graph where graph[i] is a list of all the nodes connected to node i.

Return the length of the shortest path that visits every node. You may start and end at any node.
"""

from typing import List, Dict, Set, Tuple
from collections import deque
import heapq

class ShortestPathVisitingAllNodes:
    """Multiple approaches to find shortest path visiting all nodes"""
    
    def shortestPathLength_bfs_bitmask(self, graph: List[List[int]]) -> int:
        """
        Approach 1: BFS with Bitmask DP
        
        Use BFS with state (node, visited_mask) to find shortest path.
        
        Time: O(n^2 * 2^n), Space: O(n * 2^n)
        """
        n = len(graph)
        if n == 1:
            return 0
        
        # State: (node, visited_mask)
        # visited_mask: bit i is 1 if node i has been visited
        target_mask = (1 << n) - 1  # All nodes visited
        
        # Initialize BFS from all nodes
        queue = deque()
        visited = set()
        
        for i in range(n):
            initial_mask = 1 << i
            queue.append((i, initial_mask, 0))  # (node, mask, distance)
            visited.add((i, initial_mask))
        
        while queue:
            node, mask, dist = queue.popleft()
            
            # If all nodes visited, return distance
            if mask == target_mask:
                return dist
            
            # Explore neighbors
            for neighbor in graph[node]:
                new_mask = mask | (1 << neighbor)
                state = (neighbor, new_mask)
                
                if state not in visited:
                    visited.add(state)
                    queue.append((neighbor, new_mask, dist + 1))
        
        return -1  # Should not reach here for connected graph
    
    def shortestPathLength_dijkstra_bitmask(self, graph: List[List[int]]) -> int:
        """
        Approach 2: Dijkstra with Bitmask
        
        Use Dijkstra's algorithm with state compression.
        
        Time: O(n^2 * 2^n * log(n * 2^n)), Space: O(n * 2^n)
        """
        n = len(graph)
        if n == 1:
            return 0
        
        target_mask = (1 << n) - 1
        
        # Priority queue: (distance, node, visited_mask)
        pq = []
        visited = set()
        
        # Start from all nodes
        for i in range(n):
            initial_mask = 1 << i
            heapq.heappush(pq, (0, i, initial_mask))
        
        while pq:
            dist, node, mask = heapq.heappop(pq)
            
            state = (node, mask)
            if state in visited:
                continue
            
            visited.add(state)
            
            if mask == target_mask:
                return dist
            
            # Explore neighbors
            for neighbor in graph[node]:
                new_mask = mask | (1 << neighbor)
                new_state = (neighbor, new_mask)
                
                if new_state not in visited:
                    heapq.heappush(pq, (dist + 1, neighbor, new_mask))
        
        return -1
    
    def shortestPathLength_dp_bitmask(self, graph: List[List[int]]) -> int:
        """
        Approach 3: Dynamic Programming with Bitmask
        
        Use DP where dp[mask][i] = min distance to visit nodes in mask ending at i.
        
        Time: O(n^2 * 2^n), Space: O(n * 2^n)
        """
        n = len(graph)
        if n == 1:
            return 0
        
        # dp[mask][i] = minimum distance to visit nodes in mask, ending at node i
        dp = [[float('inf')] * n for _ in range(1 << n)]
        
        # Initialize: starting at each node
        for i in range(n):
            dp[1 << i][i] = 0
        
        # Fill DP table
        for mask in range(1 << n):
            for u in range(n):
                if dp[mask][u] == float('inf'):
                    continue
                
                # Try to extend to neighbors
                for v in graph[u]:
                    new_mask = mask | (1 << v)
                    dp[new_mask][v] = min(dp[new_mask][v], dp[mask][u] + 1)
        
        # Find minimum distance visiting all nodes
        target_mask = (1 << n) - 1
        return min(dp[target_mask])
    
    def shortestPathLength_floyd_warshall_tsp(self, graph: List[List[int]]) -> int:
        """
        Approach 4: Floyd-Warshall + TSP DP
        
        First compute all-pairs shortest paths, then solve TSP.
        
        Time: O(n^3 + n^2 * 2^n), Space: O(n^2 + n * 2^n)
        """
        n = len(graph)
        if n == 1:
            return 0
        
        # Build distance matrix using Floyd-Warshall
        dist = [[float('inf')] * n for _ in range(n)]
        
        # Initialize distances
        for i in range(n):
            dist[i][i] = 0
            for j in graph[i]:
                dist[i][j] = 1
        
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        # TSP DP
        dp = [[float('inf')] * n for _ in range(1 << n)]
        
        # Initialize
        for i in range(n):
            dp[1 << i][i] = 0
        
        # Fill DP
        for mask in range(1 << n):
            for u in range(n):
                if dp[mask][u] == float('inf'):
                    continue
                
                for v in range(n):
                    if mask & (1 << v):
                        continue
                    
                    new_mask = mask | (1 << v)
                    dp[new_mask][v] = min(dp[new_mask][v], dp[mask][u] + dist[u][v])
        
        # Find minimum
        target_mask = (1 << n) - 1
        return min(dp[target_mask])
    
    def shortestPathLength_branch_bound(self, graph: List[List[int]]) -> int:
        """
        Approach 5: Branch and Bound
        
        Use branch and bound with pruning for optimization.
        
        Time: Exponential with pruning, Space: O(n)
        """
        n = len(graph)
        if n == 1:
            return 0
        
        # Precompute shortest distances between all pairs
        dist = self._compute_all_pairs_shortest_path(graph)
        
        self.min_path_length = float('inf')
        target_mask = (1 << n) - 1
        
        def branch_bound(current_node: int, visited_mask: int, current_length: int):
            if visited_mask == target_mask:
                self.min_path_length = min(self.min_path_length, current_length)
                return
            
            # Pruning: if current length + lower bound >= best known, prune
            lower_bound = self._calculate_lower_bound(visited_mask, current_node, dist, n)
            if current_length + lower_bound >= self.min_path_length:
                return
            
            # Branch to unvisited nodes
            for next_node in range(n):
                if not (visited_mask & (1 << next_node)):
                    new_mask = visited_mask | (1 << next_node)
                    new_length = current_length + dist[current_node][next_node]
                    branch_bound(next_node, new_mask, new_length)
        
        # Try starting from each node
        for start in range(n):
            branch_bound(start, 1 << start, 0)
        
        return self.min_path_length
    
    def _compute_all_pairs_shortest_path(self, graph: List[List[int]]) -> List[List[int]]:
        """Compute shortest paths between all pairs using BFS"""
        n = len(graph)
        dist = [[float('inf')] * n for _ in range(n)]
        
        for start in range(n):
            # BFS from start
            queue = deque([(start, 0)])
            visited = {start}
            dist[start][start] = 0
            
            while queue:
                node, d = queue.popleft()
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        dist[start][neighbor] = d + 1
                        queue.append((neighbor, d + 1))
        
        return dist
    
    def _calculate_lower_bound(self, visited_mask: int, current_node: int, 
                             dist: List[List[int]], n: int) -> int:
        """Calculate lower bound for remaining path length"""
        unvisited = []
        for i in range(n):
            if not (visited_mask & (1 << i)):
                unvisited.append(i)
        
        if not unvisited:
            return 0
        
        # Simple lower bound: minimum distance to nearest unvisited + MST of unvisited
        min_to_unvisited = min(dist[current_node][u] for u in unvisited)
        
        # Approximate MST of unvisited nodes
        if len(unvisited) <= 1:
            return min_to_unvisited
        
        mst_cost = 0
        remaining = set(unvisited[1:])
        current_mst = {unvisited[0]}
        
        while remaining:
            min_edge = float('inf')
            next_node = -1
            
            for u in current_mst:
                for v in remaining:
                    if dist[u][v] < min_edge:
                        min_edge = dist[u][v]
                        next_node = v
            
            if next_node != -1:
                mst_cost += min_edge
                current_mst.add(next_node)
                remaining.remove(next_node)
        
        return min_to_unvisited + mst_cost

def test_shortest_path_visiting_all():
    """Test shortest path visiting all nodes algorithms"""
    solver = ShortestPathVisitingAllNodes()
    
    test_cases = [
        ([[1,2,3],[0],[0],[0]], 4, "Star graph"),
        ([[1],[0,2,4],[1,3,4],[2],[1,2]], 4, "Complex graph"),
        ([[1,2,3],[0,2,3],[0,1,3],[0,1,2]], 4, "Complete graph K4"),
        ([[]], 0, "Single node"),
        ([[1],[0]], 1, "Two nodes"),
    ]
    
    algorithms = [
        ("BFS Bitmask", solver.shortestPathLength_bfs_bitmask),
        ("Dijkstra Bitmask", solver.shortestPathLength_dijkstra_bitmask),
        ("DP Bitmask", solver.shortestPathLength_dp_bitmask),
        ("Floyd-Warshall TSP", solver.shortestPathLength_floyd_warshall_tsp),
    ]
    
    print("=== Testing Shortest Path Visiting All Nodes ===")
    
    for graph, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Graph: {graph}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(graph)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Length: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

def demonstrate_tsp_concepts():
    """Demonstrate TSP and related concepts"""
    print("\n=== TSP and Graph Traversal Concepts ===")
    
    print("Problem Characteristics:")
    print("• Visit every node at least once")
    print("• Minimize total path length")
    print("• Can start and end at any node")
    print("• Different from classic TSP (can revisit nodes)")
    
    print("\nKey Techniques:")
    print("• Bitmask DP: Track visited nodes with bits")
    print("• State compression: (node, visited_set)")
    print("• BFS/Dijkstra: Find shortest paths between states")
    print("• Branch and bound: Prune suboptimal branches")
    
    print("\nComplexity Analysis:")
    print("• State space: O(n * 2^n)")
    print("• Time: O(n^2 * 2^n) for DP approaches")
    print("• Space: O(n * 2^n) for memoization")
    print("• Exponential but manageable for small n")

if __name__ == "__main__":
    test_shortest_path_visiting_all()
    demonstrate_tsp_concepts()

"""
Shortest Path Visiting All Nodes demonstrates advanced graph traversal
techniques combining shortest path algorithms with dynamic programming
and state compression for solving TSP-like problems efficiently.
"""
