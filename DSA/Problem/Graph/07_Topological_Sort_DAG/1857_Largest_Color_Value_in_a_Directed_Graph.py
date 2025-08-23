"""
1857. Largest Color Value in a Directed Graph - Multiple Approaches
Difficulty: Medium

There is a directed graph of n colored nodes and m edges. The nodes are numbered from 0 to n - 1.

You are given a string colors where colors[i] is a lowercase English letter representing the color of the ith node in this graph (0-indexed). You are also given a 2D array edges where edges[j] = [aj, bj] indicates that there is a directed edge from node aj to node bj.

A valid path in the graph is a sequence of nodes x1 -> x2 -> x3 -> ... -> xk such that there is a directed edge from xi to xi+1 for every 1 <= i < k.

The color value of the path is the number of nodes in the path that are colored the most frequently occurring color along the path.

Return the largest color value of any valid path in the given graph, or -1 if the graph contains a cycle.
"""

from typing import List, Dict
from collections import defaultdict, deque

class LargestColorValue:
    """Multiple approaches to find largest color value in directed graph"""
    
    def largestPathValue_topological_dp(self, colors: str, edges: List[List[int]]) -> int:
        """
        Approach 1: Topological Sort with DP
        
        Use topological sort with DP to track color counts along paths.
        
        Time: O(V + E), Space: O(V * 26)
        """
        n = len(colors)
        
        # Build graph
        graph = defaultdict(list)
        in_degree = [0] * n
        
        for u, v in edges:
            graph[u].append(v)
            in_degree[v] += 1
        
        # DP table: dp[node][color] = max count of color in paths ending at node
        dp = [[0] * 26 for _ in range(n)]
        
        # Initialize DP for nodes with no incoming edges
        queue = deque()
        for i in range(n):
            if in_degree[i] == 0:
                queue.append(i)
                dp[i][ord(colors[i]) - ord('a')] = 1
        
        processed = 0
        max_color_value = 0
        
        while queue:
            node = queue.popleft()
            processed += 1
            
            # Update max color value
            for color_count in dp[node]:
                max_color_value = max(max_color_value, color_count)
            
            # Process neighbors
            for neighbor in graph[node]:
                # Update DP values for neighbor
                for color in range(26):
                    if color == ord(colors[neighbor]) - ord('a'):
                        dp[neighbor][color] = max(dp[neighbor][color], dp[node][color] + 1)
                    else:
                        dp[neighbor][color] = max(dp[neighbor][color], dp[node][color])
                
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycle
        return max_color_value if processed == n else -1
    
    def largestPathValue_dfs_memoization(self, colors: str, edges: List[List[int]]) -> int:
        """
        Approach 2: DFS with Memoization
        
        Use DFS with memoization and cycle detection.
        
        Time: O(V + E), Space: O(V * 26)
        """
        n = len(colors)
        
        # Build graph
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
        
        # Memoization: memo[node][color] = max count of color in paths starting from node
        memo = {}
        visiting = set()  # For cycle detection
        
        def dfs(node: int, color: int) -> int:
            """Return max count of given color in paths starting from node"""
            if (node, color) in memo:
                return memo[(node, color)]
            
            if node in visiting:  # Cycle detected
                return -1
            
            visiting.add(node)
            
            # Base case: current node contributes 1 if it matches the color
            current_contribution = 1 if ord(colors[node]) - ord('a') == color else 0
            max_count = current_contribution
            
            # Explore all neighbors
            for neighbor in graph[node]:
                neighbor_count = dfs(neighbor, color)
                if neighbor_count == -1:  # Cycle detected
                    visiting.remove(node)
                    return -1
                
                if ord(colors[neighbor]) - ord('a') == color:
                    max_count = max(max_count, current_contribution + neighbor_count)
                else:
                    max_count = max(max_count, neighbor_count)
            
            visiting.remove(node)
            memo[(node, color)] = max_count
            return max_count
        
        max_color_value = 0
        
        # Try starting from each node for each color
        for start in range(n):
            for color in range(26):
                result = dfs(start, color)
                if result == -1:  # Cycle detected
                    return -1
                max_color_value = max(max_color_value, result)
        
        return max_color_value
    
    def largestPathValue_optimized_topological(self, colors: str, edges: List[List[int]]) -> int:
        """
        Approach 3: Optimized Topological Sort
        
        Optimized version with better space and time management.
        
        Time: O(V + E), Space: O(V * 26)
        """
        n = len(colors)
        
        # Build adjacency list
        graph = [[] for _ in range(n)]
        in_degree = [0] * n
        
        for u, v in edges:
            graph[u].append(v)
            in_degree[v] += 1
        
        # DP table for color counts
        color_count = [[0] * 26 for _ in range(n)]
        
        # Initialize queue with nodes having no incoming edges
        queue = deque()
        for i in range(n):
            if in_degree[i] == 0:
                queue.append(i)
                color_count[i][ord(colors[i]) - ord('a')] = 1
        
        processed_nodes = 0
        result = 0
        
        while queue:
            node = queue.popleft()
            processed_nodes += 1
            
            # Update result with current node's maximum color count
            result = max(result, max(color_count[node]))
            
            # Process all neighbors
            for neighbor in graph[node]:
                # Update color counts for neighbor
                neighbor_color = ord(colors[neighbor]) - ord('a')
                
                for c in range(26):
                    if c == neighbor_color:
                        color_count[neighbor][c] = max(color_count[neighbor][c], 
                                                     color_count[node][c] + 1)
                    else:
                        color_count[neighbor][c] = max(color_count[neighbor][c], 
                                                     color_count[node][c])
                
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Return result if no cycle, -1 otherwise
        return result if processed_nodes == n else -1
    
    def largestPathValue_iterative_dfs(self, colors: str, edges: List[List[int]]) -> int:
        """
        Approach 4: Iterative DFS with Stack
        
        Use iterative DFS to avoid recursion depth issues.
        
        Time: O(V + E), Space: O(V * 26)
        """
        n = len(colors)
        
        # Build graph
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
        
        # Color states: WHITE = 0, GRAY = 1, BLACK = 2
        WHITE, GRAY, BLACK = 0, 1, 2
        color_state = [WHITE] * n
        
        # DP table for maximum color counts
        max_color_count = [[0] * 26 for _ in range(n)]
        
        def iterative_dfs(start: int) -> bool:
            """Return True if no cycle detected, False otherwise"""
            stack = [start]
            
            while stack:
                node = stack[-1]
                
                if color_state[node] == WHITE:
                    color_state[node] = GRAY
                    
                    # Add all unvisited neighbors to stack
                    for neighbor in graph[node]:
                        if color_state[neighbor] == GRAY:  # Back edge - cycle detected
                            return False
                        if color_state[neighbor] == WHITE:
                            stack.append(neighbor)
                
                elif color_state[node] == GRAY:
                    # All neighbors processed, compute DP values
                    color_state[node] = BLACK
                    
                    # Initialize with current node's color
                    node_color = ord(colors[node]) - ord('a')
                    max_color_count[node][node_color] = 1
                    
                    # Update based on neighbors
                    for neighbor in graph[node]:
                        for c in range(26):
                            if c == node_color:
                                max_color_count[node][c] = max(max_color_count[node][c],
                                                             max_color_count[neighbor][c] + 1)
                            else:
                                max_color_count[node][c] = max(max_color_count[node][c],
                                                             max_color_count[neighbor][c])
                    
                    stack.pop()
                
                else:  # BLACK
                    stack.pop()
            
            return True
        
        # Run DFS from all unvisited nodes
        for i in range(n):
            if color_state[i] == WHITE:
                if not iterative_dfs(i):
                    return -1  # Cycle detected
        
        # Find maximum color value
        result = 0
        for i in range(n):
            result = max(result, max(max_color_count[i]))
        
        return result
    
    def largestPathValue_kahns_optimized(self, colors: str, edges: List[List[int]]) -> int:
        """
        Approach 5: Kahn's Algorithm Optimized
        
        Optimized Kahn's algorithm with efficient color tracking.
        
        Time: O(V + E), Space: O(V * 26)
        """
        n = len(colors)
        if n == 0:
            return 0
        
        # Build graph
        adj = [[] for _ in range(n)]
        indegree = [0] * n
        
        for u, v in edges:
            adj[u].append(v)
            indegree[v] += 1
        
        # Color count DP
        count = [[0] * 26 for _ in range(n)]
        
        # Initialize queue
        queue = deque()
        for i in range(n):
            if indegree[i] == 0:
                queue.append(i)
                count[i][ord(colors[i]) - ord('a')] = 1
        
        processed = 0
        ans = 0
        
        while queue:
            u = queue.popleft()
            processed += 1
            ans = max(ans, max(count[u]))
            
            for v in adj[u]:
                # Update color counts for neighbor v
                for c in range(26):
                    count[v][c] = max(count[v][c], count[u][c] + (1 if ord(colors[v]) - ord('a') == c else 0))
                
                indegree[v] -= 1
                if indegree[v] == 0:
                    queue.append(v)
        
        return ans if processed == n else -1

def test_largest_color_value():
    """Test largest color value algorithms"""
    solver = LargestColorValue()
    
    test_cases = [
        ("abaca", [[0,1],[0,2],[2,3],[3,4]], 3, "Example 1"),
        ("a", [[0,0]], -1, "Self loop"),
        ("abc", [[0,1],[1,2]], 1, "Simple path"),
        ("aaaa", [[0,1],[1,2],[2,3]], 4, "All same color"),
        ("abcde", [], 1, "No edges"),
    ]
    
    algorithms = [
        ("Topological DP", solver.largestPathValue_topological_dp),
        ("DFS Memoization", solver.largestPathValue_dfs_memoization),
        ("Optimized Topological", solver.largestPathValue_optimized_topological),
        ("Kahn's Optimized", solver.largestPathValue_kahns_optimized),
    ]
    
    print("=== Testing Largest Color Value ===")
    
    for colors, edges, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"colors: '{colors}', edges: {edges}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(colors, edges)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Color value: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_largest_color_value()

"""
Largest Color Value demonstrates advanced topological sorting
with dynamic programming for path optimization problems
in directed graphs with node attributes.
"""
