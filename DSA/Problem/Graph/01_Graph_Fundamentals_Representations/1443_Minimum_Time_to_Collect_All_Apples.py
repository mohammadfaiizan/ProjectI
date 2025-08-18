"""
1443. Minimum Time to Collect All Apples in a Tree
Difficulty: Medium

Problem:
Given an undirected tree consisting of n vertices numbered from 0 to n-1, which has 
some apples in their vertices. You spend 1 second to walk over one edge of the tree. 
Return the minimum time in seconds you have to spend to collect all apples in the tree 
and go back to the vertex 0.

The edges of the undirected tree are given in the array edges, where edges[i] = [ai, bi] 
means that exists an edge connecting the vertices ai and bi. Additionally, there is a 
boolean array hasApple, where hasApple[i] = true means that vertex i has an apple; 
otherwise, it doesn't have an apple.

Examples:
Input: n = 7, edges = [[0,1],[0,2],[1,4],[1,5],[2,3],[2,6]], hasApple = [false,false,true,false,true,true,false]
Output: 8

Input: n = 7, edges = [[0,1],[0,2],[1,4],[1,5],[2,3],[2,6]], hasApple = [false,false,false,false,false,false,false]
Output: 0

Constraints:
- 1 <= n <= 10^5
- edges.length == n - 1
- edges[i].length == 2
- 0 <= ai, bi <= n - 1
- ai != bi
- hasApple.length == n
"""

from typing import List
from collections import defaultdict, deque

class Solution:
    def minTime_approach1_dfs_recursive(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        """
        Approach 1: DFS with Recursion
        
        Key insight: We need to visit a subtree only if it contains apples.
        For each subtree with apples, we need 2 seconds (go and return).
        
        Time: O(N) - visit each node once
        Space: O(N) - recursion stack and adjacency list
        """
        # Build adjacency list
        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        
        def dfs(node, parent):
            """
            Returns the time needed to collect all apples in subtree rooted at node
            """
            total_time = 0
            
            # Check all children
            for child in adj[node]:
                if child != parent:  # Avoid going back to parent
                    child_time = dfs(child, node)
                    
                    # If child subtree has apples (child_time > 0) or child has apple
                    if child_time > 0 or hasApple[child]:
                        total_time += child_time + 2  # +2 for go and return
            
            return total_time
        
        return dfs(0, -1)
    
    def minTime_approach2_dfs_iterative(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        """
        Approach 2: DFS with Iteration (Post-order)
        
        Use iterative DFS with post-order traversal to calculate subtree times.
        
        Time: O(N)
        Space: O(N)
        """
        if n == 1:
            return 0
        
        # Build adjacency list
        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        
        # DFS to get post-order traversal
        stack = [(0, -1, False)]  # (node, parent, visited)
        post_order = []
        
        while stack:
            node, parent, visited = stack.pop()
            
            if visited:
                post_order.append((node, parent))
            else:
                stack.append((node, parent, True))
                for child in adj[node]:
                    if child != parent:
                        stack.append((child, node, False))
        
        # Calculate time for each subtree
        subtree_time = [0] * n
        
        # Process in post-order (children before parents)
        for node, parent in post_order:
            for child in adj[node]:
                if child != parent:
                    # If child subtree has apples or child itself has apple
                    if subtree_time[child] > 0 or hasApple[child]:
                        subtree_time[node] += subtree_time[child] + 2
        
        return subtree_time[0]
    
    def minTime_approach3_bottom_up_dp(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        """
        Approach 3: Bottom-up Dynamic Programming
        
        Calculate whether each subtree needs to be visited and the cost.
        
        Time: O(N)
        Space: O(N)
        """
        # Build adjacency list
        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        
        # Find parent-child relationships using BFS
        parent = [-1] * n
        queue = deque([0])
        visited = {0}
        
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = node
                    queue.append(neighbor)
        
        # Bottom-up calculation
        needs_visit = hasApple[:]  # Start with nodes that have apples
        
        # Propagate upwards: if child needs visit, parent needs visit
        for node in range(n-1, -1, -1):  # Process in reverse order
            for child in adj[node]:
                if parent[child] == node and needs_visit[child]:
                    needs_visit[node] = True
        
        # Calculate total time
        total_time = 0
        for node in range(1, n):  # Skip root (node 0)
            if needs_visit[node]:
                total_time += 2  # 2 seconds to visit and return
        
        return total_time
    
    def minTime_approach4_path_marking(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        """
        Approach 4: Path marking approach
        
        Mark all paths from root to apple nodes, then count marked edges.
        
        Time: O(N)
        Space: O(N)
        """
        # Build adjacency list
        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        
        # Find all paths from root to apple nodes
        apple_paths = set()
        
        def dfs_find_paths(node, parent, path):
            path.append(node)
            
            if hasApple[node]:
                # Add all edges in path to apple_paths
                for i in range(len(path) - 1):
                    edge = tuple(sorted([path[i], path[i+1]]))
                    apple_paths.add(edge)
            
            for child in adj[node]:
                if child != parent:
                    dfs_find_paths(child, node, path)
            
            path.pop()
        
        dfs_find_paths(0, -1, [])
        
        # Each marked edge contributes 2 seconds (go and return)
        return len(apple_paths) * 2
    
    def minTime_approach5_optimized_single_pass(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        """
        Approach 5: Optimized single-pass solution
        
        Combine tree building and apple detection in one DFS pass.
        
        Time: O(N)
        Space: O(N)
        """
        # Build adjacency list
        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        
        def dfs(node, parent):
            """
            Returns (time_needed, has_apple_in_subtree)
            """
            time = 0
            has_apple_in_subtree = hasApple[node]
            
            for child in adj[node]:
                if child != parent:
                    child_time, child_has_apple = dfs(child, node)
                    
                    if child_has_apple:
                        time += child_time + 2
                        has_apple_in_subtree = True
            
            return time, has_apple_in_subtree
        
        time_needed, _ = dfs(0, -1)
        return time_needed

def test_min_time_collect_apples():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (n, edges, hasApple, expected)
        (7, [[0,1],[0,2],[1,4],[1,5],[2,3],[2,6]], [False,False,True,False,True,True,False], 8),
        (7, [[0,1],[0,2],[1,4],[1,5],[2,3],[2,6]], [False,False,False,False,False,False,False], 0),
        (4, [[0,1],[1,2],[0,3]], [True,True,True,True], 6),
        (4, [[0,1],[1,2],[0,3]], [False,False,True,False], 4),
        (1, [], [True], 0),  # Single node with apple
        (1, [], [False], 0), # Single node without apple
        (2, [[0,1]], [False,True], 2), # Simple case
    ]
    
    approaches = [
        ("DFS Recursive", solution.minTime_approach1_dfs_recursive),
        ("DFS Iterative", solution.minTime_approach2_dfs_iterative),
        ("Bottom-up DP", solution.minTime_approach3_bottom_up_dp),
        ("Path Marking", solution.minTime_approach4_path_marking),
        ("Optimized Single Pass", solution.minTime_approach5_optimized_single_pass),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, edges, hasApple, expected) in enumerate(test_cases):
            result = func(n, edges, hasApple)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} n={n}")
            print(f"         Edges: {edges}")
            print(f"         HasApple: {hasApple}")
            print(f"         Expected: {expected}, Got: {result}")

def demonstrate_apple_collection():
    """Demonstrate the apple collection process"""
    print("\n=== Apple Collection Demonstration ===")
    
    n = 7
    edges = [[0,1],[0,2],[1,4],[1,5],[2,3],[2,6]]
    hasApple = [False,False,True,False,True,True,False]
    
    print(f"Tree structure:")
    print(f"Nodes: {list(range(n))}")
    print(f"Edges: {edges}")
    print(f"Apples at nodes: {[i for i, has in enumerate(hasApple) if has]}")
    
    # Build tree representation
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    
    print(f"\nAdjacency representation:")
    for node in sorted(adj.keys()):
        apple_status = "🍎" if hasApple[node] else "  "
        print(f"  Node {node} {apple_status}: {sorted(adj[node])}")
    
    # Trace the optimal path
    def find_apple_paths(node, parent, path, all_paths):
        path.append(node)
        
        if hasApple[node]:
            all_paths.append(path[:])
        
        for child in adj[node]:
            if child != parent:
                find_apple_paths(child, node, path, all_paths)
        
        path.pop()
    
    apple_paths = []
    find_apple_paths(0, -1, [], apple_paths)
    
    print(f"\nPaths from root to apples:")
    total_edges = set()
    for i, path in enumerate(apple_paths):
        print(f"  Path {i+1}: {' -> '.join(map(str, path))}")
        for j in range(len(path) - 1):
            edge = tuple(sorted([path[j], path[j+1]]))
            total_edges.add(edge)
    
    print(f"\nUnique edges needed: {sorted(total_edges)}")
    print(f"Total time: {len(total_edges) * 2} seconds")

if __name__ == "__main__":
    test_min_time_collect_apples()
    demonstrate_apple_collection()

"""
Graph Theory Concepts:
1. Tree Traversal (DFS/BFS)
2. Post-order Processing
3. Subtree Properties
4. Path Optimization in Trees

Key Insights:
- Tree structure guarantees unique paths between any two nodes
- Need to visit a subtree only if it contains apples
- Each edge traversed contributes 2 seconds (go and return)
- Problem reduces to finding minimal connected subgraph containing all apples

Algorithm Analysis:
- All approaches are O(N) time complexity
- DFS recursive is most intuitive and elegant
- Post-order processing ensures children are computed before parents
- Path marking provides alternative perspective on the problem

Real-world Applications:
- Resource collection in networks
- Delivery route optimization
- Network maintenance scheduling
- File system traversal
- Dependency resolution in trees
"""
