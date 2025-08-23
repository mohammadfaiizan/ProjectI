"""
310. Minimum Height Trees - Multiple Approaches
Difficulty: Medium

A tree is an undirected graph in which any two vertices are connected by exactly one path. In other words, any connected graph without simple cycles is a tree.

Given a tree of n nodes labelled from 0 to n - 1, and an array of n - 1 edges where edges[i] = [ai, bi] indicates that there is an undirected edge between the two nodes ai and bi in the tree, you can choose any node of the tree as the root. When you pick a node x as the root, the height of the tree is the number of edges on the longest downward path from x.

Return a list of all MHTs (minimum height trees) root candidates. If there are multiple MHTs, return all of them in any order.
"""

from typing import List
from collections import defaultdict, deque

class MinimumHeightTrees:
    """Multiple approaches to find minimum height tree roots"""
    
    def findMinHeightTrees_leaf_removal(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 1: Iterative Leaf Removal (Topological Sort)
        
        Remove leaves iteratively until 1-2 nodes remain (centroids).
        
        Time: O(V), Space: O(V)
        """
        if n == 1:
            return [0]
        
        # Build adjacency list
        graph = defaultdict(set)
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)
        
        # Initialize leaves (nodes with degree 1)
        leaves = deque()
        for i in range(n):
            if len(graph[i]) == 1:
                leaves.append(i)
        
        remaining_nodes = n
        
        # Remove leaves iteratively
        while remaining_nodes > 2:
            leaf_count = len(leaves)
            remaining_nodes -= leaf_count
            
            # Remove current leaves
            for _ in range(leaf_count):
                leaf = leaves.popleft()
                
                # Remove leaf from its neighbor
                neighbor = next(iter(graph[leaf]))
                graph[neighbor].remove(leaf)
                
                # If neighbor becomes a leaf, add to queue
                if len(graph[neighbor]) == 1:
                    leaves.append(neighbor)
        
        # Remaining nodes are the centroids (MHT roots)
        return list(range(n)) if remaining_nodes == n else [node for node in range(n) if len(graph[node]) > 0]
    
    def findMinHeightTrees_bfs_height_calculation(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 2: BFS Height Calculation for Each Node
        
        Calculate height for each possible root using BFS.
        
        Time: O(V²), Space: O(V)
        """
        if n == 1:
            return [0]
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def calculate_height(root: int) -> int:
            """Calculate height of tree rooted at given node"""
            visited = set([root])
            queue = deque([(root, 0)])
            max_height = 0
            
            while queue:
                node, height = queue.popleft()
                max_height = max(max_height, height)
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, height + 1))
            
            return max_height
        
        # Calculate height for each possible root
        min_height = float('inf')
        heights = {}
        
        for i in range(n):
            height = calculate_height(i)
            heights[i] = height
            min_height = min(min_height, height)
        
        # Return all nodes with minimum height
        return [node for node, height in heights.items() if height == min_height]
    
    def findMinHeightTrees_tree_centroid(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 3: Tree Centroid Finding
        
        Find tree centroids using subtree size calculation.
        
        Time: O(V), Space: O(V)
        """
        if n == 1:
            return [0]
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        # Calculate subtree sizes
        subtree_size = [0] * n
        visited = [False] * n
        
        def dfs_size(node: int, parent: int) -> int:
            """Calculate subtree size rooted at node"""
            visited[node] = True
            size = 1
            
            for neighbor in graph[node]:
                if neighbor != parent and not visited[neighbor]:
                    size += dfs_size(neighbor, node)
            
            subtree_size[node] = size
            return size
        
        # Start DFS from node 0
        dfs_size(0, -1)
        
        # Find centroids
        centroids = []
        
        def find_centroids(node: int, parent: int, total_nodes: int):
            """Find centroid nodes"""
            is_centroid = True
            max_subtree = 0
            
            for neighbor in graph[node]:
                if neighbor != parent:
                    # Size of subtree rooted at neighbor
                    subtree = subtree_size[neighbor] if subtree_size[neighbor] < subtree_size[node] else total_nodes - subtree_size[node]
                    max_subtree = max(max_subtree, subtree)
                    
                    if subtree > total_nodes // 2:
                        is_centroid = False
            
            # Check parent subtree
            if parent != -1:
                parent_subtree = total_nodes - subtree_size[node]
                max_subtree = max(max_subtree, parent_subtree)
                if parent_subtree > total_nodes // 2:
                    is_centroid = False
            
            if is_centroid:
                centroids.append(node)
            
            for neighbor in graph[node]:
                if neighbor != parent:
                    find_centroids(neighbor, node, total_nodes)
        
        find_centroids(0, -1, n)
        return centroids
    
    def findMinHeightTrees_diameter_approach(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 4: Tree Diameter Approach
        
        Find tree diameter and return middle node(s).
        
        Time: O(V), Space: O(V)
        """
        if n == 1:
            return [0]
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def bfs_farthest(start: int) -> tuple:
            """Find farthest node from start and its distance"""
            visited = set([start])
            queue = deque([(start, 0)])
            farthest_node = start
            max_distance = 0
            
            while queue:
                node, dist = queue.popleft()
                
                if dist > max_distance:
                    max_distance = dist
                    farthest_node = node
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            
            return farthest_node, max_distance
        
        def find_path(start: int, end: int) -> List[int]:
            """Find path between start and end nodes"""
            if start == end:
                return [start]
            
            visited = set([start])
            queue = deque([(start, [start])])
            
            while queue:
                node, path = queue.popleft()
                
                for neighbor in graph[node]:
                    if neighbor == end:
                        return path + [neighbor]
                    
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
            
            return []
        
        # Find one end of diameter
        end1, _ = bfs_farthest(0)
        
        # Find other end of diameter
        end2, diameter = bfs_farthest(end1)
        
        # Find path between diameter ends
        diameter_path = find_path(end1, end2)
        
        # Return middle node(s) of diameter
        path_length = len(diameter_path)
        if path_length % 2 == 1:
            return [diameter_path[path_length // 2]]
        else:
            mid = path_length // 2
            return [diameter_path[mid - 1], diameter_path[mid]]
    
    def findMinHeightTrees_optimized_leaf_removal(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 5: Optimized Leaf Removal
        
        Optimized version of leaf removal with better implementation.
        
        Time: O(V), Space: O(V)
        """
        if n == 1:
            return [0]
        if n == 2:
            return [0, 1]
        
        # Build adjacency list with degree tracking
        graph = [set() for _ in range(n)]
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)
        
        # Find initial leaves
        leaves = []
        for i in range(n):
            if len(graph[i]) == 1:
                leaves.append(i)
        
        remaining = n
        
        # Remove leaves layer by layer
        while remaining > 2:
            remaining -= len(leaves)
            new_leaves = []
            
            for leaf in leaves:
                # Get the only neighbor of this leaf
                neighbor = graph[leaf].pop()
                graph[neighbor].remove(leaf)
                
                # If neighbor becomes a leaf, add to next layer
                if len(graph[neighbor]) == 1:
                    new_leaves.append(neighbor)
            
            leaves = new_leaves
        
        return leaves

def test_minimum_height_trees():
    """Test minimum height trees algorithms"""
    solver = MinimumHeightTrees()
    
    test_cases = [
        (4, [[1,0],[1,2],[1,3]], [1], "Star graph"),
        (6, [[3,0],[3,1],[3,2],[3,4],[5,4]], [3,4], "Two centroids"),
        (1, [], [0], "Single node"),
        (2, [[0,1]], [0,1], "Two nodes"),
        (7, [[0,1],[1,2],[1,3],[2,4],[3,5],[4,6]], [1,2], "Linear-like tree"),
    ]
    
    algorithms = [
        ("Leaf Removal", solver.findMinHeightTrees_leaf_removal),
        ("BFS Height Calc", solver.findMinHeightTrees_bfs_height_calculation),
        ("Diameter Approach", solver.findMinHeightTrees_diameter_approach),
        ("Optimized Leaf Removal", solver.findMinHeightTrees_optimized_leaf_removal),
    ]
    
    print("=== Testing Minimum Height Trees ===")
    
    for n, edges, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"n={n}, edges={edges}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(n, edges)
                # Sort both for comparison since order doesn't matter
                result_sorted = sorted(result)
                expected_sorted = sorted(expected)
                status = "✓" if result_sorted == expected_sorted else "✗"
                print(f"{alg_name:22} | {status} | Roots: {result}")
            except Exception as e:
                print(f"{alg_name:22} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_minimum_height_trees()

"""
Minimum Height Trees demonstrates tree centroid finding
and topological sorting concepts for tree optimization
problems with multiple algorithmic approaches.
"""
