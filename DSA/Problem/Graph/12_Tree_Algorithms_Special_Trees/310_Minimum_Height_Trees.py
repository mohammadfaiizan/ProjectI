"""
310. Minimum Height Trees - Multiple Approaches
Difficulty: Medium

A tree is an undirected graph in which any two vertices are connected by exactly one path.
Given such a tree with n nodes labeled from 0 to n - 1, find all possible roots which give minimum height trees.
"""

from typing import List, Set
from collections import defaultdict, deque

class MinimumHeightTrees:
    """Find roots that give minimum height trees"""
    
    def findMinHeightTrees_center_finding(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 1: Tree Center Finding (Optimal)
        
        The center(s) of a tree minimize the maximum distance to any node.
        
        Time: O(n), Space: O(n)
        """
        if n == 1:
            return [0]
        
        # Build adjacency list
        graph = defaultdict(set)
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)
        
        # Initialize leaves
        leaves = deque([i for i in range(n) if len(graph[i]) == 1])
        remaining = n
        
        # Remove leaves layer by layer
        while remaining > 2:
            leaf_count = len(leaves)
            remaining -= leaf_count
            
            for _ in range(leaf_count):
                leaf = leaves.popleft()
                
                # Remove leaf from its neighbor
                neighbor = graph[leaf].pop()
                graph[neighbor].remove(leaf)
                
                # If neighbor becomes a leaf, add it to queue
                if len(graph[neighbor]) == 1:
                    leaves.append(neighbor)
        
        return list(leaves)
    
    def findMinHeightTrees_brute_force(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 2: Brute Force - Try All Roots
        
        Calculate height for each possible root and find minimum.
        
        Time: O(n^2), Space: O(n)
        """
        if n == 1:
            return [0]
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def get_height(root):
            """Calculate height of tree rooted at given node"""
            visited = {root}
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
        
        # Find minimum height
        heights = [get_height(i) for i in range(n)]
        min_height = min(heights)
        
        return [i for i in range(n) if heights[i] == min_height]

def test_minimum_height_trees():
    """Test minimum height tree algorithms"""
    solver = MinimumHeightTrees()
    
    test_cases = [
        (4, [[1,0],[1,2],[1,3]], [1], "Star graph"),
        (6, [[3,0],[3,1],[3,2],[3,4],[5,4]], [3,4], "Two centers"),
        (1, [], [0], "Single node"),
        (2, [[0,1]], [0,1], "Two nodes"),
    ]
    
    algorithms = [
        ("Center Finding", solver.findMinHeightTrees_center_finding),
        ("Brute Force", solver.findMinHeightTrees_brute_force),
    ]
    
    print("=== Testing Minimum Height Trees ===")
    
    for n, edges, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        
        for alg_name, alg_func in algorithms:
            try:
                result = sorted(alg_func(n, edges))
                expected_sorted = sorted(expected)
                status = "✓" if result == expected_sorted else "✗"
                print(f"{alg_name:15} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_minimum_height_trees()

"""
Minimum Height Trees demonstrates the concept of tree centers,
which are the optimal roots for minimizing tree height.
The center-finding approach is optimal with O(n) complexity.
"""
