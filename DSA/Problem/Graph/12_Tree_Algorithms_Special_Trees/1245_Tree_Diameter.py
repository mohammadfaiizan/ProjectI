"""
1245. Tree Diameter - Multiple Approaches
Difficulty: Medium

The diameter of a tree is the number of edges in the longest path between any two nodes.
Given an undirected tree, return its diameter.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque

class TreeDiameter:
    """Multiple approaches to find tree diameter"""
    
    def treeDiameter_two_bfs(self, edges: List[List[int]]) -> int:
        """
        Approach 1: Two BFS Method (Optimal for Trees)
        
        1. BFS from any node to find farthest node
        2. BFS from that node to find actual diameter
        
        Time: O(n), Space: O(n)
        """
        if not edges:
            return 0
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def bfs_farthest(start):
            """BFS to find farthest node and distance"""
            queue = deque([(start, 0)])
            visited = {start}
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
        
        # First BFS from node 0
        endpoint1, _ = bfs_farthest(0)
        
        # Second BFS from farthest node found
        endpoint2, diameter = bfs_farthest(endpoint1)
        
        return diameter
    
    def treeDiameter_dfs_recursive(self, edges: List[List[int]]) -> int:
        """
        Approach 2: DFS with Global Maximum
        
        Time: O(n), Space: O(n)
        """
        if not edges:
            return 0
        
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        self.max_diameter = 0
        
        def dfs(node, parent):
            """DFS returning height of subtree"""
            max_depth1 = max_depth2 = 0
            
            for neighbor in graph[node]:
                if neighbor != parent:
                    depth = dfs(neighbor, node)
                    
                    if depth > max_depth1:
                        max_depth2 = max_depth1
                        max_depth1 = depth
                    elif depth > max_depth2:
                        max_depth2 = depth
            
            # Update global diameter
            self.max_diameter = max(self.max_diameter, max_depth1 + max_depth2)
            
            return max_depth1 + 1
        
        dfs(0, -1)
        return self.max_diameter

def test_tree_diameter():
    """Test tree diameter algorithms"""
    solver = TreeDiameter()
    
    test_cases = [
        ([[0,1],[0,2]], 2, "Simple path"),
        ([[0,1],[1,2],[2,3],[1,4],[4,5]], 4, "Complex tree"),
        ([], 0, "Empty tree"),
        ([[0,1]], 1, "Single edge"),
    ]
    
    algorithms = [
        ("Two BFS", solver.treeDiameter_two_bfs),
        ("DFS Recursive", solver.treeDiameter_dfs_recursive),
    ]
    
    print("=== Testing Tree Diameter ===")
    
    for edges, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(edges)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:15} | {status} | Diameter: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_tree_diameter()
