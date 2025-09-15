"""
Problem: Tree Diameter
URL: https://leetcode.com/problems/tree-diameter/

Problem Statement:
The diameter of a tree is the number of edges in the longest path between any two nodes in the tree.
You are given an array of edges where edges[i] = [ai, bi] indicates that there is an undirected edge between nodes ai and bi.
Return the diameter of the tree.

Sample Input/Output:
Input: edges = [[0,1],[0,2]]
Output: 2
Explanation: The longest path is 0 -> 1 or 0 -> 2, which has length 2.

Input: edges = [[0,1],[1,2],[2,3],[1,4],[4,5]]
Output: 4
Explanation: The longest path is 3 -> 2 -> 1 -> 4 -> 5, which has length 4.
"""

from typing import List, Dict, Set
from collections import defaultdict, deque

class Solution:
    def Tree_Diameter_Two_DFS(self, edges: List[List[int]]) -> int:
        """
        Two DFS - Find farthest node, then find farthest from that
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not edges:
            return 0
        
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def DFS_Farthest(start: int) -> tuple:
            visited = set()
            max_dist = 0
            farthest_node = start
            
            def DFS(node: int, dist: int) -> None:
                nonlocal max_dist, farthest_node
                visited.add(node)
                
                if dist > max_dist:
                    max_dist = dist
                    farthest_node = node
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        DFS(neighbor, dist + 1)
            
            DFS(start, 0)
            return farthest_node, max_dist
        
        start_node = next(iter(graph.keys()))
        one_end, _ = DFS_Farthest(start_node)
        other_end, diameter = DFS_Farthest(one_end)
        
        return diameter
    
    def Tree_Diameter_DP_Recursive(self, edges: List[List[int]]) -> int:
        """
        DP Recursive - Each node returns max depth
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not edges:
            return 0
        
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        max_diameter = 0
        
        def DFS_Depth(node: int, parent: int) -> int:
            nonlocal max_diameter
            
            depths = []
            
            for neighbor in graph[node]:
                if neighbor != parent:
                    depth = DFS_Depth(neighbor, node)
                    depths.append(depth)
            
            depths.sort(reverse=True)
            
            if len(depths) >= 2:
                current_diameter = depths[0] + depths[1] + 2
            elif len(depths) == 1:
                current_diameter = depths[0] + 1
            else:
                current_diameter = 0
            
            max_diameter = max(max_diameter, current_diameter)
            
            return depths[0] + 1 if depths else 0
        
        start_node = next(iter(graph.keys()))
        DFS_Depth(start_node, -1)
        
        return max_diameter
    
    def Tree_Diameter_BFS_Optimal(self, edges: List[List[int]]) -> int:
        """
        BFS Optimal - Two BFS to find diameter
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not edges:
            return 0
        
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def BFS_Farthest(start: int) -> tuple:
            visited = set()
            queue = deque([(start, 0)])
            visited.add(start)
            max_dist = 0
            farthest_node = start
            
            while queue:
                node, dist = queue.popleft()
                
                if dist > max_dist:
                    max_dist = dist
                    farthest_node = node
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            
            return farthest_node, max_dist
        
        start_node = next(iter(graph.keys()))
        one_end, _ = BFS_Farthest(start_node)
        other_end, diameter = BFS_Farthest(one_end)
        
        return diameter
    
    def Tree_Diameter_With_Path(self, edges: List[List[int]]) -> tuple:
        """
        With Path - Return diameter and actual path
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not edges:
            return 0, []
        
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def BFS_With_Path(start: int) -> tuple:
            visited = set()
            queue = deque([(start, 0, [start])])
            visited.add(start)
            max_dist = 0
            farthest_node = start
            longest_path = [start]
            
            while queue:
                node, dist, path = queue.popleft()
                
                if dist > max_dist:
                    max_dist = dist
                    farthest_node = node
                    longest_path = path[:]
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1, path + [neighbor]))
            
            return farthest_node, max_dist, longest_path
        
        start_node = next(iter(graph.keys()))
        one_end, _, _ = BFS_With_Path(start_node)
        other_end, diameter, diameter_path = BFS_With_Path(one_end)
        
        return diameter, diameter_path
    
    def Tree_Diameter_Center_Based(self, edges: List[List[int]]) -> int:
        """
        Center Based - Find tree centers and compute diameter
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not edges:
            return 0
        
        if len(edges) == 1:
            return 1
        
        graph = defaultdict(list)
        degree = defaultdict(int)
        
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
            degree[u] += 1
            degree[v] += 1
        
        leaves = deque([node for node in degree if degree[node] == 1])
        remaining_nodes = len(degree)
        
        while remaining_nodes > 2:
            leaf_count = len(leaves)
            remaining_nodes -= leaf_count
            
            for _ in range(leaf_count):
                leaf = leaves.popleft()
                
                for neighbor in graph[leaf]:
                    degree[neighbor] -= 1
                    if degree[neighbor] == 1:
                        leaves.append(neighbor)
        
        centers = list(leaves)
        
        def Max_Depth_From_Node(start: int) -> int:
            visited = set()
            max_depth = 0
            
            def DFS(node: int, depth: int) -> None:
                nonlocal max_depth
                visited.add(node)
                max_depth = max(max_depth, depth)
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        DFS(neighbor, depth + 1)
            
            DFS(start, 0)
            return max_depth
        
        if len(centers) == 1:
            return 2 * Max_Depth_From_Node(centers[0])
        else:
            return 2 * Max_Depth_From_Node(centers[0]) + 1
    
    def Tree_Diameter_Memoized(self, edges: List[List[int]]) -> int:
        """
        Memoized - Cache results for repeated subproblems
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not edges:
            return 0
        
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        memo = {}
        max_diameter = 0
        
        def Max_Depth_From_Node(node: int, parent: int) -> int:
            if (node, parent) in memo:
                return memo[(node, parent)]
            
            depths = []
            
            for neighbor in graph[node]:
                if neighbor != parent:
                    depth = Max_Depth_From_Node(neighbor, node)
                    depths.append(depth)
            
            depths.sort(reverse=True)
            
            nonlocal max_diameter
            if len(depths) >= 2:
                current_diameter = depths[0] + depths[1] + 2
            elif len(depths) == 1:
                current_diameter = depths[0] + 1
            else:
                current_diameter = 0
            
            max_diameter = max(max_diameter, current_diameter)
            
            result = depths[0] + 1 if depths else 0
            memo[(node, parent)] = result
            return result
        
        start_node = next(iter(graph.keys()))
        Max_Depth_From_Node(start_node, -1)
        
        return max_diameter
    
    def Tree_Diameter_Iterative_DFS(self, edges: List[List[int]]) -> int:
        """
        Iterative DFS - Avoid recursion using stack
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not edges:
            return 0
        
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def DFS_Iterative_Farthest(start: int) -> tuple:
            stack = [(start, 0)]
            visited = set()
            max_dist = 0
            farthest_node = start
            
            while stack:
                node, dist = stack.pop()
                
                if node in visited:
                    continue
                
                visited.add(node)
                
                if dist > max_dist:
                    max_dist = dist
                    farthest_node = node
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        stack.append((neighbor, dist + 1))
            
            return farthest_node, max_dist
        
        start_node = next(iter(graph.keys()))
        one_end, _ = DFS_Iterative_Farthest(start_node)
        other_end, diameter = DFS_Iterative_Farthest(one_end)
        
        return diameter

def Test_Tree_Diameter():
    solution = Solution()
    
    test_cases = [
        ([[0,1],[0,2]], 2),
        ([[0,1],[1,2],[2,3],[1,4],[4,5]], 4),
        ([[0,1]], 1),
        ([[0,1],[0,2],[0,3]], 2),
        ([[0,1],[2,3],[1,2]], 3),
        ([[0,1],[1,2],[2,3],[3,4],[4,5]], 5)
    ]
    
    methods = [
        ("Two DFS", solution.Tree_Diameter_Two_DFS),
        ("DP Recursive", solution.Tree_Diameter_DP_Recursive),
        ("BFS Optimal", solution.Tree_Diameter_BFS_Optimal),
        ("Center Based", solution.Tree_Diameter_Center_Based),
        ("Memoized", solution.Tree_Diameter_Memoized),
        ("Iterative DFS", solution.Tree_Diameter_Iterative_DFS)
    ]
    
    for edges, expected in test_cases:
        print(f"Edges: {edges}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            try:
                result = method([edge[:] for edge in edges])
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        diameter, path = solution.Tree_Diameter_With_Path([edge[:] for edge in edges])
        print(f"With Path: Diameter={diameter}, Path={path}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Tree_Diameter()
