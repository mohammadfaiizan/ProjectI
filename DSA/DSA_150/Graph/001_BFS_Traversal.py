"""
Problem: BFS Traversal of Graph
URL: https://www.geeksforgeeks.org/problems/bfs-traversal-of-graph/1

Problem Statement:
Given a connected undirected graph represented by an adjacency list adj, 
where adj[i] contains all the nodes connected to node i. 
Perform a Breadth-First Search (BFS) traversal of the graph starting from node 0, 
and return an array containing the BFS traversal of the graph.

Sample Input/Output:
Input: adj = [[2,3,1], [0], [0,4], [0], [2]]
Output: [0, 2, 3, 1, 4]
Explanation: Starting from 0, we first visit the neighbors of 0: 2, 3, 1. Then we visit neighbors of 2: 4.

Input: adj = [[1], [0,2,3], [1], [1]]
Output: [0, 1, 2, 3]
Explanation: Starting from 0, we visit 1, then 2 and 3.
"""

from typing import List
from collections import deque

class Solution:
    def BFS_Queue_Based(self, V: int, adj: List[List[int]]) -> List[int]:
        """
        Queue Based BFS - Standard BFS using queue
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        visited = [False] * V
        result = []
        queue = deque([0])
        visited[0] = True
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        return result
    
    def BFS_Level_Order(self, V: int, adj: List[List[int]]) -> List[int]:
        """
        Level Order BFS - Process nodes level by level
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        visited = [False] * V
        result = []
        queue = deque([0])
        visited[0] = True
        
        while queue:
            level_size = len(queue)
            for _ in range(level_size):
                node = queue.popleft()
                result.append(node)
                
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
        
        return result
    
    def BFS_List_Based_Queue(self, V: int, adj: List[List[int]]) -> List[int]:
        """
        List Based Queue - Use list as queue (less efficient)
        Time Complexity: O(V² + E)
        Space Complexity: O(V)
        """
        visited = [False] * V
        result = []
        queue = [0]
        visited[0] = True
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        return result
    
    def BFS_Multiple_Components(self, V: int, adj: List[List[int]]) -> List[int]:
        """
        Multiple Components BFS - Handle disconnected graph
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        visited = [False] * V
        result = []
        
        def BFS_Component(start: int) -> None:
            queue = deque([start])
            visited[start] = True
            
            while queue:
                node = queue.popleft()
                result.append(node)
                
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
        
        for i in range(V):
            if not visited[i]:
                BFS_Component(i)
        
        return result
    
    def BFS_With_Distance(self, V: int, adj: List[List[int]]) -> List[tuple]:
        """
        BFS With Distance - Track distance from source
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        visited = [False] * V
        result = []
        queue = deque([(0, 0)])
        visited[0] = True
        
        while queue:
            node, distance = queue.popleft()
            result.append((node, distance))
            
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, distance + 1))
        
        return result
    
    def BFS_Parent_Tracking(self, V: int, adj: List[List[int]]) -> tuple:
        """
        BFS Parent Tracking - Track parent for path reconstruction
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        visited = [False] * V
        parent = [-1] * V
        result = []
        queue = deque([0])
        visited[0] = True
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parent[neighbor] = node
                    queue.append(neighbor)
        
        return result, parent

def Test_BFS():
    solution = Solution()
    
    test_cases = [
        (5, [[2,3,1], [0], [0,4], [0], [2]], [0, 2, 3, 1, 4]),
        (4, [[1], [0,2,3], [1], [1]], [0, 1, 2, 3]),
        (3, [[1,2], [0,2], [0,1]], [0, 1, 2]),
        (1, [[]], [0])
    ]
    
    for V, adj, expected in test_cases:
        result1 = solution.BFS_Queue_Based(V, [row.copy() for row in adj])
        result2 = solution.BFS_Level_Order(V, [row.copy() for row in adj])
        result3 = solution.BFS_List_Based_Queue(V, [row.copy() for row in adj])
        result4 = solution.BFS_Multiple_Components(V, [row.copy() for row in adj])
        result5 = solution.BFS_With_Distance(V, [row.copy() for row in adj])
        result6, parent = solution.BFS_Parent_Tracking(V, [row.copy() for row in adj])
        
        print(f"V: {V}, Adjacency List: {adj}")
        print(f"Expected: {expected}")
        print(f"Queue Based: {result1}")
        print(f"Level Order: {result2}")
        print(f"List Based Queue: {result3}")
        print(f"Multiple Components: {result4}")
        print(f"With Distance: {result5}")
        print(f"Parent Tracking: {result6}, Parents: {parent}")
        print("-" * 50)

if __name__ == "__main__":
    Test_BFS()
