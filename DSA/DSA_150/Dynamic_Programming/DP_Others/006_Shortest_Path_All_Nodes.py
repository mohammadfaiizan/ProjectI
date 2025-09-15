"""
Problem: Shortest Path Visiting All Nodes
URL: https://leetcode.com/problems/shortest-path-visiting-all-nodes/

Problem Statement:
You have an undirected, connected graph of n nodes labeled from 0 to n - 1. 
You are given an array graph where graph[i] is a list of all the nodes connected to node i with an edge.
Return the length of the shortest path that visits every node. You may start and end at any node.

Sample Input/Output:
Input: graph = [[1,2,3],[0],[0],[0]]
Output: 4
Explanation: One possible path is [1,0,2,0,3]

Input: graph = [[1],[0,2,4],[1,3,4],[2],[1,2]]
Output: 4
Explanation: One possible path is [0,1,4,2,3]
"""

from typing import List, Tuple
from collections import deque
import heapq

class Solution:
    def Shortest_Path_Length_Brute_Force(self, graph: List[List[int]]) -> int:
        """
        Brute Force - Try all possible paths using DFS
        Time Complexity: O(n!)
        Space Complexity: O(n)
        """
        n = len(graph)
        all_visited = (1 << n) - 1
        min_path = float('inf')
        
        def DFS(node: int, visited_mask: int, path_length: int) -> None:
            nonlocal min_path
            
            if visited_mask == all_visited:
                min_path = min(min_path, path_length)
                return
            
            if path_length >= min_path:
                return
            
            for neighbor in graph[node]:
                new_visited = visited_mask | (1 << neighbor)
                DFS(neighbor, new_visited, path_length + 1)
        
        for start in range(n):
            DFS(start, 1 << start, 0)
        
        return min_path
    
    def Shortest_Path_Length_BFS_Bitmask_Optimal(self, graph: List[List[int]]) -> int:
        """
        BFS Bitmask Optimal - Use BFS with bitmask for visited states
        Time Complexity: O(n² * 2^n)
        Space Complexity: O(n * 2^n)
        """
        n = len(graph)
        all_visited = (1 << n) - 1
        
        queue = deque()
        visited = set()
        
        for i in range(n):
            queue.append((i, 1 << i, 0))
            visited.add((i, 1 << i))
        
        while queue:
            node, visited_mask, path_length = queue.popleft()
            
            if visited_mask == all_visited:
                return path_length
            
            for neighbor in graph[node]:
                new_visited = visited_mask | (1 << neighbor)
                
                if (neighbor, new_visited) not in visited:
                    visited.add((neighbor, new_visited))
                    queue.append((neighbor, new_visited, path_length + 1))
        
        return -1
    
    def Shortest_Path_Length_DP_Bitmask(self, graph: List[List[int]]) -> int:
        """
        DP Bitmask - Dynamic programming with bitmask
        Time Complexity: O(n² * 2^n)
        Space Complexity: O(n * 2^n)
        """
        n = len(graph)
        all_visited = (1 << n) - 1
        
        dp = {}
        
        def Min_Path_DP(node: int, visited_mask: int) -> int:
            if visited_mask == all_visited:
                return 0
            
            if (node, visited_mask) in dp:
                return dp[(node, visited_mask)]
            
            min_cost = float('inf')
            
            for neighbor in graph[node]:
                new_visited = visited_mask | (1 << neighbor)
                cost = 1 + Min_Path_DP(neighbor, new_visited)
                min_cost = min(min_cost, cost)
            
            dp[(node, visited_mask)] = min_cost
            return min_cost
        
        result = float('inf')
        
        for start in range(n):
            result = min(result, Min_Path_DP(start, 1 << start))
        
        return result
    
    def Shortest_Path_Length_Dijkstra(self, graph: List[List[int]]) -> int:
        """
        Dijkstra - Use Dijkstra's algorithm with state space
        Time Complexity: O(n² * 2^n * log(n * 2^n))
        Space Complexity: O(n * 2^n)
        """
        n = len(graph)
        all_visited = (1 << n) - 1
        
        heap = []
        distances = {}
        
        for i in range(n):
            heapq.heappush(heap, (0, i, 1 << i))
            distances[(i, 1 << i)] = 0
        
        while heap:
            dist, node, visited_mask = heapq.heappop(heap)
            
            if visited_mask == all_visited:
                return dist
            
            if (node, visited_mask) in distances and distances[(node, visited_mask)] < dist:
                continue
            
            for neighbor in graph[node]:
                new_visited = visited_mask | (1 << neighbor)
                new_dist = dist + 1
                
                if (neighbor, new_visited) not in distances or distances[(neighbor, new_visited)] > new_dist:
                    distances[(neighbor, new_visited)] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor, new_visited))
        
        return -1
    
    def Shortest_Path_Length_A_Star(self, graph: List[List[int]]) -> int:
        """
        A Star - Use A* algorithm with heuristic
        Time Complexity: O(n² * 2^n * log(n * 2^n))
        Space Complexity: O(n * 2^n)
        """
        n = len(graph)
        all_visited = (1 << n) - 1
        
        def Heuristic(visited_mask: int) -> int:
            unvisited_count = bin(visited_mask ^ all_visited).count('1')
            return unvisited_count
        
        heap = []
        distances = {}
        
        for i in range(n):
            initial_mask = 1 << i
            h = Heuristic(initial_mask)
            heapq.heappush(heap, (h, 0, i, initial_mask))
            distances[(i, initial_mask)] = 0
        
        while heap:
            _, dist, node, visited_mask = heapq.heappop(heap)
            
            if visited_mask == all_visited:
                return dist
            
            if (node, visited_mask) in distances and distances[(node, visited_mask)] < dist:
                continue
            
            for neighbor in graph[node]:
                new_visited = visited_mask | (1 << neighbor)
                new_dist = dist + 1
                
                if (neighbor, new_visited) not in distances or distances[(neighbor, new_visited)] > new_dist:
                    distances[(neighbor, new_visited)] = new_dist
                    h = Heuristic(new_visited)
                    heapq.heappush(heap, (new_dist + h, new_dist, neighbor, new_visited))
        
        return -1
    
    def Shortest_Path_Length_Bidirectional_BFS(self, graph: List[List[int]]) -> int:
        """
        Bidirectional BFS - BFS from all start states and end state
        Time Complexity: O(n² * 2^n)
        Space Complexity: O(n * 2^n)
        """
        n = len(graph)
        all_visited = (1 << n) - 1
        
        if n == 1:
            return 0
        
        forward_queue = deque()
        backward_queue = deque()
        forward_visited = {}
        backward_visited = {}
        
        for i in range(n):
            forward_queue.append((i, 1 << i))
            forward_visited[(i, 1 << i)] = 0
        
        backward_queue.append((0, all_visited))
        backward_visited[(0, all_visited)] = 0
        
        forward_dist = 0
        backward_dist = 0
        
        while forward_queue or backward_queue:
            if forward_queue:
                for _ in range(len(forward_queue)):
                    node, visited_mask = forward_queue.popleft()
                    
                    if (node, visited_mask) in backward_visited:
                        return forward_dist + backward_visited[(node, visited_mask)]
                    
                    for neighbor in graph[node]:
                        new_visited = visited_mask | (1 << neighbor)
                        
                        if (neighbor, new_visited) not in forward_visited:
                            forward_visited[(neighbor, new_visited)] = forward_dist + 1
                            forward_queue.append((neighbor, new_visited))
                
                forward_dist += 1
            
            if backward_queue:
                for _ in range(len(backward_queue)):
                    node, visited_mask = backward_queue.popleft()
                    
                    if (node, visited_mask) in forward_visited:
                        return backward_dist + forward_visited[(node, visited_mask)]
                    
                    for neighbor in graph[node]:
                        if (neighbor, visited_mask) not in backward_visited:
                            backward_visited[(neighbor, visited_mask)] = backward_dist + 1
                            backward_queue.append((neighbor, visited_mask))
                
                backward_dist += 1
        
        return -1
    
    def Shortest_Path_Length_With_Path(self, graph: List[List[int]]) -> Tuple[int, List[int]]:
        """
        With Path - Return shortest length and actual path
        Time Complexity: O(n² * 2^n)
        Space Complexity: O(n * 2^n)
        """
        n = len(graph)
        all_visited = (1 << n) - 1
        
        queue = deque()
        visited = set()
        parent = {}
        
        for i in range(n):
            queue.append((i, 1 << i, 0))
            visited.add((i, 1 << i))
            parent[(i, 1 << i)] = None
        
        while queue:
            node, visited_mask, path_length = queue.popleft()
            
            if visited_mask == all_visited:
                path = []
                current = (node, visited_mask)
                
                while current and parent[current]:
                    path.append(current[0])
                    current = parent[current]
                
                if current:
                    path.append(current[0])
                
                return path_length, path[::-1]
            
            for neighbor in graph[node]:
                new_visited = visited_mask | (1 << neighbor)
                
                if (neighbor, new_visited) not in visited:
                    visited.add((neighbor, new_visited))
                    parent[(neighbor, new_visited)] = (node, visited_mask)
                    queue.append((neighbor, new_visited, path_length + 1))
        
        return -1, []

def Test_Shortest_Path_Length():
    solution = Solution()
    
    test_cases = [
        ([[1,2,3],[0],[0],[0]], 4),
        ([[1],[0,2,4],[1,3,4],[2],[1,2]], 4),
        ([[1],[0]], 1),
        ([[]], 0),
        ([[1,2],[0,3],[0,3],[1,2]], 3)
    ]
    
    methods = [
        ("BFS Bitmask Optimal", solution.Shortest_Path_Length_BFS_Bitmask_Optimal),
        ("DP Bitmask", solution.Shortest_Path_Length_DP_Bitmask),
        ("Dijkstra", solution.Shortest_Path_Length_Dijkstra),
        ("A Star", solution.Shortest_Path_Length_A_Star)
    ]
    
    for graph, expected in test_cases:
        print(f"Graph: {graph}")
        print(f"Expected: {expected}")
        
        if len(graph) <= 4:
            try:
                result_bf = solution.Shortest_Path_Length_Brute_Force([neighbors[:] for neighbors in graph])
                print(f"Brute Force: {result_bf}")
            except Exception as e:
                print(f"Brute Force: Error - {e}")
        
        for method_name, method in methods:
            try:
                result = method([neighbors[:] for neighbors in graph])
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        if len(graph) <= 5:
            length, path = solution.Shortest_Path_Length_With_Path([neighbors[:] for neighbors in graph])
            print(f"With Path: Length={length}, Path={path}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Shortest_Path_Length()
