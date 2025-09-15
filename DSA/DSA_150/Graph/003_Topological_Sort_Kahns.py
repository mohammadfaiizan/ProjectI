"""
Problem: Topological Sort (Kahn's Algorithm)
URL: https://www.geeksforgeeks.org/problems/topological-sort/1

Problem Statement:
Given a Directed Acyclic Graph (DAG) with V vertices and E edges, 
Find any Topological Sorting of that Graph.

Sample Input/Output:
Input: V = 6, E = 6, edges = [[5,2],[5,0],[4,0],[4,1],[2,3],[3,1]]
Output: [4, 5, 0, 2, 3, 1]
Explanation: A topological sorting of the given graph is "4 5 0 2 3 1", 
there may be other valid topological sortings for the same graph.

Input: V = 4, E = 3, edges = [[0,1],[1,2],[2,3]]
Output: [0, 1, 2, 3]
Explanation: Linear dependency chain gives unique topological order.
"""

from typing import List
from collections import deque

class Solution:
    def Topological_Sort_DFS(self, V: int, adj: List[List[int]]) -> List[int]:
        """
        DFS Based - Use DFS with finishing time order
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        visited = [False] * V
        stack = []
        
        def DFS(v: int) -> None:
            visited[v] = True
            for neighbor in adj[v]:
                if not visited[neighbor]:
                    DFS(neighbor)
            stack.append(v)
        
        for i in range(V):
            if not visited[i]:
                DFS(i)
        
        return stack[::-1]
    
    def Topological_Sort_Kahns_Optimal(self, V: int, adj: List[List[int]]) -> List[int]:
        """
        Kahn's Algorithm Optimal - BFS based using in-degree
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        in_degree = [0] * V
        
        for u in range(V):
            for v in adj[u]:
                in_degree[v] += 1
        
        queue = deque()
        for i in range(V):
            if in_degree[i] == 0:
                queue.append(i)
        
        result = []
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result if len(result) == V else []
    
    def Topological_Sort_Priority_Queue(self, V: int, adj: List[List[int]]) -> List[int]:
        """
        Priority Queue - Use priority queue for lexicographically smallest
        Time Complexity: O(V log V + E)
        Space Complexity: O(V)
        """
        import heapq
        
        in_degree = [0] * V
        
        for u in range(V):
            for v in adj[u]:
                in_degree[v] += 1
        
        heap = []
        for i in range(V):
            if in_degree[i] == 0:
                heapq.heappush(heap, i)
        
        result = []
        while heap:
            node = heapq.heappop(heap)
            result.append(node)
            
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    heapq.heappush(heap, neighbor)
        
        return result if len(result) == V else []
    
    def Topological_Sort_Cycle_Detection(self, V: int, adj: List[List[int]]) -> tuple:
        """
        Cycle Detection - Detect cycles while finding topological order
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * V
        result = []
        has_cycle = False
        
        def DFS(v: int) -> bool:
            nonlocal has_cycle
            if color[v] == GRAY:
                has_cycle = True
                return True
            
            if color[v] == BLACK:
                return False
            
            color[v] = GRAY
            
            for neighbor in adj[v]:
                if DFS(neighbor):
                    return True
            
            color[v] = BLACK
            result.append(v)
            return False
        
        for i in range(V):
            if color[i] == WHITE:
                if DFS(i):
                    break
        
        return (result[::-1] if not has_cycle else [], has_cycle)
    
    def Topological_Sort_All_Orders(self, V: int, adj: List[List[int]]) -> List[List[int]]:
        """
        All Topological Orders - Find all possible topological orders
        Time Complexity: O(V! * V)
        Space Complexity: O(V)
        """
        in_degree = [0] * V
        
        for u in range(V):
            for v in adj[u]:
                in_degree[v] += 1
        
        all_orders = []
        
        def Backtrack(current_order: List[int], remaining_in_degree: List[int]) -> None:
            if len(current_order) == V:
                all_orders.append(current_order[:])
                return
            
            for i in range(V):
                if remaining_in_degree[i] == 0 and i not in current_order:
                    current_order.append(i)
                    
                    new_in_degree = remaining_in_degree[:]
                    new_in_degree[i] = -1
                    
                    for neighbor in adj[i]:
                        new_in_degree[neighbor] -= 1
                    
                    Backtrack(current_order, new_in_degree)
                    current_order.pop()
        
        Backtrack([], in_degree[:])
        return all_orders
    
    def Topological_Sort_Lexicographic(self, V: int, adj: List[List[int]]) -> List[int]:
        """
        Lexicographic Order - Always choose smallest available vertex
        Time Complexity: O(V² + E)
        Space Complexity: O(V)
        """
        in_degree = [0] * V
        
        for u in range(V):
            for v in adj[u]:
                in_degree[v] += 1
        
        result = []
        
        for _ in range(V):
            next_vertex = -1
            for i in range(V):
                if in_degree[i] == 0:
                    next_vertex = i
                    break
            
            if next_vertex == -1:
                return []
            
            result.append(next_vertex)
            in_degree[next_vertex] = -1
            
            for neighbor in adj[next_vertex]:
                in_degree[neighbor] -= 1
        
        return result

def Test_Topological_Sort():
    solution = Solution()
    
    test_cases = [
        (6, [[2, 3], [3], [1], [1], [0, 1], [0, 2]], [4, 5, 0, 2, 3, 1]),
        (4, [[1], [2], [3], []], [0, 1, 2, 3]),
        (3, [[1], [2], []], [0, 1, 2]),
        (2, [[1], []], [0, 1])
    ]
    
    for V, adj, expected in test_cases:
        result1 = solution.Topological_Sort_DFS(V, [row.copy() for row in adj])
        result2 = solution.Topological_Sort_Kahns_Optimal(V, [row.copy() for row in adj])
        result3 = solution.Topological_Sort_Priority_Queue(V, [row.copy() for row in adj])
        result4, has_cycle = solution.Topological_Sort_Cycle_Detection(V, [row.copy() for row in adj])
        result5 = solution.Topological_Sort_Lexicographic(V, [row.copy() for row in adj])
        
        print(f"V: {V}, Adjacency List: {adj}")
        print(f"Expected: {expected}")
        print(f"DFS Based: {result1}")
        print(f"Kahn's Optimal: {result2}")
        print(f"Priority Queue: {result3}")
        print(f"Cycle Detection: {result4}, Has Cycle: {has_cycle}")
        print(f"Lexicographic: {result5}")
        
        if V <= 4:
            all_orders = solution.Topological_Sort_All_Orders(V, [row.copy() for row in adj])
            print(f"All Orders: {all_orders}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Topological_Sort()
