"""
Problem: Dijkstra's Algorithm
URL: https://www.geeksforgeeks.org/problems/implementing-dijkstra-set-1-adjacency-matrix/1

Problem Statement:
Given a weighted, undirected and connected graph of V vertices and an adjacency list adj 
where adj[i] is a list of lists containing two integers where the first integer of each list j 
denotes there is edge between i and j, second integers corresponds to the weight of that edge. 
You are given the source vertex S and You have to Find the shortest distance of all the vertex's 
from the source vertex S. You have to return a list of integers denoting shortest distance 
between each node and Source vertex S.

Sample Input/Output:
Input: V = 2, adj = [[[1, 9]], [[0, 9]]], S = 0
Output: [0, 9]
Explanation: The source vertex is 0. Hence, the shortest distance of node 0 is 0 and the shortest distance of node 1 is 9.

Input: V = 3, adj = [[[1, 1], [2, 6]], [[2, 3], [0, 1]], [[1, 3], [0, 6]]], S = 2
Output: [4, 3, 0]
Explanation: For nodes 2 to 0, we can follow the path 2-1-0. This has a distance of 1+3 = 4.
"""

from typing import List
import heapq
import sys

class Solution:
    def Dijkstra_Brute_Force(self, V: int, adj: List[List[List[int]]], S: int) -> List[int]:
        """
        Brute Force - Find minimum distance vertex each time
        Time Complexity: O(V²)
        Space Complexity: O(V)
        """
        dist = [float('inf')] * V
        visited = [False] * V
        dist[S] = 0
        
        for _ in range(V):
            min_dist = float('inf')
            min_vertex = -1
            
            for v in range(V):
                if not visited[v] and dist[v] < min_dist:
                    min_dist = dist[v]
                    min_vertex = v
            
            if min_vertex == -1:
                break
            
            visited[min_vertex] = True
            
            for neighbor, weight in adj[min_vertex]:
                if not visited[neighbor]:
                    new_dist = dist[min_vertex] + weight
                    if new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
        
        return [d if d != float('inf') else -1 for d in dist]
    
    def Dijkstra_Priority_Queue_Optimal(self, V: int, adj: List[List[List[int]]], S: int) -> List[int]:
        """
        Priority Queue Optimal - Use min heap for efficiency
        Time Complexity: O((V + E) log V)
        Space Complexity: O(V)
        """
        dist = [float('inf')] * V
        dist[S] = 0
        pq = [(0, S)]
        
        while pq:
            current_dist, u = heapq.heappop(pq)
            
            if current_dist > dist[u]:
                continue
            
            for neighbor, weight in adj[u]:
                new_dist = dist[u] + weight
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
        
        return [d if d != float('inf') else -1 for d in dist]
    
    def Dijkstra_Set_Based(self, V: int, adj: List[List[List[int]]], S: int) -> List[int]:
        """
        Set Based - Use set for efficient min extraction
        Time Complexity: O((V + E) log V)
        Space Complexity: O(V)
        """
        dist = [float('inf')] * V
        dist[S] = 0
        vertex_set = [(0, S)]
        
        while vertex_set:
            vertex_set.sort()
            current_dist, u = vertex_set.pop(0)
            
            for neighbor, weight in adj[u]:
                new_dist = dist[u] + weight
                if new_dist < dist[neighbor]:
                    if (dist[neighbor], neighbor) in vertex_set:
                        vertex_set.remove((dist[neighbor], neighbor))
                    dist[neighbor] = new_dist
                    vertex_set.append((new_dist, neighbor))
        
        return [d if d != float('inf') else -1 for d in dist]
    
    def Dijkstra_With_Path(self, V: int, adj: List[List[List[int]]], S: int) -> tuple:
        """
        Dijkstra With Path - Track shortest paths
        Time Complexity: O((V + E) log V)
        Space Complexity: O(V)
        """
        dist = [float('inf')] * V
        parent = [-1] * V
        dist[S] = 0
        pq = [(0, S)]
        
        while pq:
            current_dist, u = heapq.heappop(pq)
            
            if current_dist > dist[u]:
                continue
            
            for neighbor, weight in adj[u]:
                new_dist = dist[u] + weight
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    parent[neighbor] = u
                    heapq.heappush(pq, (new_dist, neighbor))
        
        def Get_Path(target: int) -> List[int]:
            if dist[target] == float('inf'):
                return []
            path = []
            current = target
            while current != -1:
                path.append(current)
                current = parent[current]
            return path[::-1]
        
        return [d if d != float('inf') else -1 for d in dist], Get_Path
    
    def Dijkstra_Matrix_Input(self, graph: List[List[int]], S: int) -> List[int]:
        """
        Matrix Input - Handle adjacency matrix input
        Time Complexity: O(V²)
        Space Complexity: O(V)
        """
        V = len(graph)
        dist = [float('inf')] * V
        visited = [False] * V
        dist[S] = 0
        
        for _ in range(V):
            min_dist = float('inf')
            min_vertex = -1
            
            for v in range(V):
                if not visited[v] and dist[v] < min_dist:
                    min_dist = dist[v]
                    min_vertex = v
            
            if min_vertex == -1:
                break
            
            visited[min_vertex] = True
            
            for v in range(V):
                if (not visited[v] and graph[min_vertex][v] != 0 and 
                    dist[min_vertex] != float('inf') and 
                    dist[min_vertex] + graph[min_vertex][v] < dist[v]):
                    dist[v] = dist[min_vertex] + graph[min_vertex][v]
        
        return [d if d != float('inf') else -1 for d in dist]

def Test_Dijkstra():
    solution = Solution()
    
    test_cases = [
        (2, [[[1, 9]], [[0, 9]]], 0, [0, 9]),
        (3, [[[1, 1], [2, 6]], [[2, 3], [0, 1]], [[1, 3], [0, 6]]], 2, [4, 3, 0]),
        (4, [[[1, 5], [2, 3]], [[3, 7], [2, 2]], [[3, 2]], [[1, 1]]], 0, [0, 5, 3, 5])
    ]
    
    matrix_test = [
        [0, 4, 0, 0, 0, 0, 0, 8, 0],
        [4, 0, 8, 0, 0, 0, 0, 11, 0],
        [0, 8, 0, 7, 0, 4, 0, 0, 2],
        [0, 0, 7, 0, 9, 14, 0, 0, 0],
        [0, 0, 0, 9, 0, 10, 0, 0, 0],
        [0, 0, 4, 14, 10, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 1, 6],
        [8, 11, 0, 0, 0, 0, 1, 0, 7],
        [0, 0, 2, 0, 0, 0, 6, 7, 0]
    ]
    
    for V, adj, S, expected in test_cases:
        result1 = solution.Dijkstra_Brute_Force(V, [row.copy() for row in adj], S)
        result2 = solution.Dijkstra_Priority_Queue_Optimal(V, [row.copy() for row in adj], S)
        result3 = solution.Dijkstra_Set_Based(V, [row.copy() for row in adj], S)
        result4, path_func = solution.Dijkstra_With_Path(V, [row.copy() for row in adj], S)
        
        print(f"V: {V}, Adjacency List: {adj}, Source: {S}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Priority Queue: {result2}")
        print(f"Set Based: {result3}")
        print(f"With Path: {result4}")
        if V <= 4:
            for i in range(V):
                if i != S:
                    print(f"Path to {i}: {path_func(i)}")
        print("-" * 50)
    
    print("Matrix Input Test:")
    result_matrix = solution.Dijkstra_Matrix_Input(matrix_test, 0)
    print(f"Matrix Dijkstra from 0: {result_matrix}")

if __name__ == "__main__":
    Test_Dijkstra()
