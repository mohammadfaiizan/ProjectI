"""
Problem: Shortest Path in Undirected Graph with Unit Weights
URL: https://www.geeksforgeeks.org/shortest-path-unweighted-graph/

Problem Statement:
Find shortest distance from source to all vertices in an unweighted graph.

Sample Input/Output:
Input: Unweighted graph with source vertex
Output: Shortest distances from source to all vertices
"""

from collections import deque


class Solution:
    def Shortest_Path_BFS(self, V, edges, src):
        """
        Simple BFS
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        """
        adj = [[] for _ in range(V)]
        for edge in edges:
            u, v = edge[0], edge[1]
            adj[u].append(v)
            adj[v].append(u)
        
        dist = [-1] * V
        q = deque()
        
        dist[src] = 0
        q.append(src)
        
        while q:
            u = q.popleft()
            
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        
        return dist


def Test_Shortest_Path_Unit_Weight():
    solution = Solution()
    
    print("Test Case 1:")
    V1 = 6
    edges1 = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)]
    src1 = 0
    dist1 = solution.Shortest_Path_BFS(V1, edges1, src1)
    print(f"Source: {src1}")
    print("Distances:", end=" ")
    for i in range(V1):
        print(f"[{i}:{dist1[i]}]", end=" ")
    print()
    print()
    
    print("Test Case 2:")
    V2 = 5
    edges2 = [(0, 1), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (3, 4)]
    src2 = 0
    dist2 = solution.Shortest_Path_BFS(V2, edges2, src2)
    print(f"Source: {src2}")
    print("Distances:", end=" ")
    for i in range(V2):
        print(f"[{i}:{dist2[i]}]", end=" ")
    print()
    print()
    
    print("Test Case 3:")
    V3 = 4
    edges3 = [(0, 1), (1, 2), (2, 3)]
    src3 = 0
    dist3 = solution.Shortest_Path_BFS(V3, edges3, src3)
    print(f"Source: {src3}")
    print("Distances:", end=" ")
    for i in range(V3):
        print(f"[{i}:{dist3[i]}]", end=" ")
    print()


if __name__ == "__main__":
    Test_Shortest_Path_Unit_Weight()
