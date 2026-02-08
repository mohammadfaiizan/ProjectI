"""
Problem: Create and Print a Graph
URL: https://www.geeksforgeeks.org/graph-and-its-representations/

Problem Statement:
Create a graph using adjacency matrix and adjacency list representations and print them.

Sample Input/Output:
Input: 5 vertices, edges: (0,1), (0,4), (1,2), (1,3), (1,4), (2,3), (3,4)
Output: Adjacency Matrix and Adjacency List representations
"""


class Solution:
    def Create_Graph_Adjacency_Matrix(self, V, edges):
        """
        Adjacency Matrix Representation
        Time Complexity: O(V^2)
        Space Complexity: O(V^2)
        """
        adjMatrix = [[0] * V for _ in range(V)]
        for u, v in edges:
            adjMatrix[u][v] = 1
            adjMatrix[v][u] = 1
        return adjMatrix

    def Create_Graph_Adjacency_List(self, V, edges):
        """
        Adjacency List Representation
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        """
        adjList = [[] for _ in range(V)]
        for u, v in edges:
            adjList[u].append(v)
            adjList[v].append(u)
        return adjList


def Test_Create_Print_Graph():
    solution = Solution()
    V = 5
    edges = [(0, 1), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (3, 4)]
    
    print("Adjacency Matrix:")
    adjMatrix = solution.Create_Graph_Adjacency_Matrix(V, edges)
    for i in range(V):
        for j in range(V):
            print(adjMatrix[i][j], end=" ")
        print()
    
    print("\nAdjacency List:")
    adjList = solution.Create_Graph_Adjacency_List(V, edges)
    for i in range(V):
        print(f"{i}: ", end="")
        for neighbor in adjList[i]:
            print(neighbor, end=" ")
        print()


if __name__ == "__main__":
    Test_Create_Print_Graph()
