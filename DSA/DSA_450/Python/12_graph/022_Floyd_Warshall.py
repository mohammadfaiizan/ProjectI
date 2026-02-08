"""
Problem: Floyd-Warshall Algorithm
URL: https://practice.geeksforgeeks.org/problems/implementing-floyd-warshall2042/1

Problem Statement:
Find all-pairs shortest paths in a weighted directed graph. The algorithm uses dynamic programming with intermediate vertices to compute shortest distances between all pairs of vertices.

Sample Input/Output:
Input: Graph with 4 vertices, adjacency matrix with weights
Output: Shortest distance matrix for all pairs
"""


class Solution:
    def Floyd_Warshall_DP(self, V, graph):
        """
        3 nested loops with intermediate vertex
        Time Complexity: O(V^3)
        Space Complexity: O(V^2)
        """
        dist = [row[:] for row in graph]
        
        for i in range(V):
            for j in range(V):
                if dist[i][j] == -1:
                    dist[i][j] = float('inf')
                if i == j:
                    dist[i][j] = 0
        
        for k in range(V):
            for i in range(V):
                for j in range(V):
                    if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        
        for i in range(V):
            for j in range(V):
                if dist[i][j] == float('inf'):
                    dist[i][j] = -1
        
        return dist


def Test_Floyd_Warshall():
    solution = Solution()
    
    print("Test Case 1: Weighted graph with 4 vertices")
    V1 = 4
    graph1 = [
        [0, 5, -1, 10],
        [-1, 0, 3, -1],
        [-1, -1, 0, 1],
        [-1, -1, -1, 0]
    ]
    result1 = solution.Floyd_Warshall_DP(V1, graph1)
    print("Shortest distance matrix:")
    for i in range(V1):
        for j in range(V1):
            print(result1[i][j], end=" ")
        print()
    
    print("\nTest Case 2: Complete graph")
    V2 = 3
    graph2 = [
        [0, 1, 4],
        [1, 0, 2],
        [4, 2, 0]
    ]
    result2 = solution.Floyd_Warshall_DP(V2, graph2)
    print("Shortest distance matrix:")
    for i in range(V2):
        for j in range(V2):
            print(result2[i][j], end=" ")
        print()
    
    print("\nTest Case 3: Graph with no direct paths")
    V3 = 4
    graph3 = [
        [0, -1, -1, -1],
        [-1, 0, 2, -1],
        [-1, -1, 0, 3],
        [-1, -1, -1, 0]
    ]
    result3 = solution.Floyd_Warshall_DP(V3, graph3)
    print("Shortest distance matrix:")
    for i in range(V3):
        for j in range(V3):
            print(result3[i][j], end=" ")
        print()


if __name__ == "__main__":
    Test_Floyd_Warshall()
