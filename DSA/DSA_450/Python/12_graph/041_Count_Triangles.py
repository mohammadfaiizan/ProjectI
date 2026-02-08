"""
Problem: Number of Triangles in a Graph
URL: https://www.geeksforgeeks.org/number-of-triangles-in-directed-and-undirected-graphs/

Problem Statement:
Count the number of triangles in directed and undirected graphs.

Sample Input/Output:
Input: Graph with edges
Output: Number of triangles
"""


class Solution:
    def Count_Triangles_Brute(self, V, edges, directed):
        """
        Check all triplets (i,j,k)
        Time Complexity: O(V^3)
        Space Complexity: O(V^2)
        """
        adj = [[False] * V for _ in range(V)]
        
        for edge in edges:
            u, v = edge[0], edge[1]
            adj[u][v] = True
            if not directed:
                adj[v][u] = True
        
        count = 0
        for i in range(V):
            for j in range(V):
                if i == j or not adj[i][j]:
                    continue
                for k in range(V):
                    if i == k or j == k:
                        continue
                    if adj[j][k] and adj[k][i]:
                        count += 1
        
        if directed:
            return count // 3
        else:
            return count // 6

    def Count_Triangles_Matrix(self, V, edges, directed):
        """
        Matrix multiplication trace method
        Time Complexity: O(V^3)
        Space Complexity: O(V^2)
        """
        adj = [[0] * V for _ in range(V)]
        
        for edge in edges:
            u, v = edge[0], edge[1]
            adj[u][v] = 1
            if not directed:
                adj[v][u] = 1
        
        adj2 = [[0] * V for _ in range(V)]
        for i in range(V):
            for j in range(V):
                for k in range(V):
                    adj2[i][j] += adj[i][k] * adj[k][j]
        
        adj3 = [[0] * V for _ in range(V)]
        for i in range(V):
            for j in range(V):
                for k in range(V):
                    adj3[i][j] += adj2[i][k] * adj[k][j]
        
        trace = 0
        for i in range(V):
            trace += adj3[i][i]
        
        if directed:
            return trace // 3
        else:
            return trace // 6


def Test_Count_Triangles():
    solution = Solution()
    
    print("Test Case 1: Undirected Graph")
    V1 = 4
    edges1 = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    result1_brute = solution.Count_Triangles_Brute(V1, edges1, False)
    result1_matrix = solution.Count_Triangles_Matrix(V1, edges1, False)
    print(f"Brute Force: {result1_brute} triangles")
    print(f"Matrix Method: {result1_matrix} triangles")
    print()
    
    print("Test Case 2: Directed Graph")
    V2 = 4
    edges2 = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 1)]
    result2_brute = solution.Count_Triangles_Brute(V2, edges2, True)
    result2_matrix = solution.Count_Triangles_Matrix(V2, edges2, True)
    print(f"Brute Force: {result2_brute} triangles")
    print(f"Matrix Method: {result2_matrix} triangles")
    print()
    
    print("Test Case 3: Complete Graph K4")
    V3 = 4
    edges3 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    result3_brute = solution.Count_Triangles_Brute(V3, edges3, False)
    result3_matrix = solution.Count_Triangles_Matrix(V3, edges3, False)
    print(f"Brute Force: {result3_brute} triangles")
    print(f"Matrix Method: {result3_matrix} triangles")


if __name__ == "__main__":
    Test_Count_Triangles()
