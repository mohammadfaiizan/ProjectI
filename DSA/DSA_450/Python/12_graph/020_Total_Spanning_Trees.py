"""
Problem: Total Number of Spanning Trees in a Graph
URL: https://www.geeksforgeeks.org/total-number-spanning-trees-graph/

Problem Statement:
Count the total number of spanning trees in a connected undirected graph using Kirchhoff's Matrix Tree Theorem. The theorem states that the number of spanning trees equals any cofactor of the Laplacian matrix.

Sample Input/Output:
Input: Triangle graph (3 vertices, 3 edges)
Output: 3 spanning trees
Input: Complete graph K4 (4 vertices, 6 edges)
Output: 16 spanning trees
"""


class Solution:
    def Spanning_Trees_Kirchhoff(self, V, graph):
        """
        Construct Laplacian matrix, compute cofactor/determinant
        Time Complexity: O(V^3)
        Space Complexity: O(V^2)
        """
        laplacian = [[0] * V for _ in range(V)]
        
        for i in range(V):
            for j in range(V):
                if i != j and graph[i][j] > 0:
                    laplacian[i][j] = -graph[i][j]
                    laplacian[i][i] += graph[i][j]
        
        matrix = [[float(laplacian[i][j]) for j in range(1, V)] for i in range(1, V)]
        
        return round(self.Determinant(matrix))
    
    def Determinant(self, mat):
        n = len(mat)
        if n == 1:
            return mat[0][0]
        if n == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        
        det = 1.0
        for i in range(n):
            maxRow = i
            for k in range(i + 1, n):
                if abs(mat[k][i]) > abs(mat[maxRow][i]):
                    maxRow = k
            
            if maxRow != i:
                mat[i], mat[maxRow] = mat[maxRow], mat[i]
                det *= -1
            
            if abs(mat[i][i]) < 1e-9:
                return 0
            
            for k in range(i + 1, n):
                factor = mat[k][i] / mat[i][i]
                for j in range(i, n):
                    mat[k][j] -= factor * mat[i][j]
        
        for i in range(n):
            det *= mat[i][i]
        
        return det


def Test_Spanning_Trees():
    solution = Solution()
    
    print("Test Case 1: Triangle graph (3 vertices)")
    V1 = 3
    graph1 = [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]
    print(f"Number of spanning trees: {solution.Spanning_Trees_Kirchhoff(V1, graph1)}")
    
    print("\nTest Case 2: Complete graph K4 (4 vertices)")
    V2 = 4
    graph2 = [
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0]
    ]
    print(f"Number of spanning trees: {solution.Spanning_Trees_Kirchhoff(V2, graph2)}")
    
    print("\nTest Case 3: Simple path graph (4 vertices)")
    V3 = 4
    graph3 = [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ]
    print(f"Number of spanning trees: {solution.Spanning_Trees_Kirchhoff(V3, graph3)}")
    
    print("\nTest Case 4: Star graph (5 vertices)")
    V4 = 5
    graph4 = [
        [0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0]
    ]
    print(f"Number of spanning trees: {solution.Spanning_Trees_Kirchhoff(V4, graph4)}")


if __name__ == "__main__":
    Test_Spanning_Trees()
