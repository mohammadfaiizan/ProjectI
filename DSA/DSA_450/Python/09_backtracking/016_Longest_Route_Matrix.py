"""
Problem: Longest Route in Matrix
URL: https://www.geeksforgeeks.org/longest-possible-route-in-a-matrix-with-hurdles/

Problem Statement:
Find the longest path in a matrix from source to destination with hurdles (0 = blocked). Can move in 4 directions.

Sample Input/Output:
Input: Matrix = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
       Source = (0, 0), Destination = (1, 7)
Output: 24
Explanation: Longest path length is 24
"""


class Solution:
    def Longest_Route_Matrix_DFS(self, matrix, source, destination):
        """
        DFS backtracking
        Time Complexity: O(4^(R*C))
        Space Complexity: O(R*C)
        """
        R = len(matrix)
        C = len(matrix[0])
        visited = [[0] * C for _ in range(R)]
        max_path = -1
        
        def dfs(x, y, length):
            nonlocal max_path
            
            if x == destination[0] and y == destination[1]:
                max_path = max(max_path, length)
                return
            
            dx = [-1, 1, 0, 0]
            dy = [0, 0, -1, 1]
            
            for k in range(4):
                nx = x + dx[k]
                ny = y + dy[k]
                if (0 <= nx < R and 0 <= ny < C and matrix[nx][ny] == 1 and visited[nx][ny] == 0):
                    visited[nx][ny] = 1
                    dfs(nx, ny, length + 1)
                    visited[nx][ny] = 0
        
        if matrix[source[0]][source[1]] == 1:
            visited[source[0]][source[1]] = 1
            dfs(source[0], source[1], 0)
        
        return max_path


def Test_Longest_Route_Matrix():
    solution = Solution()
    matrix = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
    source = (0, 0)
    destination = (1, 7)
    result = solution.Longest_Route_Matrix_DFS(matrix, source, destination)
    print("Longest path length:", result)


if __name__ == "__main__":
    Test_Longest_Route_Matrix()
