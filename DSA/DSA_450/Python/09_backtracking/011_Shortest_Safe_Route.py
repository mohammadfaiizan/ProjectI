"""
Problem: Shortest Safe Route
URL: https://www.geeksforgeeks.org/find-shortest-safe-route-in-a-path-with-landmines/

Problem Statement:
Given a matrix with landmines (marked as 0), find the shortest safe route from any cell in the first column to any cell in the last column. Adjacent cells to landmines are also unsafe.

Sample Input/Output:
Input: 
Matrix = [[1, 1, 1, 1, 1],
          [1, 0, 1, 1, 1],
          [1, 1, 1, 1, 1],
          [1, 1, 1, 0, 1],
          [1, 0, 1, 1, 1]]
Output: 6
Explanation: Shortest path length is 6 from (0,0) to (4,4)
"""

from collections import deque


class Solution:
    def Shortest_Safe_Route_BFS(self, matrix):
        """
        BFS after marking unsafe cells
        Time Complexity: O(R*C)
        Space Complexity: O(R*C)
        """
        R = len(matrix)
        C = len(matrix[0])
        safe = [[1] * C for _ in range(R)]
        
        for i in range(R):
            for j in range(C):
                if matrix[i][j] == 0:
                    safe[i][j] = 0
                    dx = [-1, 1, 0, 0]
                    dy = [0, 0, -1, 1]
                    for k in range(4):
                        ni = i + dx[k]
                        nj = j + dy[k]
                        if 0 <= ni < R and 0 <= nj < C:
                            safe[ni][nj] = 0
        
        q = deque()
        dist = [[-1] * C for _ in range(R)]
        
        for i in range(R):
            if safe[i][0] == 1:
                q.append((i, 0))
                dist[i][0] = 1
        
        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]
        
        while q:
            x, y = q.popleft()
            
            if y == C - 1:
                return dist[x][y]
            
            for k in range(4):
                nx = x + dx[k]
                ny = y + dy[k]
                if (0 <= nx < R and 0 <= ny < C and safe[nx][ny] == 1 and dist[nx][ny] == -1):
                    dist[nx][ny] = dist[x][y] + 1
                    q.append((nx, ny))
        
        return -1
    
    def Shortest_Safe_Route_Backtracking(self, matrix):
        """
        Backtracking DFS approach
        Time Complexity: O(4^(R*C))
        Space Complexity: O(R*C)
        """
        R = len(matrix)
        C = len(matrix[0])
        safe = [[1] * C for _ in range(R)]
        
        for i in range(R):
            for j in range(C):
                if matrix[i][j] == 0:
                    safe[i][j] = 0
                    dx = [-1, 1, 0, 0]
                    dy = [0, 0, -1, 1]
                    for k in range(4):
                        ni = i + dx[k]
                        nj = j + dy[k]
                        if 0 <= ni < R and 0 <= nj < C:
                            safe[ni][nj] = 0
        
        min_path = float('inf')
        visited = [[0] * C for _ in range(R)]
        
        def dfs(x, y, length):
            nonlocal min_path
            
            if y == C - 1:
                min_path = min(min_path, length)
                return
            
            dx = [-1, 1, 0, 0]
            dy = [0, 0, -1, 1]
            
            for k in range(4):
                nx = x + dx[k]
                ny = y + dy[k]
                if (0 <= nx < R and 0 <= ny < C and safe[nx][ny] == 1 and visited[nx][ny] == 0):
                    visited[nx][ny] = 1
                    dfs(nx, ny, length + 1)
                    visited[nx][ny] = 0
        
        for i in range(R):
            if safe[i][0] == 1:
                visited[i][0] = 1
                dfs(i, 0, 1)
                visited[i][0] = 0
        
        return min_path if min_path != float('inf') else -1


def Test_Shortest_Safe_Route():
    solution = Solution()
    matrix = [
        [1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1]
    ]
    print("BFS Approach:", solution.Shortest_Safe_Route_BFS(matrix))
    print("Backtracking Approach:", solution.Shortest_Safe_Route_Backtracking(matrix))


if __name__ == "__main__":
    Test_Shortest_Safe_Route()
