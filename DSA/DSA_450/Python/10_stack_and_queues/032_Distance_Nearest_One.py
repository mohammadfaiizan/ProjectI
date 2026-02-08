"""
Problem: Distance of Nearest Cell Having 1 in Binary Matrix
URL: https://practice.geeksforgeeks.org/problems/distance-of-nearest-cell-having-1-1587115620/1

Problem Statement:
Given a binary matrix of size N x M. For each cell of the matrix, find the distance of the nearest cell having 1 in the matrix.
Distance between two cells (x1, y1) and (x2, y2) is defined as |x1 - x2| + |y1 - y2|.

Sample Input/Output:
Input: grid = [[0,0,0],[0,1,0],[0,0,0]]
Output: [[2,1,2],[1,0,1],[2,1,2]]
"""

from collections import deque


class Solution:
    def Distance_Nearest_One_BFS(self, grid):
        """
        Find distances using multi-source BFS from all 1s.
        Time Complexity: O(N*M)
        Space Complexity: O(N*M)
        """
        n = len(grid)
        m = len(grid[0])
        dist = [[-1] * m for _ in range(n)]
        q = deque()
        
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    dist[i][j] = 0
                    q.append((i, j))
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while q:
            x, y = q.popleft()
            
            for dx, dy in directions:
                nx = x + dx
                ny = y + dy
                
                if 0 <= nx < n and 0 <= ny < m and dist[nx][ny] == -1:
                    dist[nx][ny] = dist[x][y] + 1
                    q.append((nx, ny))
        
        return dist
    
    def Distance_Nearest_One_Brute_Force(self, grid):
        """
        Find distances using brute force.
        Time Complexity: O(N^2*M^2)
        Space Complexity: O(N*M)
        """
        n = len(grid)
        m = len(grid[0])
        dist = [[float('inf')] * m for _ in range(n)]
        
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    dist[i][j] = 0
                else:
                    for x in range(n):
                        for y in range(m):
                            if grid[x][y] == 1:
                                dist[i][j] = min(dist[i][j], abs(i - x) + abs(j - y))
        
        return dist


def Test_Distance_Nearest_One():
    solution = Solution()
    
    grid1 = [[0,0,0],[0,1,0],[0,0,0]]
    result1 = solution.Distance_Nearest_One_BFS(grid1)
    print("Test 1 - BFS Result:")
    for row in result1:
        print(row)
    
    grid2 = [[0,0,0],[0,1,0],[1,0,1]]
    result2 = solution.Distance_Nearest_One_BFS(grid2)
    print("\nTest 2 - BFS Result:")
    for row in result2:
        print(row)
    
    grid3 = [[1,0,1],[0,1,0],[1,0,1]]
    result3 = solution.Distance_Nearest_One_BFS(grid3)
    print("\nTest 3 - BFS Result:")
    for row in result3:
        print(row)


if __name__ == "__main__":
    Test_Distance_Nearest_One()
