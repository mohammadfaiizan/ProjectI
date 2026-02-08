"""
Problem: Minimum Time Required to Rot All Oranges
URL: https://practice.geeksforgeeks.org/problems/rotten-oranges2536/1

Problem Statement:
Given a grid of dimension nxm where each cell in the grid can have 3 values:
0: Empty cell
1: Cells have fresh oranges
2: Cells have rotten oranges
Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.
Return the minimum time in minutes until no cell has a fresh orange. If this is impossible, return -1.

Sample Input/Output:
Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4
"""

from collections import deque


class Solution:
    def Rotting_Oranges_BFS(self, grid):
        """
        Rot oranges using BFS level-by-level.
        Time Complexity: O(R*C)
        Space Complexity: O(R*C)
        """
        rows = len(grid)
        cols = len(grid[0])
        q = deque()
        fresh = 0
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 2:
                    q.append((i, j))
                elif grid[i][j] == 1:
                    fresh += 1
        
        if fresh == 0:
            return 0
        
        time = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while q:
            size = len(q)
            rotted = False
            
            for i in range(size):
                x, y = q.popleft()
                
                for dx, dy in directions:
                    nx = x + dx
                    ny = y + dy
                    
                    if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 1:
                        grid[nx][ny] = 2
                        q.append((nx, ny))
                        fresh -= 1
                        rotted = True
            
            if rotted:
                time += 1
        
        return time if fresh == 0 else -1
    
    def Rotting_Oranges_Brute_Force(self, grid):
        """
        Rot oranges using brute force simulation.
        Time Complexity: O(R*C * R*C)
        Space Complexity: O(R*C)
        """
        rows = len(grid)
        cols = len(grid[0])
        time = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while True:
            next_grid = [row[:] for row in grid]
            changed = False
            
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] == 2:
                        for dx, dy in directions:
                            nx = i + dx
                            ny = j + dy
                            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 1:
                                next_grid[nx][ny] = 2
                                changed = True
            
            if not changed:
                break
            grid = next_grid
            time += 1
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    return -1
        
        return time


def Test_Rotting_Oranges():
    solution = Solution()
    
    grid1 = [[2,1,1],[1,1,0],[0,1,1]]
    grid1_copy = [row[:] for row in grid1]
    print(f"Test 1 - BFS: {solution.Rotting_Oranges_BFS(grid1)}")
    print(f"Test 1 - Brute Force: {solution.Rotting_Oranges_Brute_Force(grid1_copy)}")
    
    grid2 = [[2,1,1],[0,1,1],[1,0,1]]
    grid2_copy = [row[:] for row in grid2]
    print(f"Test 2 - BFS: {solution.Rotting_Oranges_BFS(grid2)}")
    print(f"Test 2 - Brute Force: {solution.Rotting_Oranges_Brute_Force(grid2_copy)}")
    
    grid3 = [[0,2]]
    grid3_copy = [row[:] for row in grid3]
    print(f"Test 3 - BFS: {solution.Rotting_Oranges_BFS(grid3)}")
    print(f"Test 3 - Brute Force: {solution.Rotting_Oranges_Brute_Force(grid3_copy)}")


if __name__ == "__main__":
    Test_Rotting_Oranges()
