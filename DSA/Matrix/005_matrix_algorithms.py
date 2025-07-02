"""
Matrix Algorithms
=================

Topics: Path finding, DP on grids, matrix problems
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Medium to Hard
"""

from typing import List
from collections import deque

class MatrixAlgorithms:
    
    # ==========================================
    # 1. PATH FINDING ALGORITHMS
    # ==========================================
    
    def unique_paths(self, m: int, n: int) -> int:
        """LC 62: Unique Paths"""
        dp = [[1] * n for _ in range(m)]
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        
        return dp[m-1][n-1]
    
    def unique_paths_with_obstacles(self, obstacleGrid: List[List[int]]) -> int:
        """LC 63: Unique Paths II"""
        if not obstacleGrid or obstacleGrid[0][0] == 1:
            return 0
        
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = 1
        
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    dp[i][j] = 0
                else:
                    if i > 0:
                        dp[i][j] += dp[i-1][j]
                    if j > 0:
                        dp[i][j] += dp[i][j-1]
        
        return dp[m-1][n-1]
    
    def minimum_path_sum(self, grid: List[List[int]]) -> int:
        """LC 64: Minimum Path Sum"""
        m, n = len(grid), len(grid[0])
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                elif i == 0:
                    grid[i][j] += grid[i][j-1]
                elif j == 0:
                    grid[i][j] += grid[i-1][j]
                else:
                    grid[i][j] += min(grid[i-1][j], grid[i][j-1])
        
        return grid[m-1][n-1]
    
    # ==========================================
    # 2. BFS/DFS ON GRIDS
    # ==========================================
    
    def num_islands(self, grid: List[List[str]]) -> int:
        """LC 200: Number of Islands"""
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        count = 0
        
        def dfs(i, j):
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == '0':
                return
            
            grid[i][j] = '0'
            dfs(i+1, j)
            dfs(i-1, j)
            dfs(i, j+1)
            dfs(i, j-1)
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    dfs(i, j)
                    count += 1
        
        return count
    
    def shortest_path_binary_matrix(self, grid: List[List[int]]) -> int:
        """LC 1091: Shortest Path in Binary Matrix"""
        n = len(grid)
        if grid[0][0] or grid[n-1][n-1]:
            return -1
        
        queue = deque([(0, 0, 1)])
        visited = {(0, 0)}
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        while queue:
            row, col, path_length = queue.popleft()
            
            if row == n-1 and col == n-1:
                return path_length
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < n and 0 <= new_col < n and 
                    grid[new_row][new_col] == 0 and (new_row, new_col) not in visited):
                    
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col, path_length + 1))
        
        return -1
    
    # ==========================================
    # 3. MATRIX DP PROBLEMS
    # ==========================================
    
    def maximal_square(self, matrix: List[List[str]]) -> int:
        """LC 221: Maximal Square"""
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        dp = [[0] * n for _ in range(m)]
        max_side = 0
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                    
                    max_side = max(max_side, dp[i][j])
        
        return max_side * max_side
    
    def max_rectangle(self, matrix: List[List[str]]) -> int:
        """LC 85: Maximal Rectangle"""
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        heights = [0] * n
        max_area = 0
        
        def largest_rectangle_area(heights):
            stack = []
            max_area = 0
            
            for i, h in enumerate(heights):
                while stack and heights[stack[-1]] > h:
                    height = heights[stack.pop()]
                    width = i if not stack else i - stack[-1] - 1
                    max_area = max(max_area, height * width)
                stack.append(i)
            
            while stack:
                height = heights[stack.pop()]
                width = len(heights) if not stack else len(heights) - stack[-1] - 1
                max_area = max(max_area, height * width)
            
            return max_area
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            max_area = max(max_area, largest_rectangle_area(heights))
        
        return max_area

# Test Examples
def run_examples():
    ma = MatrixAlgorithms()
    
    print("=== MATRIX ALGORITHMS ===\n")
    
    # Unique paths
    m, n = 3, 7
    paths = ma.unique_paths(m, n)
    print(f"Unique paths in {m}x{n} grid: {paths}")
    
    # Number of islands
    grid = [
        ["1","1","1","1","0"],
        ["1","1","0","1","0"],
        ["1","1","0","0","0"],
        ["0","0","0","0","0"]
    ]
    islands = ma.num_islands(grid)
    print(f"Number of islands: {islands}")

if __name__ == "__main__":
    run_examples() 