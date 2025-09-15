"""
Problem: Unique Paths II
URL: https://leetcode.com/problems/unique-paths-ii/

Problem Statement:
You are given an m x n integer array grid. There is a robot initially located at the top-left corner (i.e., grid[0][0]). 
The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). 
The robot can only move either down or right at any point in time.
An obstacle and space are marked as 1 and 0 respectively in grid. A path that the robot takes cannot include any square that is an obstacle.
Return the number of possible unique paths that the robot can take to reach the bottom-right corner.

Sample Input/Output:
Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
Output: 2
Explanation: There is one obstacle in the middle of the 3x3 grid above.
There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right

Input: obstacleGrid = [[0,1],[0,0]]
Output: 1
"""

from typing import List

class Solution:
    def Unique_Paths_With_Obstacles_Recursive(self, obstacleGrid: List[List[int]]) -> int:
        """
        Recursive - Try both directions avoiding obstacles
        Time Complexity: O(2^(m+n))
        Space Complexity: O(m+n)
        """
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        
        def Count_Paths(row: int, col: int) -> int:
            if row >= m or col >= n or obstacleGrid[row][col] == 1:
                return 0
            
            if row == m - 1 and col == n - 1:
                return 1
            
            right_paths = Count_Paths(row, col + 1)
            down_paths = Count_Paths(row + 1, col)
            
            return right_paths + down_paths
        
        return Count_Paths(0, 0)
    
    def Unique_Paths_With_Obstacles_Memoized(self, obstacleGrid: List[List[int]]) -> int:
        """
        Memoized - Top-down DP with memoization
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        memo = {}
        
        def Count_Paths_Memo(row: int, col: int) -> int:
            if row >= m or col >= n or obstacleGrid[row][col] == 1:
                return 0
            
            if row == m - 1 and col == n - 1:
                return 1
            
            if (row, col) in memo:
                return memo[(row, col)]
            
            right_paths = Count_Paths_Memo(row, col + 1)
            down_paths = Count_Paths_Memo(row + 1, col)
            
            memo[(row, col)] = right_paths + down_paths
            return memo[(row, col)]
        
        return Count_Paths_Memo(0, 0)
    
    def Unique_Paths_With_Obstacles_Tabulation_Optimal(self, obstacleGrid: List[List[int]]) -> int:
        """
        Tabulation Optimal - Bottom-up DP with 2D table
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        
        if obstacleGrid[0][0] == 1 or obstacleGrid[m-1][n-1] == 1:
            return 0
        
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = 1
        
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] if obstacleGrid[i][0] == 0 else 0
        
        for j in range(1, n):
            dp[0][j] = dp[0][j-1] if obstacleGrid[0][j] == 0 else 0
        
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
                else:
                    dp[i][j] = 0
        
        return dp[m-1][n-1]
    
    def Unique_Paths_With_Obstacles_Space_Optimized(self, obstacleGrid: List[List[int]]) -> int:
        """
        Space Optimized - Use 1D array
        Time Complexity: O(m*n)
        Space Complexity: O(n)
        """
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        
        if obstacleGrid[0][0] == 1:
            return 0
        
        dp = [0] * n
        dp[0] = 1
        
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    dp[j] = 0
                elif j > 0:
                    dp[j] += dp[j-1]
        
        return dp[n-1]
    
    def Unique_Paths_With_Obstacles_In_Place(self, obstacleGrid: List[List[int]]) -> int:
        """
        In Place - Modify input grid for DP
        Time Complexity: O(m*n)
        Space Complexity: O(1)
        """
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        
        if obstacleGrid[0][0] == 1:
            return 0
        
        obstacleGrid[0][0] = 1
        
        for i in range(1, m):
            obstacleGrid[i][0] = int(obstacleGrid[i][0] == 0 and obstacleGrid[i-1][0] == 1)
        
        for j in range(1, n):
            obstacleGrid[0][j] = int(obstacleGrid[0][j] == 0 and obstacleGrid[0][j-1] == 1)
        
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:
                    obstacleGrid[i][j] = obstacleGrid[i-1][j] + obstacleGrid[i][j-1]
                else:
                    obstacleGrid[i][j] = 0
        
        return obstacleGrid[m-1][n-1]
    
    def Unique_Paths_With_Obstacles_BFS(self, obstacleGrid: List[List[int]]) -> int:
        """
        BFS - Use breadth-first search with path counting
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        from collections import deque
        
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        
        if obstacleGrid[0][0] == 1 or obstacleGrid[m-1][n-1] == 1:
            return 0
        
        paths_count = [[0] * n for _ in range(m)]
        paths_count[0][0] = 1
        
        queue = deque([(0, 0)])
        
        while queue:
            row, col = queue.popleft()
            
            directions = [(0, 1), (1, 0)]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < m and 0 <= new_col < n and 
                    obstacleGrid[new_row][new_col] == 0):
                    
                    if paths_count[new_row][new_col] == 0:
                        queue.append((new_row, new_col))
                    
                    paths_count[new_row][new_col] += paths_count[row][col]
        
        return paths_count[m-1][n-1]
    
    def Unique_Paths_With_Obstacles_Rolling_Array(self, obstacleGrid: List[List[int]]) -> int:
        """
        Rolling Array - Use two arrays alternating
        Time Complexity: O(m*n)
        Space Complexity: O(n)
        """
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        
        if obstacleGrid[0][0] == 1:
            return 0
        
        prev = [0] * n
        curr = [0] * n
        
        prev[0] = 1
        
        for j in range(1, n):
            prev[j] = prev[j-1] if obstacleGrid[0][j] == 0 else 0
        
        for i in range(1, m):
            curr[0] = prev[0] if obstacleGrid[i][0] == 0 else 0
            
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:
                    curr[j] = prev[j] + curr[j-1]
                else:
                    curr[j] = 0
            
            prev, curr = curr, prev
        
        return prev[n-1]
    
    def Unique_Paths_With_Obstacles_Path_Reconstruction(self, obstacleGrid: List[List[int]]) -> tuple:
        """
        With Path Reconstruction - Return count and sample path
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        
        if obstacleGrid[0][0] == 1 or obstacleGrid[m-1][n-1] == 1:
            return 0, []
        
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = 1
        
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] if obstacleGrid[i][0] == 0 else 0
        
        for j in range(1, n):
            dp[0][j] = dp[0][j-1] if obstacleGrid[0][j] == 0 else 0
        
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
                else:
                    dp[i][j] = 0
        
        if dp[m-1][n-1] == 0:
            return 0, []
        
        path = []
        i, j = m - 1, n - 1
        
        while i > 0 or j > 0:
            if i == 0:
                path.append('R')
                j -= 1
            elif j == 0:
                path.append('D')
                i -= 1
            elif dp[i-1][j] > 0 and dp[i][j-1] == 0:
                path.append('D')
                i -= 1
            elif dp[i][j-1] > 0:
                path.append('R')
                j -= 1
            else:
                path.append('D')
                i -= 1
        
        return dp[m-1][n-1], path[::-1]

def Test_Unique_Paths_With_Obstacles():
    solution = Solution()
    
    test_cases = [
        ([[0,0,0],[0,1,0],[0,0,0]], 2),
        ([[0,1],[0,0]], 1),
        ([[1]], 0),
        ([[0]], 1),
        ([[0,0],[1,1],[0,0]], 0),
        ([[0,0,0],[0,0,0],[0,0,0]], 6),
        ([[0,1,0],[0,0,0],[0,0,0]], 2)
    ]
    
    methods = [
        ("Memoized", solution.Unique_Paths_With_Obstacles_Memoized),
        ("Tabulation Optimal", solution.Unique_Paths_With_Obstacles_Tabulation_Optimal),
        ("Space Optimized", solution.Unique_Paths_With_Obstacles_Space_Optimized),
        ("BFS", solution.Unique_Paths_With_Obstacles_BFS),
        ("Rolling Array", solution.Unique_Paths_With_Obstacles_Rolling_Array)
    ]
    
    for grid, expected in test_cases:
        print(f"Grid: {grid}")
        print(f"Expected: {expected}")
        
        if len(grid) <= 4 and len(grid[0]) <= 4:
            result_rec = solution.Unique_Paths_With_Obstacles_Recursive([row[:] for row in grid])
            print(f"Recursive: {result_rec}")
        
        for method_name, method in methods:
            try:
                result = method([row[:] for row in grid])
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        count, path = solution.Unique_Paths_With_Obstacles_Path_Reconstruction([row[:] for row in grid])
        print(f"With Path: Count={count}, Sample Path={''.join(path)}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Unique_Paths_With_Obstacles()
