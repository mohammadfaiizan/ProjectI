"""
Problem: The Chunin Ninja
URL: https://www.naukri.com/code360/problems/the-chunin-ninja

Problem Statement:
A ninja has to collect maximum coins placed in a grid. The ninja can only move right or down 
from any cell. Return the maximum coins the ninja can collect starting from (0,0) to (m-1,n-1).

Sample Input/Output:
Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
Output: 12
Explanation: Path: 1→3→5→2→1 = 12

Input: grid = [[1,2,3],[4,5,6]]
Output: 16
Explanation: Path: 1→4→5→6 = 16

Input: grid = [[1]]
Output: 1
"""

from typing import List

class Solution:
    def Max_Coins_Recursive(self, grid: List[List[int]]) -> int:
        """
        Recursive Approach
        Time Complexity: O(2^(m+n))
        Space Complexity: O(m + n)
        """
        def Helper(i: int, j: int) -> int:
            if i >= len(grid) or j >= len(grid[0]):
                return float('-inf')
            
            if i == len(grid) - 1 and j == len(grid[0]) - 1:
                return grid[i][j]
            
            right = Helper(i, j + 1)
            down = Helper(i + 1, j)
            
            return grid[i][j] + max(right, down)
        
        return Helper(0, 0)
    
    def Max_Coins_Memoization(self, grid: List[List[int]]) -> int:
        """
        Memoization Approach - Top-down DP
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        memo = {}
        
        def Helper(i: int, j: int) -> int:
            if i >= len(grid) or j >= len(grid[0]):
                return float('-inf')
            
            if i == len(grid) - 1 and j == len(grid[0]) - 1:
                return grid[i][j]
            
            if (i, j) in memo:
                return memo[(i, j)]
            
            right = Helper(i, j + 1)
            down = Helper(i + 1, j)
            
            memo[(i, j)] = grid[i][j] + max(right, down)
            return memo[(i, j)]
        
        return Helper(0, 0)
    
    def Max_Coins_DP_Tabulation(self, grid: List[List[int]]) -> int:
        """
        Dynamic Programming Tabulation - Bottom-up
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]
        
        dp[0][0] = grid[0][0]
        
        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        
        for j in range(1, n):
            dp[0][j] = dp[0][j - 1] + grid[0][j]
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = grid[i][j] + max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m - 1][n - 1]
    
    def Max_Coins_DP_Optimized(self, grid: List[List[int]]) -> int:
        """
        Space Optimized DP - Optimal solution
        Time Complexity: O(m * n)
        Space Complexity: O(n)
        """
        m, n = len(grid), len(grid[0])
        prev = [0] * n
        
        for i in range(m):
            curr = [0] * n
            for j in range(n):
                if i == 0 and j == 0:
                    curr[j] = grid[i][j]
                elif i == 0:
                    curr[j] = curr[j - 1] + grid[i][j]
                elif j == 0:
                    curr[j] = prev[j] + grid[i][j]
                else:
                    curr[j] = grid[i][j] + max(prev[j], curr[j - 1])
            prev = curr
        
        return prev[n - 1]
    
    def Max_Coins_In_Place(self, grid: List[List[int]]) -> int:
        """
        In-Place DP Modification
        Time Complexity: O(m * n)
        Space Complexity: O(1)
        """
        m, n = len(grid), len(grid[0])
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                elif i == 0:
                    grid[i][j] += grid[i][j - 1]
                elif j == 0:
                    grid[i][j] += grid[i - 1][j]
                else:
                    grid[i][j] += max(grid[i - 1][j], grid[i][j - 1])
        
        return grid[m - 1][n - 1]

def Test_Max_Coins():
    solution = Solution()
    
    test_cases = [
        ([[1,3,1],[1,5,1],[4,2,1]], 12),
        ([[1,2,3],[4,5,6]], 16),
        ([[1]], 1),
        ([[1,2],[1,1]], 5),
        ([[5,1,2],[3,6,1]], 17)
    ]
    
    for grid, expected in test_cases:
        result1 = solution.Max_Coins_Recursive([row[:] for row in grid])
        result2 = solution.Max_Coins_Memoization([row[:] for row in grid])
        result3 = solution.Max_Coins_DP_Tabulation([row[:] for row in grid])
        result4 = solution.Max_Coins_DP_Optimized([row[:] for row in grid])
        result5 = solution.Max_Coins_In_Place([row[:] for row in grid])
        
        print(f"Grid: {grid}")
        print(f"Expected: {expected}")
        print(f"Recursive: {result1}")
        print(f"Memoization: {result2}")
        print(f"DP Tabulation: {result3}")
        print(f"DP Optimized: {result4}")
        print(f"In-Place: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Max_Coins()

