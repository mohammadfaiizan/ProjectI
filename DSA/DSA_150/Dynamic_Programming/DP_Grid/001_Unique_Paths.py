"""
Problem: Unique Paths
URL: https://leetcode.com/problems/unique-paths/

Problem Statement:
There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). 
The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). 
The robot can only move either down or right at any point in time.
Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

Sample Input/Output:
Input: m = 3, n = 7
Output: 28

Input: m = 3, n = 2
Output: 3
Explanation: From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Down -> Down
2. Down -> Right -> Down  
3. Down -> Down -> Right
"""

from typing import List
import math

class Solution:
    def Unique_Paths_Recursive(self, m: int, n: int) -> int:
        """
        Recursive - Try both directions from each cell
        Time Complexity: O(2^(m+n))
        Space Complexity: O(m+n)
        """
        def Count_Paths(row: int, col: int) -> int:
            if row == m - 1 and col == n - 1:
                return 1
            
            if row >= m or col >= n:
                return 0
            
            right_paths = Count_Paths(row, col + 1)
            down_paths = Count_Paths(row + 1, col)
            
            return right_paths + down_paths
        
        return Count_Paths(0, 0)
    
    def Unique_Paths_Memoized(self, m: int, n: int) -> int:
        """
        Memoized - Top-down DP with memoization
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        memo = {}
        
        def Count_Paths_Memo(row: int, col: int) -> int:
            if row == m - 1 and col == n - 1:
                return 1
            
            if row >= m or col >= n:
                return 0
            
            if (row, col) in memo:
                return memo[(row, col)]
            
            right_paths = Count_Paths_Memo(row, col + 1)
            down_paths = Count_Paths_Memo(row + 1, col)
            
            memo[(row, col)] = right_paths + down_paths
            return memo[(row, col)]
        
        return Count_Paths_Memo(0, 0)
    
    def Unique_Paths_Tabulation_2D(self, m: int, n: int) -> int:
        """
        Tabulation 2D - Bottom-up DP with 2D table
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        dp = [[0] * n for _ in range(m)]
        
        for i in range(m):
            dp[i][0] = 1
        
        for j in range(n):
            dp[0][j] = 1
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        
        return dp[m-1][n-1]
    
    def Unique_Paths_Space_Optimized_Optimal(self, m: int, n: int) -> int:
        """
        Space Optimized Optimal - Use 1D array
        Time Complexity: O(m*n)
        Space Complexity: O(min(m,n))
        """
        if m < n:
            m, n = n, m
        
        dp = [1] * n
        
        for i in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j-1]
        
        return dp[n-1]
    
    def Unique_Paths_Mathematical_Formula(self, m: int, n: int) -> int:
        """
        Mathematical Formula - Combinatorial approach
        Time Complexity: O(min(m,n))
        Space Complexity: O(1)
        """
        total_moves = m + n - 2
        down_moves = m - 1
        
        return math.comb(total_moves, down_moves)
    
    def Unique_Paths_Mathematical_Optimized(self, m: int, n: int) -> int:
        """
        Mathematical Optimized - Avoid large factorials
        Time Complexity: O(min(m,n))
        Space Complexity: O(1)
        """
        total_moves = m + n - 2
        down_moves = min(m - 1, n - 1)
        
        result = 1
        
        for i in range(down_moves):
            result = result * (total_moves - i) // (i + 1)
        
        return result
    
    def Unique_Paths_Pascal_Triangle(self, m: int, n: int) -> int:
        """
        Pascal Triangle - Use Pascal's triangle property
        Time Complexity: O(m*n)
        Space Complexity: O(n)
        """
        if m == 1 or n == 1:
            return 1
        
        prev_row = [1] * n
        
        for i in range(1, m):
            curr_row = [1] * n
            
            for j in range(1, n):
                curr_row[j] = curr_row[j-1] + prev_row[j]
            
            prev_row = curr_row
        
        return prev_row[n-1]
    
    def Unique_Paths_Bottom_Up_Alternative(self, m: int, n: int) -> int:
        """
        Bottom Up Alternative - Different iteration order
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        dp = [[1] * n for _ in range(m)]
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        
        return dp[m-1][n-1]
    
    def Unique_Paths_With_Path_Reconstruction(self, m: int, n: int) -> tuple:
        """
        With Path Reconstruction - Return count and sample path
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        dp = [[0] * n for _ in range(m)]
        
        for i in range(m):
            dp[i][0] = 1
        
        for j in range(n):
            dp[0][j] = 1
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        
        path = []
        i, j = m - 1, n - 1
        
        while i > 0 or j > 0:
            if i == 0:
                path.append('R')
                j -= 1
            elif j == 0:
                path.append('D')
                i -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                path.append('D')
                i -= 1
            else:
                path.append('R')
                j -= 1
        
        return dp[m-1][n-1], path[::-1]

def Test_Unique_Paths():
    solution = Solution()
    
    test_cases = [
        (3, 7, 28),
        (3, 2, 3),
        (7, 3, 28),
        (3, 3, 6),
        (1, 1, 1),
        (1, 10, 1),
        (10, 1, 1),
        (4, 4, 20)
    ]
    
    methods = [
        ("Memoized", solution.Unique_Paths_Memoized),
        ("Tabulation 2D", solution.Unique_Paths_Tabulation_2D),
        ("Space Optimized Optimal", solution.Unique_Paths_Space_Optimized_Optimal),
        ("Mathematical Formula", solution.Unique_Paths_Mathematical_Formula),
        ("Mathematical Optimized", solution.Unique_Paths_Mathematical_Optimized),
        ("Pascal Triangle", solution.Unique_Paths_Pascal_Triangle),
        ("Bottom Up Alternative", solution.Unique_Paths_Bottom_Up_Alternative)
    ]
    
    for m, n, expected in test_cases:
        print(f"Grid: {m}x{n}")
        print(f"Expected: {expected}")
        
        if m <= 5 and n <= 5:
            result_rec = solution.Unique_Paths_Recursive(m, n)
            print(f"Recursive: {result_rec}")
        
        for method_name, method in methods:
            try:
                result = method(m, n)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        if m <= 6 and n <= 6:
            count, path = solution.Unique_Paths_With_Path_Reconstruction(m, n)
            print(f"With Path: Count={count}, Sample Path={''.join(path)}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Unique_Paths()
