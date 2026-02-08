"""
Problem: Matrix Chain Multiplication
URL: https://practice.geeksforgeeks.org/problems/matrix-chain-multiplication0303/1

Problem Statement:
Given an array p[] which represents the chain of matrices such that the ith matrix Ai is of dimension p[i-1] x p[i]. Find the minimum number of multiplications needed to multiply the chain.

Sample Input/Output:
Input: p = [40,20,30,10,30]
Output: 26000
"""


class Solution:
    def MCM_Recursive(self, p: list[int], i: int, j: int) -> int:
        """
        Recursive approach
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        if i >= j:
            return 0
        
        min_cost = float('inf')
        for k in range(i, j):
            cost = (self.MCM_Recursive(p, i, k) + 
                   self.MCM_Recursive(p, k + 1, j) + 
                   p[i - 1] * p[k] * p[j])
            min_cost = min(min_cost, cost)
        
        return min_cost
    
    def MCM_Memo(self, p: list[int], i: int, j: int, dp: list[list[int]]) -> int:
        """
        Memoization approach
        Time Complexity: O(n^3)
        Space Complexity: O(n^2)
        """
        if i >= j:
            return 0
        if dp[i][j] != -1:
            return dp[i][j]
        
        min_cost = float('inf')
        for k in range(i, j):
            cost = (self.MCM_Memo(p, i, k, dp) + 
                   self.MCM_Memo(p, k + 1, j, dp) + 
                   p[i - 1] * p[k] * p[j])
            min_cost = min(min_cost, cost)
        
        dp[i][j] = min_cost
        return dp[i][j]
    
    def MCM_Tab(self, p: list[int]) -> int:
        """
        Tabulation approach
        Time Complexity: O(n^3)
        Space Complexity: O(n^2)
        """
        n = len(p)
        dp = [[0] * n for _ in range(n)]
        
        for length in range(2, n):
            for i in range(1, n - length + 1):
                j = i + length - 1
                dp[i][j] = float('inf')
                
                for k in range(i, j):
                    cost = dp[i][k] + dp[k + 1][j] + p[i - 1] * p[k] * p[j]
                    dp[i][j] = min(dp[i][j], cost)
        
        return dp[1][n - 1]


def Test_MatrixChainMultiplication():
    solution = Solution()
    
    p = [40, 20, 30, 10, 30]
    
    result1 = solution.MCM_Recursive(p, 1, len(p) - 1)
    assert result1 == 26000
    
    dp = [[-1] * len(p) for _ in range(len(p))]
    result2 = solution.MCM_Memo(p, 1, len(p) - 1, dp)
    assert result2 == 26000
    
    result3 = solution.MCM_Tab(p)
    assert result3 == 26000


if __name__ == "__main__":
    Test_MatrixChainMultiplication()
