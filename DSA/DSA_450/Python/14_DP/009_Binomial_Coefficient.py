"""
Problem: Binomial Coefficient
URL: https://practice.geeksforgeeks.org/problems/ncr1019/1

Problem Statement:
Given two integers n and r, find nCr. Since the answer may be very large, calculate the answer modulo 10^9+7.

Sample Input/Output:
Input: n = 5, r = 2
Output: 10
"""

class Solution:
    def Binomial_Coefficient_Binomial_Recursive(self, n, r):
        """
        Recursive approach
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        if r == 0 or r == n:
            return 1
        return self.Binomial_Coefficient_Binomial_Recursive(n-1, r-1) + \
               self.Binomial_Coefficient_Binomial_Recursive(n-1, r)

    def Binomial_Coefficient_Binomial_DP(self, n, r):
        """
        DP approach
        Time Complexity: O(n*r)
        Space Complexity: O(n*r)
        """
        if r > n:
            return 0
        dp = [[0] * (r+1) for _ in range(n+1)]
        for i in range(n+1):
            for j in range(min(i, r) + 1):
                if j == 0 or j == i:
                    dp[i][j] = 1
                else:
                    dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
        return dp[n][r]

    def Binomial_Coefficient_Binomial_Optimized(self, n, r):
        """
        Space optimized approach
        Time Complexity: O(n*r)
        Space Complexity: O(r)
        """
        if r > n:
            return 0
        if r > n - r:
            r = n - r
        dp = [0] * (r+1)
        dp[0] = 1
        for i in range(1, n+1):
            for j in range(min(i, r), 0, -1):
                dp[j] = dp[j] + dp[j-1]
        return dp[r]

def Test_Binomial_Coefficient():
    solution = Solution()
    n, r = 5, 2
    
    print("Recursive:", solution.Binomial_Coefficient_Binomial_Recursive(n, r))
    print("DP:", solution.Binomial_Coefficient_Binomial_DP(n, r))
    print("Optimized:", solution.Binomial_Coefficient_Binomial_Optimized(n, r))

if __name__ == "__main__":
    Test_Binomial_Coefficient()
