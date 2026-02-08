"""
Problem: 0/1 Knapsack
URL: https://practice.geeksforgeeks.org/problems/0-1-knapsack-problem0945/1

Problem Statement:
Given weights and values of n items, put these items in a knapsack of capacity W to get the maximum total value in the knapsack. In other words, given two integer arrays val[0..n-1] and wt[0..n-1] which represent values and weights associated with n items respectively. Also given an integer W which represents knapsack capacity, find out the maximum value subset of val[] such that sum of the weights of this subset is smaller than or equal to W. You cannot break an item, either pick the complete item or don't pick it (0-1 property).

Sample Input/Output:
Input: val = [60, 100, 120], wt = [10, 20, 30], W = 50
Output: 220
"""

class Solution:
    def Knapsack_Recursive(self, val, wt, n, W):
        """
        Recursive approach
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        if n == 0 or W == 0:
            return 0
        if wt[n-1] > W:
            return self.Knapsack_Recursive(val, wt, n-1, W)
        return max(val[n-1] + self.Knapsack_Recursive(val, wt, n-1, W-wt[n-1]),
                   self.Knapsack_Recursive(val, wt, n-1, W))

    def Knapsack_Memoization(self, val, wt, n, W):
        """
        Memoization approach
        Time Complexity: O(n*W)
        Space Complexity: O(n*W)
        """
        dp = [[-1] * (W+1) for _ in range(n+1)]
        return self.Knapsack_Memo_Helper(val, wt, n, W, dp)

    def Knapsack_Memo_Helper(self, val, wt, n, W, dp):
        if n == 0 or W == 0:
            return 0
        if dp[n][W] != -1:
            return dp[n][W]
        if wt[n-1] > W:
            dp[n][W] = self.Knapsack_Memo_Helper(val, wt, n-1, W, dp)
        else:
            dp[n][W] = max(val[n-1] + self.Knapsack_Memo_Helper(val, wt, n-1, W-wt[n-1], dp),
                          self.Knapsack_Memo_Helper(val, wt, n-1, W, dp))
        return dp[n][W]

    def Knapsack_Tabulation(self, val, wt, n, W):
        """
        Tabulation approach
        Time Complexity: O(n*W)
        Space Complexity: O(n*W)
        """
        dp = [[0] * (W+1) for _ in range(n+1)]
        for i in range(1, n+1):
            for w in range(1, W+1):
                if wt[i-1] <= w:
                    dp[i][w] = max(val[i-1] + dp[i-1][w-wt[i-1]], dp[i-1][w])
                else:
                    dp[i][w] = dp[i-1][w]
        return dp[n][W]

    def Knapsack_Space_Optimized(self, val, wt, n, W):
        """
        Space optimized approach
        Time Complexity: O(n*W)
        Space Complexity: O(W)
        """
        dp = [0] * (W+1)
        for i in range(n):
            for w in range(W, wt[i]-1, -1):
                dp[w] = max(dp[w], val[i] + dp[w-wt[i]])
        return dp[W]

def Test_01_Knapsack():
    solution = Solution()
    val = [60, 100, 120]
    wt = [10, 20, 30]
    W = 50
    n = len(val)
    
    print("Recursive:", solution.Knapsack_Recursive(val, wt, n, W))
    print("Memoization:", solution.Knapsack_Memoization(val, wt, n, W))
    print("Tabulation:", solution.Knapsack_Tabulation(val, wt, n, W))
    print("Space Optimized:", solution.Knapsack_Space_Optimized(val, wt, n, W))

if __name__ == "__main__":
    Test_01_Knapsack()
