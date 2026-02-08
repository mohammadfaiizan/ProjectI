"""
Problem: Unbounded Knapsack
URL: https://practice.geeksforgeeks.org/problems/knapsack-with-duplicate-items4201/1

Problem Statement:
Given a knapsack weight W and a set of items with certain value and weight, we need to calculate the maximum amount that could make up this quantity exactly. This is different from classical knapsack problem, here we are allowed to use unlimited number of instances of an item.

Sample Input/Output:
Input: val=[1,30], wt=[1,50], W=100
Output: 100
"""


class Solution:
    def Unbounded_KS_Tab(self, val: list[int], wt: list[int], W: int) -> int:
        """
        Tabulation
        Time Complexity: O(n*W)
        Space Complexity: O(n*W)
        """
        n = len(val)
        dp = [[0] * (W + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(1, W + 1):
                if wt[i - 1] <= w:
                    dp[i][w] = max(dp[i - 1][w], val[i - 1] + dp[i][w - wt[i - 1]])
                else:
                    dp[i][w] = dp[i - 1][w]
        
        return dp[n][W]
    
    def Unbounded_KS_Space(self, val: list[int], wt: list[int], W: int) -> int:
        """
        Space Optimized
        Time Complexity: O(n*W)
        Space Complexity: O(W)
        """
        n = len(val)
        dp = [0] * (W + 1)
        
        for i in range(n):
            for w in range(wt[i], W + 1):
                dp[w] = max(dp[w], val[i] + dp[w - wt[i]])
        
        return dp[W]


def Test_UnboundedKnapsack():
    solution = Solution()
    val = [1, 30]
    wt = [1, 50]
    assert solution.Unbounded_KS_Tab(val, wt, 100) == 100
    assert solution.Unbounded_KS_Space(val, wt, 100) == 100


if __name__ == "__main__":
    Test_UnboundedKnapsack()
