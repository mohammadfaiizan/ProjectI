"""
Problem: Minimum Cost to Fill Bag
URL: https://practice.geeksforgeeks.org/problems/minimum-cost-to-fill-given-weight-in-a-bag1956/1

Problem Statement:
Given an array cost[] of positive integers of size n where cost[i] represents the cost of i kg packet of oranges, the task is to find the minimum cost to buy W kgs of oranges. If it is impossible to buy exactly W kg oranges then the output will be -1.

Sample Input/Output:
Input: cost = [20, 10, 4, 50, 100], W = 5
Output: 14
"""

import sys

class Solution:
    def Min_Cost_DP(self, cost, n, W):
        """
        DP approach
        Time Complexity: O(n*W)
        Space Complexity: O(W)
        """
        dp = [sys.maxsize] * (W+1)
        dp[0] = 0
        for i in range(1, W+1):
            for j in range(min(n, i)):
                if cost[j] != -1 and dp[i-j-1] != sys.maxsize:
                    dp[i] = min(dp[i], cost[j] + dp[i-j-1])
        return -1 if dp[W] == sys.maxsize else dp[W]

def Test_Min_Cost():
    solution = Solution()
    cost = [20, 10, 4, 50, 100]
    W = 5
    print("Min Cost:", solution.Min_Cost_DP(cost, len(cost), W))

if __name__ == "__main__":
    Test_Min_Cost()
