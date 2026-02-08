"""
Problem: Maximum Sum Increasing Subsequence
URL: https://practice.geeksforgeeks.org/problems/maximum-sum-increasing-subsequence4749/1

Problem Statement:
Given an array of n positive integers. Find the sum of maximum sum increasing subsequence of the given array.

Sample Input/Output:
Input: [1, 101, 2, 3, 100, 4, 5]
Output: 106
"""

class Solution:
    def MSIS_DP(self, arr, n):
        """
        DP approach
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        dp = [0] * n
        for i in range(n):
            dp[i] = arr[i]
        for i in range(1, n):
            for j in range(i):
                if arr[j] < arr[i]:
                    dp[i] = max(dp[i], dp[j] + arr[i])
        return max(dp)

def Test_MSIS():
    solution = Solution()
    arr = [1, 101, 2, 3, 100, 4, 5]
    print("Max Sum:", solution.MSIS_DP(arr, len(arr)))

if __name__ == "__main__":
    Test_MSIS()
