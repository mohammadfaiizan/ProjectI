"""
Problem: Kadane's Algorithm
URL: https://practice.geeksforgeeks.org/problems/kadanes-algorithm-1587115620/1

Problem Statement:
Given an array Arr[] of N integers. Find the contiguous sub-array(containing at least one number) which has the maximum sum and return its sum.

Sample Input/Output:
Input: [-2,-3,4,-1,-2,1,5,-3]
Output: 7
"""


class Solution:
    def Kadane_Standard(self, arr: list[int]) -> int:
        """
        Standard Kadane's Algorithm
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(arr)
        max_sum = arr[0]
        current_sum = arr[0]
        
        for i in range(1, n):
            current_sum = max(arr[i], current_sum + arr[i])
            max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    def Kadane_DP(self, arr: list[int]) -> int:
        """
        Dynamic Programming Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(arr)
        dp = [0] * n
        dp[0] = arr[0]
        max_sum = dp[0]
        
        for i in range(1, n):
            dp[i] = max(arr[i], dp[i - 1] + arr[i])
            max_sum = max(max_sum, dp[i])
        
        return max_sum


def Test_KadanesAlgorithm():
    solution = Solution()
    arr = [-2, -3, 4, -1, -2, 1, 5, -3]
    assert solution.Kadane_Standard(arr) == 7
    assert solution.Kadane_DP(arr) == 7


if __name__ == "__main__":
    Test_KadanesAlgorithm()
