"""
Problem: Longest Alternating Subsequence
URL: https://practice.geeksforgeeks.org/problems/longest-alternating-subsequence5951/1

Problem Statement:
Given an array of integers, find the longest alternating subsequence. A sequence is alternating if the elements alternate between increasing and decreasing.

Sample Input/Output:
Input: [1,5,4]
Output: 3
"""


class Solution:
    def LAS_DP(self, arr: list[int]) -> int:
        """
        Dynamic Programming
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        n = len(arr)
        if n <= 1:
            return n
        
        dp = [[1, 1] for _ in range(n)]
        result = 1
        
        for i in range(1, n):
            for j in range(i):
                if arr[j] < arr[i]:
                    dp[i][0] = max(dp[i][0], dp[j][1] + 1)
                elif arr[j] > arr[i]:
                    dp[i][1] = max(dp[i][1], dp[j][0] + 1)
            result = max(result, max(dp[i][0], dp[i][1]))
        
        return result
    
    def LAS_Optimized(self, arr: list[int]) -> int:
        """
        Optimized Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(arr)
        if n <= 1:
            return n
        
        inc = 1
        dec = 1
        
        for i in range(1, n):
            if arr[i] > arr[i - 1]:
                inc = dec + 1
            elif arr[i] < arr[i - 1]:
                dec = inc + 1
        
        return max(inc, dec)


def Test_LongestAlternatingSubsequence():
    solution = Solution()
    arr = [1, 5, 4]
    assert solution.LAS_DP(arr) == 3
    assert solution.LAS_Optimized(arr) == 3


if __name__ == "__main__":
    Test_LongestAlternatingSubsequence()
