"""
Problem: Maximum Sum Pairs with Specific Difference
URL: https://www.geeksforgeeks.org/maximum-sum-pairs-specific-difference/

Problem Statement:
Given an array of integers and a number k, find the maximum sum of pairs such that the difference between elements in each pair is less than k.

Sample Input/Output:
Input: arr = [3,5,10,15,17,12,9], k = 4
Output: 62
"""


class Solution:
    def Max_Pairs_DP(self, arr: list[int], k: int) -> int:
        """
        DP approach after sorting
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        n = len(arr)
        arr.sort()
        
        dp = [0] * n
        dp[0] = 0
        
        for i in range(1, n):
            dp[i] = dp[i - 1]
            
            if arr[i] - arr[i - 1] < k:
                prev = dp[i - 2] if i >= 2 else 0
                dp[i] = max(dp[i], prev + arr[i] + arr[i - 1])
        
        return dp[n - 1]


def Test_MaxSumPairsDiff():
    solution = Solution()
    arr = [3, 5, 10, 15, 17, 12, 9]
    k = 4
    result = solution.Max_Pairs_DP(arr, k)
    assert result == 62


if __name__ == "__main__":
    Test_MaxSumPairsDiff()
