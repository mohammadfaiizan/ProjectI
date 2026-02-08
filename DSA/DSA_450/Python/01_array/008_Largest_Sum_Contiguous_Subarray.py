"""
Problem: Largest Sum Contiguous Subarray
URL: https://practice.geeksforgeeks.org/problems/kadanes-algorithm-1587115620/1

Problem Statement:
Given an array arr[] of N integers, find the contiguous sub-array (containing at least
one number) which has the maximum sum and return its sum.

Sample Input/Output:
Input: arr = [1, 2, 3, -2, 5]
Output: 9
Explanation: Max subarray sum is 1 + 2 + 3 + (-2) + 5 = 9.

Input: arr = [-1, -2, -3, -4]
Output: -1
Explanation: Max subarray sum is -1 (single element).
"""


class Solution:
    def Max_Subarray_Kadane_Optimal(self, arr):
        """
        Kadane's Algorithm - Track current sum and global max
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        current_sum = max_sum = arr[0]
        for i in range(1, len(arr)):
            current_sum = max(arr[i], current_sum + arr[i])
            max_sum = max(max_sum, current_sum)
        return max_sum

    def Max_Subarray_DP(self, arr):
        """
        DP Array - Store max subarray sum ending at each index
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

    def Max_Subarray_Brute_Force(self, arr):
        """
        Brute Force - Check all subarrays
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        n = len(arr)
        max_sum = float('-inf')
        for i in range(n):
            current_sum = 0
            for j in range(i, n):
                current_sum += arr[j]
                max_sum = max(max_sum, current_sum)
        return max_sum


def Test_Largest_Sum_Contiguous_Subarray():
    solution = Solution()

    class TestCase:
        def __init__(self, arr, expected):
            self.arr = arr
            self.expected = expected

    test_cases = [
        TestCase([1, 2, 3, -2, 5], 9),
        TestCase([-1, -2, -3, -4], -1),
        TestCase([-2, -3, 4, -1, -2, 1, 5, -3], 7),
        TestCase([5, 4, -1, 7, 8], 23)
    ]

    for tc in test_cases:
        print(f"Array: {tc.arr}, Expected: {tc.expected}")

        print("Kadane's:", solution.Max_Subarray_Kadane_Optimal(tc.arr))
        print("DP:", solution.Max_Subarray_DP(tc.arr))
        print("Brute Force:", solution.Max_Subarray_Brute_Force(tc.arr))

        print("-" * 50)


if __name__ == "__main__":
    Test_Largest_Sum_Contiguous_Subarray()
