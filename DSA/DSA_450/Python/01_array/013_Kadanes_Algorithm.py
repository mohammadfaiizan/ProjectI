"""
Problem: Kadane's Algorithm
URL: https://practice.geeksforgeeks.org/problems/kadanes-algorithm-1587115620/1

Problem Statement:
Given an integer array arr of size N, find the maximum sum subarray using Kadane's
Algorithm. The subarray must contain at least one element. Also track the subarray indices.

Sample Input/Output:
Input: arr = [-2, -3, 4, -1, -2, 1, 5, -3]
Output: 7
Explanation: Subarray [4, -1, -2, 1, 5] has maximum sum 7.

Input: arr = [1, 2, 3, -2, 5]
Output: 9
Explanation: Subarray [1, 2, 3, -2, 5] has maximum sum 9.
"""


class Solution:
    def Kadane_Standard_Optimal(self, arr):
        """
        Standard Kadane's - Track max ending here and global max
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        max_ending_here = max_so_far = arr[0]
        for i in range(1, len(arr)):
            max_ending_here = max(arr[i], max_ending_here + arr[i])
            max_so_far = max(max_so_far, max_ending_here)
        return max_so_far

    def Kadane_With_Indices(self, arr):
        """
        Kadane's with Subarray Indices - Track start and end of max subarray
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        max_ending_here = max_so_far = arr[0]
        start = end = temp_start = 0
        for i in range(1, len(arr)):
            if arr[i] > max_ending_here + arr[i]:
                max_ending_here = arr[i]
                temp_start = i
            else:
                max_ending_here += arr[i]
            if max_ending_here > max_so_far:
                max_so_far = max_ending_here
                start = temp_start
                end = i
        return (max_so_far, (start, end))

    def Kadane_DP_Array(self, arr):
        """
        DP Array Variant - Explicit DP array for max sum ending at each index
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


def Test_Kadanes_Algorithm():
    solution = Solution()

    class TestCase:
        def __init__(self, arr, expected):
            self.arr = arr
            self.expected = expected

    test_cases = [
        TestCase([-2, -3, 4, -1, -2, 1, 5, -3], 7),
        TestCase([1, 2, 3, -2, 5], 9),
        TestCase([-1, -2, -3, -4], -1),
        TestCase([5, 4, -1, 7, 8], 23)
    ]

    for tc in test_cases:
        print(f"Array: {tc.arr}, Expected: {tc.expected}")

        print("Standard:", solution.Kadane_Standard_Optimal(tc.arr))

        sum_val, indices = solution.Kadane_With_Indices(tc.arr)
        print(f"With Indices: Sum={sum_val}, Range=[{indices[0]}, {indices[1]}]")

        print("DP Array:", solution.Kadane_DP_Array(tc.arr))

        print("-" * 50)


if __name__ == "__main__":
    Test_Kadanes_Algorithm()
