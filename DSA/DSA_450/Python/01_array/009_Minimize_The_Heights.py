"""
Problem: Minimize the Heights
URL: https://practice.geeksforgeeks.org/problems/minimize-the-heights3351/1

Problem Statement:
Given an array arr[] denoting heights of N towers and a positive integer K, for each tower
you must perform exactly one of: increase height by K or decrease height by K.
Find the minimum possible difference between the tallest and shortest towers.
Negative heights are not allowed.

Sample Input/Output:
Input: K = 2, arr = [1, 5, 8, 10]
Output: 5
Explanation: Modified array [3, 3, 6, 8]. Diff = 8 - 3 = 5.

Input: K = 3, arr = [3, 9, 12, 16, 20]
Output: 11
Explanation: Modified array [6, 12, 9, 13, 17]. Diff = 17 - 6 = 11.
"""


class Solution:
    def Minimize_Heights_Sorting_Optimal(self, arr, k):
        """
        Sorting + Greedy - Sort and try all split points
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        n = len(arr)
        if n == 1:
            return 0
        arr.sort()
        ans = arr[n - 1] - arr[0]
        for i in range(1, n):
            if arr[i] - k < 0:
                continue
            curr_min = min(arr[0] + k, arr[i] - k)
            curr_max = max(arr[n - 1] - k, arr[i - 1] + k)
            ans = min(ans, curr_max - curr_min)
        return ans


def Test_Minimize_The_Heights():
    solution = Solution()

    test_cases = [
        ([1, 5, 8, 10], 2, 5),
        ([3, 9, 12, 16, 20], 3, 11),
        ([1], 10, 0),
        ([1, 10, 14, 14, 14, 15], 6, 5)
    ]

    for arr, k, expected in test_cases:
        print(f"Array: {arr}, K={k}, Expected={expected}")
        result = solution.Minimize_Heights_Sorting_Optimal(arr, k)
        print(f"Sorting+Greedy: {result}")
        print("-" * 50)


if __name__ == "__main__":
    Test_Minimize_The_Heights()
