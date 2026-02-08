"""
Problem: Chocolate Distribution Problem
URL: https://practice.geeksforgeeks.org/problems/chocolate-distribution-problem3825/1

Problem Statement:
Given an array of N integers where each value represents the number of chocolates in a packet.
There are M students, distribute chocolate packets such that each student gets one packet,
and the difference between max and min chocolates given is minimized.

Sample Input/Output:
Input: arr = [3, 4, 1, 9, 56, 7, 9, 12], M = 5
Output: 6
Explanation: Selected packets: [3, 4, 7, 9, 9]. Max-Min = 9-3 = 6.

Input: arr = [7, 3, 2, 4, 9, 12, 56], M = 3
Output: 2
Explanation: Selected packets: [2, 3, 4]. Max-Min = 4-2 = 2.
"""


class Solution:
    def Chocolate_Distribution_Sliding_Window_Optimal(self, arr, m):
        """
        Sorting + Sliding Window - Sort and check all windows of size m
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        arr.sort()
        n = len(arr)
        min_diff = arr[m - 1] - arr[0]
        for i in range(1, n - m + 1):
            min_diff = min(min_diff, arr[i + m - 1] - arr[i])
        return min_diff


def Test_Chocolate_Distribution():
    solution = Solution()

    test_cases = [
        ([3, 4, 1, 9, 56, 7, 9, 12], 5, 6),
        ([7, 3, 2, 4, 9, 12, 56], 3, 2),
        ([12, 4, 7, 9, 2, 23, 25, 41, 30, 40, 28, 42, 30, 44, 48, 43, 50], 7, 10)
    ]

    for arr, m, expected in test_cases:
        print(f"Chocolates: {arr}, M={m}, Expected={expected}")
        result = solution.Chocolate_Distribution_Sliding_Window_Optimal(arr, m)
        print(f"Sliding Window: {result}")
        print("-" * 50)


if __name__ == "__main__":
    Test_Chocolate_Distribution()
