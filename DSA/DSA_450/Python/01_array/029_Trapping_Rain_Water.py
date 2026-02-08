"""
Problem: Trapping Rain Water
URL: https://practice.geeksforgeeks.org/problems/trapping-rain-water-1587115621/1

Problem Statement:
Given an array arr[] of N non-negative integers representing an elevation map where
the width of each bar is 1, compute how much water it can trap after raining.

Sample Input/Output:
Input: arr = [3, 0, 2, 0, 4]
Output: 7

Input: arr = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
Output: 6
"""


class Solution:
    def Trap_Water_Two_Pointer_Optimal(self, arr):
        """
        Two Pointer Approach - Left and right pointers with max tracking
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left = 0
        right = len(arr) - 1
        left_max = right_max = water = 0
        while left <= right:
            if arr[left] <= arr[right]:
                if arr[left] >= left_max:
                    left_max = arr[left]
                else:
                    water += left_max - arr[left]
                left += 1
            else:
                if arr[right] >= right_max:
                    right_max = arr[right]
                else:
                    water += right_max - arr[right]
                right -= 1
        return water

    def Trap_Water_Prefix_Suffix(self, arr):
        """
        Prefix-Suffix Max Arrays - Precompute left and right max heights
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(arr)
        left_max = [0] * n
        right_max = [0] * n
        left_max[0] = arr[0]
        right_max[n - 1] = arr[n - 1]
        for i in range(1, n):
            left_max[i] = max(arr[i], left_max[i - 1])
        for i in range(n - 2, -1, -1):
            right_max[i] = max(arr[i], right_max[i + 1])
        water = 0
        for i in range(1, n - 1):
            water += max(0, min(left_max[i], right_max[i]) - arr[i])
        return water


def Test_Trapping_Rain_Water():
    solution = Solution()

    class TestCase:
        def __init__(self, arr, expected):
            self.arr = arr
            self.expected = expected

    test_cases = [
        TestCase([3, 0, 2, 0, 4], 7),
        TestCase([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1], 6),
        TestCase([4, 2, 0, 3, 2, 5], 9),
        TestCase([1, 2, 3, 4, 5], 0)
    ]

    for tc in test_cases:
        print(f"Heights: {tc.arr}, Expected: {tc.expected}")

        print("Two Pointer:", solution.Trap_Water_Two_Pointer_Optimal(tc.arr))
        print("Prefix-Suffix:", solution.Trap_Water_Prefix_Suffix(tc.arr))

        print("-" * 50)


if __name__ == "__main__":
    Test_Trapping_Rain_Water()
