"""
Problem: Smallest Subarray with Sum Greater Than X
URL: https://practice.geeksforgeeks.org/problems/smallest-subarray-with-sum-greater-than-x5651/1

Problem Statement:
Given an array of integers and a number x, find the smallest subarray with sum greater
than the given value x.

Sample Input/Output:
Input: arr = [1, 4, 45, 6, 0, 19], X = 51
Output: 3
Explanation: Subarray [4, 45, 6] has sum 55 > 51 with length 3.

Input: arr = [1, 10, 5, 2, 7], X = 9
Output: 1
Explanation: Element [10] has sum 10 > 9 with length 1.
"""


class Solution:
    def Smallest_Subarray_Sliding_Window_Optimal(self, arr, x):
        """
        Sliding Window - Expand right, shrink left when sum > x
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(arr)
        curr_sum = 0
        min_len = n + 1
        start = 0
        for end in range(n):
            curr_sum += arr[end]
            while curr_sum > x:
                min_len = min(min_len, end - start + 1)
                curr_sum -= arr[start]
                start += 1
        return min_len

    def Smallest_Subarray_Brute_Force(self, arr, x):
        """
        Brute Force - Check all subarrays
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        n = len(arr)
        min_len = n + 1
        for i in range(n):
            sum_val = 0
            for j in range(i, n):
                sum_val += arr[j]
                if sum_val > x:
                    min_len = min(min_len, j - i + 1)
                    break
        return min_len


def Test_Smallest_Subarray():
    solution = Solution()

    class TestCase:
        def __init__(self, arr, x, expected):
            self.arr = arr
            self.x = x
            self.expected = expected

    test_cases = [
        TestCase([1, 4, 45, 6, 0, 19], 51, 3),
        TestCase([1, 10, 5, 2, 7], 9, 1),
        TestCase([1, 11, 100, 1, 0, 200, 3, 2, 1, 250], 280, 4),
        TestCase([1, 2, 4], 8, 4)
    ]

    for tc in test_cases:
        print(f"Array: {tc.arr}, X={tc.x}, Expected={tc.expected}")

        print("Sliding Window:", solution.Smallest_Subarray_Sliding_Window_Optimal(tc.arr, tc.x))
        print("Brute Force:", solution.Smallest_Subarray_Brute_Force(tc.arr, tc.x))

        print("-" * 50)


if __name__ == "__main__":
    Test_Smallest_Subarray()
