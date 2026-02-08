"""
Problem: Subarray with Sum 0
URL: https://practice.geeksforgeeks.org/problems/subarray-with-0-sum-1587115621/1

Problem Statement:
Given an array of positive and negative numbers, find if there is a subarray
(of size at least one) with 0 sum.

Sample Input/Output:
Input: arr = [4, 2, -3, 1, 6]
Output: true
Explanation: Subarray [2, -3, 1] has sum 0.

Input: arr = [4, 2, 0, 1, 6]
Output: true
Explanation: Subarray [0] has sum 0.
"""


class Solution:
    def Subarray_Sum_Zero_Hashing_Optimal(self, arr):
        """
        Prefix Sum + HashSet - Check for repeated prefix sums
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        prefix_sums = set()
        sum_val = 0
        for x in arr:
            sum_val += x
            if sum_val == 0 or sum_val in prefix_sums:
                return True
            prefix_sums.add(sum_val)
        return False

    def Subarray_Sum_Zero_Map(self, arr):
        """
        Prefix Sum + HashMap - Use map for prefix sum tracking
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        sum_map = {}
        sum_val = 0
        for i in range(len(arr)):
            sum_val += arr[i]
            if sum_val == 0 or sum_val in sum_map:
                return True
            sum_map[sum_val] = True
        return False

    def Subarray_Sum_Zero_Brute_Force(self, arr):
        """
        Brute Force - Check all subarrays
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        n = len(arr)
        for i in range(n):
            sum_val = 0
            for j in range(i, n):
                sum_val += arr[j]
                if sum_val == 0:
                    return True
        return False


def Test_Subarray_With_Sum_Zero():
    solution = Solution()

    class TestCase:
        def __init__(self, arr, expected):
            self.arr = arr
            self.expected = expected

    test_cases = [
        TestCase([4, 2, -3, 1, 6], True),
        TestCase([4, 2, 0, 1, 6], True),
        TestCase([-3, 2, 3, 1, 6], False),
        TestCase([1, -1], True)
    ]

    for tc in test_cases:
        print(f"Array: {tc.arr}, Expected: {tc.expected}")

        print("Hashing:", solution.Subarray_Sum_Zero_Hashing_Optimal(tc.arr))
        print("Map:", solution.Subarray_Sum_Zero_Map(tc.arr))
        print("Brute Force:", solution.Subarray_Sum_Zero_Brute_Force(tc.arr))

        print("-" * 50)


if __name__ == "__main__":
    Test_Subarray_With_Sum_Zero()
