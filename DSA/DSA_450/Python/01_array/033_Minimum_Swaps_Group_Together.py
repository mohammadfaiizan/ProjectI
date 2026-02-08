"""
Problem: Minimum Swaps Required to Group Together
URL: https://practice.geeksforgeeks.org/problems/minimum-swaps-required-to-bring-all-elements-less-than-or-equal-to-k-together4847/1

Problem Statement:
Given an array of n positive integers and a number k, find the minimum number of swaps
required to bring all the numbers less than or equal to k together in a contiguous subarray.

Sample Input/Output:
Input: arr = [2, 1, 5, 6, 3], K = 3
Output: 1
Explanation: Swap 5 with 3, resulting [2, 1, 3, 6, 5].

Input: arr = [2, 7, 9, 5, 8, 7, 4], K = 6
Output: 2
"""


class Solution:
    def Min_Swaps_Sliding_Window_Optimal(self, arr, k):
        """
        Sliding Window - Count bad elements in windows of size = count of elements <= k
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(arr)
        count = sum(1 for x in arr if x <= k)
        bad = sum(1 for i in range(count) if arr[i] > k)
        ans = bad
        for i in range(count, n):
            if arr[i - count] > k:
                bad -= 1
            if arr[i] > k:
                bad += 1
            ans = min(ans, bad)
        return ans


def Test_Minimum_Swaps_Group():
    solution = Solution()

    class TestCase:
        def __init__(self, arr, k, expected):
            self.arr = arr
            self.k = k
            self.expected = expected

    test_cases = [
        TestCase([2, 1, 5, 6, 3], 3, 1),
        TestCase([2, 7, 9, 5, 8, 7, 4], 6, 2),
        TestCase([1, 2, 3], 3, 0),
        TestCase([5, 4, 3, 2, 1], 3, 0)
    ]

    for tc in test_cases:
        print(f"Array: {tc.arr}, K={tc.k}, Expected={tc.expected}")

        print("Sliding Window:", solution.Min_Swaps_Sliding_Window_Optimal(tc.arr, tc.k))

        print("-" * 50)


if __name__ == "__main__":
    Test_Minimum_Swaps_Group()
