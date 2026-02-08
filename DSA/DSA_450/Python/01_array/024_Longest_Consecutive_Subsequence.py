"""
Problem: Longest Consecutive Subsequence
URL: https://practice.geeksforgeeks.org/problems/longest-consecutive-subsequence2449/1

Problem Statement:
Given an array of positive integers, find the length of the longest sub-sequence such that
elements are consecutive integers (can be in any order).

Sample Input/Output:
Input: arr = [2, 6, 1, 9, 4, 5, 3]
Output: 6
Explanation: The consecutive subsequence is [1, 2, 3, 4, 5, 6].

Input: arr = [1, 9, 3, 10, 4, 20, 2]
Output: 4
Explanation: The consecutive subsequence is [1, 2, 3, 4].
"""


class Solution:
    def Longest_Consecutive_Hashing_Optimal(self, arr):
        """
        HashSet Approach - Check sequence start and count forward
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        s = set(arr)
        longest = 0
        for x in s:
            if x - 1 not in s:
                current = x
                count = 1
                while current + 1 in s:
                    current += 1
                    count += 1
                longest = max(longest, count)
        return longest

    def Longest_Consecutive_Sorting(self, arr):
        """
        Sorting Approach - Sort and find longest consecutive run
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        if not arr:
            return 0
        arr = sorted(set(arr))
        longest = 1
        current = 1
        for i in range(1, len(arr)):
            if arr[i] == arr[i - 1] + 1:
                current += 1
                longest = max(longest, current)
            else:
                current = 1
        return longest


def Test_Longest_Consecutive_Subsequence():
    solution = Solution()

    test_cases = [
        ([2, 6, 1, 9, 4, 5, 3], 6),
        ([1, 9, 3, 10, 4, 20, 2], 4),
        ([100, 4, 200, 1, 3, 2], 4),
        ([0, 3, 7, 2, 5, 8, 4, 6, 0, 1], 9)
    ]

    for arr, expected in test_cases:
        print(f"Array: {arr}, Expected: {expected}")
        result_hashing = solution.Longest_Consecutive_Hashing_Optimal(arr)
        result_sorting = solution.Longest_Consecutive_Sorting(arr)
        print(f"Hashing: {result_hashing}")
        print(f"Sorting: {result_sorting}")
        print("-" * 50)


if __name__ == "__main__":
    Test_Longest_Consecutive_Subsequence()
