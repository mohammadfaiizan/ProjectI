"""
Problem: Word Wrap
URL: https://practice.geeksforgeeks.org/problems/word-wrap1646/1

Problem Statement:
Given an array nums[] of size n, where nums[i] represents the number of characters
in the ith word. Let K be the limit on the number of characters that can be put in
one line (including spaces). Put line breaks in the given sequence such that the
lines are printed neatly. The cost of a line = (Number of extra spaces)^2.
Find the minimum total cost.

Sample Input/Output:
Input: nums = [3, 2, 2, 5], K = 6
Output: 10

Input: nums = [3, 2, 2], K = 4
Output: 5
"""

import sys


class Solution:
    def Word_Wrap_DP(self, nums, k):
        """
        DP approach - minimize total cost of extra spaces squared
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        n = len(nums)
        dp = [sys.maxsize] * (n + 1)
        dp[n] = 0

        for i in range(n - 1, -1, -1):
            length = 0
            for j in range(i, n):
                length += nums[j]
                if length > k:
                    break
                extra = k - length
                cost = 0 if j == n - 1 else extra * extra
                if dp[j + 1] != sys.maxsize:
                    dp[i] = min(dp[i], cost + dp[j + 1])
                length += 1

        return dp[0]

    def Word_Wrap_Memoization(self, nums, k, i, memo):
        """
        Top-down memoization
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        n = len(nums)
        if i >= n:
            return 0
        if memo[i] != -1:
            return memo[i]

        length = 0
        memo[i] = sys.maxsize
        for j in range(i, n):
            length += nums[j]
            if length > k:
                break
            extra = k - length
            cost = 0 if j == n - 1 else extra * extra
            sub = self.Word_Wrap_Memoization(nums, k, j + 1, memo)
            if sub != sys.maxsize:
                memo[i] = min(memo[i], cost + sub)
            length += 1

        return memo[i]

    def Word_Wrap_Greedy(self, nums, k):
        """
        Greedy - fill as many words per line as possible (not optimal but simple)
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(nums)
        total_cost = 0
        i = 0
        while i < n:
            length = nums[i]
            j = i + 1
            while j < n and length + 1 + nums[j] <= k:
                length += 1 + nums[j]
                j += 1
            if j < n:
                extra = k - length
                total_cost += extra * extra
            i = j

        return total_cost


def Test_Word_Wrap():
    sol = Solution()
    tests = [
        ([3, 2, 2, 5], 6),
        ([3, 2, 2], 4),
        ([1, 1, 1, 1, 1], 5)
    ]

    for nums, k in tests:
        print(f"Words: {nums} K={k}")

        print(f"DP: {sol.Word_Wrap_DP(nums, k)}")
        memo = [-1] * len(nums)
        print(f"Memoization: {sol.Word_Wrap_Memoization(nums, k, 0, memo)}")
        print(f"Greedy: {sol.Word_Wrap_Greedy(nums, k)}")

        print('-' * 50)


if __name__ == "__main__":
    Test_Word_Wrap()
