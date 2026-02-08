"""
Problem: Longest Common Subsequence
URL: https://practice.geeksforgeeks.org/problems/longest-common-subsequence-1587115620/1

Problem Statement:
Given two strings s1 and s2, find the length of the longest common subsequence.
A subsequence is a sequence that appears in the same relative order but not
necessarily contiguous.

Sample Input/Output:
Input: s1 = "ABCDGH", s2 = "AEDFHR"
Output: 3 (ADH)

Input: s1 = "ABC", s2 = "AC"
Output: 2 (AC)
"""


class Solution:
    def LCS_Tabulation(self, s1, s2):
        """
        Bottom-up DP
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = 1 + dp[i - 1][j - 1]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def LCS_Space_Optimized(self, s1, s2):
        """
        Space optimized using two rows
        Time Complexity: O(m * n)
        Space Complexity: O(n)
        """
        m, n = len(s1), len(s2)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    curr[j] = 1 + prev[j - 1]
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev = curr[:]
            curr = [0] * (n + 1)

        return prev[n]

    def LCS_Memoization(self, s1, s2, i, j, memo):
        """
        Top-down memoization
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        if i == 0 or j == 0:
            return 0
        if memo[i][j] != -1:
            return memo[i][j]

        if s1[i - 1] == s2[j - 1]:
            memo[i][j] = 1 + self.LCS_Memoization(s1, s2, i - 1, j - 1, memo)
        else:
            memo[i][j] = max(self.LCS_Memoization(s1, s2, i - 1, j, memo),
                           self.LCS_Memoization(s1, s2, i, j - 1, memo))
        return memo[i][j]

    def Print_LCS(self, s1, s2):
        """
        Print the actual LCS string
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = 1 + dp[i - 1][j - 1]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs = ""
        i, j = m, n
        while i > 0 and j > 0:
            if s1[i - 1] == s2[j - 1]:
                lcs = s1[i - 1] + lcs
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1

        return lcs


def Test_Longest_Common_Subsequence():
    sol = Solution()
    tests = [
        ("ABCDGH", "AEDFHR"),
        ("ABC", "AC"),
        ("AGGTAB", "GXTXAYB"),
        ("abc", "def")
    ]

    for s1, s2 in tests:
        print(f"s1: {s1}, s2: {s2}")
        print(f"Tabulation: {sol.LCS_Tabulation(s1, s2)}")
        print(f"Space Optimized: {sol.LCS_Space_Optimized(s1, s2)}")
        m, n = len(s1), len(s2)
        memo = [[-1] * (n + 1) for _ in range(m + 1)]
        print(f"Memoization: {sol.LCS_Memoization(s1, s2, m, n, memo)}")
        print(f"LCS String: {sol.Print_LCS(s1, s2)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Longest_Common_Subsequence()
