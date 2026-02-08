"""
Problem: Longest Repeating Subsequence
URL: https://practice.geeksforgeeks.org/problems/longest-repeating-subsequence2004/1

Problem Statement:
Given string str, find the length of the longest repeating subsequence such that
the two subsequences don't use same element at same position.

Sample Input/Output:
Input: str = "axxxy"
Output: 2 (xx)

Input: str = "aab"
Output: 1 (a)
"""


class Solution:
    def LRS_Tabulation(self, s):
        """
        LCS variant - LCS of string with itself where i != j
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        n = len(s)
        dp = [[0] * (n + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if s[i - 1] == s[j - 1] and i != j:
                    dp[i][j] = 1 + dp[i - 1][j - 1]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[n][n]

    def LRS_Space_Optimized(self, s):
        """
        Space optimized using two rows
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        n = len(s)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if s[i - 1] == s[j - 1] and i != j:
                    curr[j] = 1 + prev[j - 1]
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev = curr[:]

        return prev[n]

    def LRS_Memoization(self, s, i, j, memo):
        """
        Top-down memoization
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        if i == 0 or j == 0:
            return 0
        if memo[i][j] != -1:
            return memo[i][j]

        if s[i - 1] == s[j - 1] and i != j:
            memo[i][j] = 1 + self.LRS_Memoization(s, i - 1, j - 1, memo)
        else:
            memo[i][j] = max(self.LRS_Memoization(s, i - 1, j, memo),
                           self.LRS_Memoization(s, i, j - 1, memo))
        return memo[i][j]


def Test_Longest_Repeating_Subsequence():
    sol = Solution()
    tests = ["axxxy", "aab", "aabb", "abc", "aabebcdd"]

    for s in tests:
        print(f"Input: {s}")
        print(f"Tabulation: {sol.LRS_Tabulation(s)}")
        print(f"Space Optimized: {sol.LRS_Space_Optimized(s)}")
        n = len(s)
        memo = [[-1] * (n + 1) for _ in range(n + 1)]
        print(f"Memoization: {sol.LRS_Memoization(s, n, n, memo)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Longest_Repeating_Subsequence()
