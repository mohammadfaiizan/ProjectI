"""
Problem: Count All Palindromic Subsequences
URL: https://practice.geeksforgeeks.org/problems/count-palindromic-subsequences/1

Problem Statement:
Given a string str of length N, find the number of palindromic subsequences
of length greater than or equal to 1.

Sample Input/Output:
Input: str = "abcd"
Output: 4 (a, b, c, d)

Input: str = "aab"
Output: 4 (a, a, b, aa)
"""


class Solution:
    def Count_Palindromic_Subseq_DP(self, s):
        """
        DP approach - dp[i][j] = count of palindromic subsequences in s[i..j]
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        n = len(s)
        dp = [[0] * n for _ in range(n)]

        for i in range(n):
            dp[i][i] = 1

        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j] + dp[i][j - 1] + 1
                else:
                    dp[i][j] = dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1]

        return dp[0][n - 1]

    def Count_Palindromic_Subseq_Recursive(self, s, i, j, memo):
        """
        Top-down memoization
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        if i > j:
            return 0
        if i == j:
            return 1
        if memo[i][j] != -1:
            return memo[i][j]

        if s[i] == s[j]:
            memo[i][j] = (self.Count_Palindromic_Subseq_Recursive(s, i + 1, j, memo) +
                          self.Count_Palindromic_Subseq_Recursive(s, i, j - 1, memo) + 1)
        else:
            memo[i][j] = (self.Count_Palindromic_Subseq_Recursive(s, i + 1, j, memo) +
                          self.Count_Palindromic_Subseq_Recursive(s, i, j - 1, memo) -
                          self.Count_Palindromic_Subseq_Recursive(s, i + 1, j - 1, memo))
        return memo[i][j]

    def Count_Palindromic_Substrings(self, s):
        """
        Count palindromic substrings (not subsequences) using expand around center
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        n = len(s)
        count = 0
        for center in range(n):
            lo = hi = center
            while lo >= 0 and hi < n and s[lo] == s[hi]:
                count += 1
                lo -= 1
                hi += 1

            lo = center
            hi = center + 1
            while lo >= 0 and hi < n and s[lo] == s[hi]:
                count += 1
                lo -= 1
                hi += 1

        return count


def Test_Count_Palindromic_Subsequences():
    sol = Solution()
    tests = ["abcd", "aab", "aaaa", "abcb", "a"]

    for s in tests:
        print(f"Input: {s}")
        print(f"DP: {sol.Count_Palindromic_Subseq_DP(s)}")
        n = len(s)
        memo = [[-1] * n for _ in range(n)]
        print(f"Recursive: {sol.Count_Palindromic_Subseq_Recursive(s, 0, n - 1, memo)}")
        print(f"Substrings: {sol.Count_Palindromic_Substrings(s)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Count_Palindromic_Subsequences()
