"""
Problem: Longest Palindromic Substring
URL: https://practice.geeksforgeeks.org/problems/longest-palindrome-in-a-string3411/1

Problem Statement:
Given a string S, find the longest palindromic substring in S.

Sample Input/Output:
Input: S = "aaaabbaa"
Output: "aabbaa"

Input: S = "abc"
Output: "a" (or "b" or "c")
"""


class Solution:
    def Longest_Palindrome_Expand_Center(self, s):
        """
        Expand around center for both odd and even length palindromes
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        n = len(s)
        if n == 0:
            return ""
        start = 0
        max_len = 1

        for i in range(1, n):
            low = i - 1
            high = i
            while low >= 0 and high < n and s[low] == s[high]:
                if high - low + 1 > max_len:
                    start = low
                    max_len = high - low + 1
                low -= 1
                high += 1

            low = i - 1
            high = i + 1
            while low >= 0 and high < n and s[low] == s[high]:
                if high - low + 1 > max_len:
                    start = low
                    max_len = high - low + 1
                low -= 1
                high += 1

        return s[start:start + max_len]

    def Longest_Palindrome_DP(self, s):
        """
        DP - dp[i][j] = true if s[i..j] is palindrome
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        n = len(s)
        if n == 0:
            return ""
        dp = [[False] * n for _ in range(n)]
        start = 0
        max_len = 1

        for i in range(n):
            dp[i][i] = True

        for i in range(n - 1):
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
                start = i
                max_len = 2

        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and dp[i + 1][j - 1]:
                    dp[i][j] = True
                    if length > max_len:
                        start = i
                        max_len = length

        return s[start:start + max_len]


def Test_Longest_Palindromic_Substring():
    sol = Solution()
    tests = ["aaaabbaa", "abc", "babad", "cbbd", "a", "forgeeksskeegfor"]

    for s in tests:
        print(f"Input: {s}")
        print(f"Expand Center: {sol.Longest_Palindrome_Expand_Center(s)}")
        print(f"DP: {sol.Longest_Palindrome_DP(s)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Longest_Palindromic_Substring()
