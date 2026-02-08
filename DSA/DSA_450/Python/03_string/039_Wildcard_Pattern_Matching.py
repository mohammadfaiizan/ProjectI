"""
Problem: Wildcard Pattern Matching
URL: https://practice.geeksforgeeks.org/problems/wildcard-string-matching1126/1

Problem Statement:
Given a text string and a wildcard pattern, implement wildcard pattern matching
with support for '?' (matches single character) and '*' (matches any sequence
of characters including empty).

Sample Input/Output:
Input: str = "baaabab", pattern = "*****ba*****ab"
Output: true

Input: str = "aa", pattern = "a"
Output: false
"""


class Solution:
    def Wildcard_DP(self, s, p):
        """
        Bottom-up DP
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True

        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 1]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    dp[i][j] = dp[i - 1][j] or dp[i][j - 1]
                elif p[j - 1] == '?' or s[i - 1] == p[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]

        return dp[m][n]

    def Wildcard_Memoization(self, s, p, i, j, memo):
        """
        Top-down memoization
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        if i < 0 and j < 0:
            return True
        if j < 0:
            return False
        if i < 0:
            while j >= 0:
                if p[j] != '*':
                    return False
                j -= 1
            return True

        if memo[i][j] != -1:
            return memo[i][j]

        if p[j] == '*':
            memo[i][j] = (self.Wildcard_Memoization(s, p, i - 1, j, memo) or
                          self.Wildcard_Memoization(s, p, i, j - 1, memo))
        elif p[j] == '?' or s[i] == p[j]:
            memo[i][j] = self.Wildcard_Memoization(s, p, i - 1, j - 1, memo)
        else:
            memo[i][j] = False

        return memo[i][j]

    def Wildcard_Two_Pointer(self, s, p):
        """
        Two pointer / greedy approach
        Time Complexity: O(m * n) worst case, O(m + n) average
        Space Complexity: O(1)
        """
        si = pi = 0
        starIdx = matchIdx = -1
        m, n = len(s), len(p)

        while si < m:
            if pi < n and (p[pi] == '?' or p[pi] == s[si]):
                si += 1
                pi += 1
            elif pi < n and p[pi] == '*':
                starIdx = pi
                matchIdx = si
                pi += 1
            elif starIdx != -1:
                pi = starIdx + 1
                matchIdx += 1
                si = matchIdx
            else:
                return False

        while pi < n and p[pi] == '*':
            pi += 1

        return pi == n


def Test_Wildcard_Pattern_Matching():
    sol = Solution()
    tests = [
        ("baaabab", "*****ba*****ab"),
        ("aa", "a"),
        ("aa", "*"),
        ("cb", "?a"),
        ("adceb", "*a*b"),
        ("acdcb", "a*c?b"),
        ("", "*")
    ]

    for s, p in tests:
        print(f'str: "{s}", pattern: "{p}"')
        print(f"DP: {sol.Wildcard_DP(s, p)}")
        m, n = len(s), len(p)
        if m > 0 and n > 0:
            memo = [[-1] * n for _ in range(m)]
            print(f"Memoization: {sol.Wildcard_Memoization(s, p, m - 1, n - 1, memo)}")
        else:
            print(f"Memoization: {sol.Wildcard_DP(s, p)}")
        print(f"Two Pointer: {sol.Wildcard_Two_Pointer(s, p)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Wildcard_Pattern_Matching()
