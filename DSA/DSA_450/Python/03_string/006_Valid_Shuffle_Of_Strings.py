"""
Problem: Check if a String is a Valid Shuffle of Two Other Strings
URL: https://www.programiz.com/java-programming/examples/check-valid-shuffle-of-strings

Problem Statement:
Given three strings s1, s2, and result, check if result is a valid shuffle of s1 and s2.
A valid shuffle maintains the relative order of characters from both strings.

Sample Input/Output:
Input: s1 = "XY", s2 = "12", result = "1XY2"
Output: YES (order of XY and 12 both maintained)

Input: s1 = "XY", s2 = "12", result = "Y12X"
Output: NO (order of XY not maintained)
"""


class Solution:
    def Valid_Shuffle_Greedy(self, s1, s2, result):
        """
        Greedy two-pointer approach
        Time Complexity: O(n) where n = result.size()
        Space Complexity: O(1)
        """
        if len(result) != len(s1) + len(s2):
            return False
        i, j = 0, 0
        for c in result:
            if i < len(s1) and c == s1[i]:
                i += 1
            elif j < len(s2) and c == s2[j]:
                j += 1
            else:
                return False
        return i == len(s1) and j == len(s2)

    def Valid_Shuffle_Recursive(self, s1, s2, result, i, j, k):
        """
        Recursive approach
        Time Complexity: O(2^n) worst case
        Space Complexity: O(n) recursion stack
        """
        if k == len(result):
            return i == len(s1) and j == len(s2)
        take_s1 = False
        take_s2 = False
        if i < len(s1) and s1[i] == result[k]:
            take_s1 = self.Valid_Shuffle_Recursive(s1, s2, result, i + 1, j, k + 1)
        if j < len(s2) and s2[j] == result[k]:
            take_s2 = self.Valid_Shuffle_Recursive(s1, s2, result, i, j + 1, k + 1)
        return take_s1 or take_s2

    def Valid_Shuffle_DP(self, s1, s2, result):
        """
        DP approach
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(s1), len(s2)
        if len(result) != m + n:
            return False
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True

        for i in range(m + 1):
            for j in range(n + 1):
                if i > 0 and s1[i - 1] == result[i + j - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
                if j > 0 and s2[j - 1] == result[i + j - 1]:
                    dp[i][j] = dp[i][j] or dp[i][j - 1]

        return dp[m][n]


def Test_Valid_Shuffle():
    sol = Solution()
    tests = [
        ("XY", "12", "1XY2"),
        ("XY", "12", "Y12X"),
        ("XY", "12", "X1Y2"),
        ("abc", "def", "adbecf"),
        ("abc", "def", "abcdef")
    ]

    for s1, s2, result in tests:
        print(f"s1: {s1}, s2: {s2}, result: {result}")
        print(f"Greedy: {'YES' if sol.Valid_Shuffle_Greedy(s1, s2, result) else 'NO'}")
        print(f"Recursive: {'YES' if sol.Valid_Shuffle_Recursive(s1, s2, result, 0, 0, 0) else 'NO'}")
        print(f"DP: {'YES' if sol.Valid_Shuffle_DP(s1, s2, result) else 'NO'}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Valid_Shuffle()
