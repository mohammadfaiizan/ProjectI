"""
Problem: Edit Distance
URL: https://practice.geeksforgeeks.org/problems/edit-distance3702/1

Problem Statement:
Given two strings s and t of lengths m and n respectively, find the edit distance
between them. Edit Distance is defined as the minimum number of operations required
to convert string s to string t. Operations: Insert, Remove, Replace.

Sample Input/Output:
Input: s = "geek", t = "gesek"
Output: 1 (insert 's')

Input: s = "horse", t = "ros"
Output: 3
"""


class Solution:
    def Edit_Distance_Tabulation(self, s, t):
        """
        Bottom-up DP
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(s), len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return dp[m][n]

    def Edit_Distance_Space_Optimized(self, s, t):
        """
        Space optimized using two rows
        Time Complexity: O(m * n)
        Space Complexity: O(n)
        """
        m, n = len(s), len(t)
        prev = list(range(n + 1))
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            curr[0] = i
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    curr[j] = prev[j - 1]
                else:
                    curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
            prev = curr[:]

        return prev[n]

    def Edit_Distance_Recursive(self, s, t, i, j, memo):
        """
        Top-down memoization
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        if i == 0:
            return j
        if j == 0:
            return i
        if memo[i][j] != -1:
            return memo[i][j]

        if s[i - 1] == t[j - 1]:
            memo[i][j] = self.Edit_Distance_Recursive(s, t, i - 1, j - 1, memo)
        else:
            memo[i][j] = 1 + min(
                self.Edit_Distance_Recursive(s, t, i - 1, j, memo),
                self.Edit_Distance_Recursive(s, t, i, j - 1, memo),
                self.Edit_Distance_Recursive(s, t, i - 1, j - 1, memo)
            )
        return memo[i][j]


def Test_Edit_Distance():
    sol = Solution()
    tests = [
        ("geek", "gesek"),
        ("horse", "ros"),
        ("intention", "execution"),
        ("abc", "abc"),
        ("", "abc")
    ]

    for s, t in tests:
        print(f's: "{s}", t: "{t}"')
        print(f"Tabulation: {sol.Edit_Distance_Tabulation(s, t)}")
        print(f"Space Optimized: {sol.Edit_Distance_Space_Optimized(s, t)}")
        m, n = len(s), len(t)
        memo = [[-1] * (n + 1) for _ in range(m + 1)]
        print(f"Memoization: {sol.Edit_Distance_Recursive(s, t, m, n, memo)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Edit_Distance()
