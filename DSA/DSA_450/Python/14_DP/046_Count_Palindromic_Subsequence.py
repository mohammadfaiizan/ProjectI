"""
Problem: Count Palindromic Subsequence
URL: https://practice.geeksforgeeks.org/problems/count-palindromic-subsequences/1

Problem Statement:
Given a string str of length N, count the number of palindromic subsequences (not necessarily contiguous) in the string.

Sample Input/Output:
Input: "abcd"
Output: 4
Input: "aab"
Output: 4
"""


class Solution:
    def Count_Pal_Sub_Memo(self, s: str, i: int, j: int, dp: list[list[int]]) -> int:
        """
        Memoization approach
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        if i > j:
            return 0
        if i == j:
            return 1
        if dp[i][j] != -1:
            return dp[i][j]
        
        if s[i] == s[j]:
            dp[i][j] = (self.Count_Pal_Sub_Memo(s, i + 1, j, dp) + 
                       self.Count_Pal_Sub_Memo(s, i, j - 1, dp) + 1)
        else:
            dp[i][j] = (self.Count_Pal_Sub_Memo(s, i + 1, j, dp) + 
                       self.Count_Pal_Sub_Memo(s, i, j - 1, dp) - 
                       self.Count_Pal_Sub_Memo(s, i + 1, j - 1, dp))
        return dp[i][j]
    
    def Count_Pal_Sub_Tab(self, s: str) -> int:
        """
        Tabulation approach
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if i == j:
                    dp[i][j] = 1
                elif s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j] + dp[i][j - 1] + 1
                else:
                    dp[i][j] = dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1]
        
        return dp[0][n - 1]


def Test_CountPalindromicSubsequence():
    solution = Solution()
    
    s1 = "abcd"
    dp1 = [[-1] * len(s1) for _ in range(len(s1))]
    result1 = solution.Count_Pal_Sub_Memo(s1, 0, len(s1) - 1, dp1)
    assert result1 == 4
    assert solution.Count_Pal_Sub_Tab(s1) == 4
    
    s2 = "aab"
    dp2 = [[-1] * len(s2) for _ in range(len(s2))]
    result2 = solution.Count_Pal_Sub_Memo(s2, 0, len(s2) - 1, dp2)
    assert result2 == 4
    assert solution.Count_Pal_Sub_Tab(s2) == 4


if __name__ == "__main__":
    Test_CountPalindromicSubsequence()
