"""
Problem: Longest Palindromic Subsequence
URL: https://leetcode.com/problems/longest-palindromic-subsequence/

Problem Statement:
Given a string s, find the longest palindromic subsequence's length in s. A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.

Sample Input/Output:
Input: "bbbab"
Output: 4
Input: "cbbd"
Output: 2
"""

class Solution:
    def Longest_Palindromic_Subsequence_LPS_DP(self, s, n):
        """
        Direct DP approach
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        for length in range(2, n+1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j]:
                    dp[i][j] = 2 + (0 if length == 2 else dp[i+1][j-1])
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])
        return dp[0][n-1]

    def Longest_Palindromic_Subsequence_LPS_Via_LCS(self, s, n):
        """
        LPS via LCS approach
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        rev = s[::-1]
        return self.LCS_Helper(s, rev, n, n)

    def LCS_Helper(self, s1, s2, m, n):
        dp = [[0] * (n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

def Test_Longest_Palindromic_Subsequence():
    solution = Solution()
    s1 = "bbbab"
    s2 = "cbbd"
    
    print("Test 1 (bbbab) - LPS DP:", solution.Longest_Palindromic_Subsequence_LPS_DP(s1, len(s1)))
    print("Test 1 (bbbab) - Via LCS:", solution.Longest_Palindromic_Subsequence_LPS_Via_LCS(s1, len(s1)))
    print("Test 2 (cbbd) - LPS DP:", solution.Longest_Palindromic_Subsequence_LPS_DP(s2, len(s2)))
    print("Test 2 (cbbd) - Via LCS:", solution.Longest_Palindromic_Subsequence_LPS_Via_LCS(s2, len(s2)))

if __name__ == "__main__":
    Test_Longest_Palindromic_Subsequence()
