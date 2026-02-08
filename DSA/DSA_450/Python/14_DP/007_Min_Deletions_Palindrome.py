"""
Problem: Minimum Deletions to Make Palindrome
URL: https://practice.geeksforgeeks.org/problems/minimum-number-of-deletions4610/1

Problem Statement:
Given a string of S as input. Your task is to write a program to remove or delete minimum number of characters from the string so that the resultant string is palindrome.

Sample Input/Output:
Input: "aebcbda"
Output: 2
Explanation: Remove characters 'e' and 'd', result: "abcba"
"""

class Solution:
    def Min_Deletions_Min_Del_Via_LPS(self, s, n):
        """
        Min deletions via LPS approach
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        lps = self.Longest_Palindromic_Subsequence(s, n)
        return n - lps

    def Longest_Palindromic_Subsequence(self, s, n):
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

    def Min_Deletions_Min_Del_Direct_DP(self, s, n):
        """
        Direct DP approach
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 0
        for length in range(2, n+1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i+1][j], dp[i][j-1])
        return dp[0][n-1]

def Test_Min_Deletions_Palindrome():
    solution = Solution()
    s = "aebcbda"
    
    print("Via LPS:", solution.Min_Deletions_Min_Del_Via_LPS(s, len(s)))
    print("Direct DP:", solution.Min_Deletions_Min_Del_Direct_DP(s, len(s)))

if __name__ == "__main__":
    Test_Min_Deletions_Palindrome()
