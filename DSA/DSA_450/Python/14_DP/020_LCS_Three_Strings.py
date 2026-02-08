"""
Problem: LCS of Three Strings
URL: https://practice.geeksforgeeks.org/problems/lcs-of-three-strings0028/1

Problem Statement:
Given 3 strings A, B and C, the task is to find the length of the longest sub-sequence that is common in all the three given strings.

Sample Input/Output:
Input: A = "geeks", B = "geeksfor", C = "geeksforgeeks"
Output: 5
"""

class Solution:
    def LCS_Three_Tab(self, A, B, C):
        """
        Tabulation approach
        Time Complexity: O(l*m*n)
        Space Complexity: O(l*m*n)
        """
        l, m, n = len(A), len(B), len(C)
        dp = [[[0] * (n+1) for _ in range(m+1)] for _ in range(l+1)]
        for i in range(1, l+1):
            for j in range(1, m+1):
                for k in range(1, n+1):
                    if A[i-1] == B[j-1] == C[k-1]:
                        dp[i][j][k] = 1 + dp[i-1][j-1][k-1]
                    else:
                        dp[i][j][k] = max(dp[i-1][j][k], dp[i][j-1][k], dp[i][j][k-1])
        return dp[l][m][n]

    def LCS_Three_Memo(self, A, B, C):
        """
        Memoization approach
        Time Complexity: O(l*m*n)
        Space Complexity: O(l*m*n)
        """
        l, m, n = len(A), len(B), len(C)
        dp = [[[-1] * (n+1) for _ in range(m+1)] for _ in range(l+1)]
        return self.LCS_Three_Memo_Helper(A, B, C, l, m, n, dp)

    def LCS_Three_Memo_Helper(self, A, B, C, i, j, k, dp):
        if i == 0 or j == 0 or k == 0:
            return 0
        if dp[i][j][k] != -1:
            return dp[i][j][k]
        if A[i-1] == B[j-1] == C[k-1]:
            dp[i][j][k] = 1 + self.LCS_Three_Memo_Helper(A, B, C, i-1, j-1, k-1, dp)
        else:
            dp[i][j][k] = max(self.LCS_Three_Memo_Helper(A, B, C, i-1, j, k, dp),
                              self.LCS_Three_Memo_Helper(A, B, C, i, j-1, k, dp),
                              self.LCS_Three_Memo_Helper(A, B, C, i, j, k-1, dp))
        return dp[i][j][k]

def Test_LCS_Three():
    solution = Solution()
    A, B, C = "geeks", "geeksfor", "geeksforgeeks"
    print("Tabulation:", solution.LCS_Three_Tab(A, B, C))
    print("Memoization:", solution.LCS_Three_Memo(A, B, C))

if __name__ == "__main__":
    Test_LCS_Three()
