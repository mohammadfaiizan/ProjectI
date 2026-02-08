"""
Problem: Longest Common Subsequence
URL: https://practice.geeksforgeeks.org/problems/longest-common-subsequence-1587115620/1

Problem Statement:
Given two strings, find the length of longest subsequence present in both of them. A subsequence is a sequence that appears in the same relative order, but not necessarily contiguous.

Sample Input/Output:
Input: "ABCBDAB", "BDCAB"
Output: 4
Explanation: LCS is "BCAB" or "BDAB"
"""

class Solution:
    def LCS_Recursive(self, s1, s2, m, n):
        """
        Recursive approach
        Time Complexity: O(2^(m+n))
        Space Complexity: O(m+n)
        """
        if m == 0 or n == 0:
            return 0
        if s1[m-1] == s2[n-1]:
            return 1 + self.LCS_Recursive(s1, s2, m-1, n-1)
        return max(self.LCS_Recursive(s1, s2, m-1, n), self.LCS_Recursive(s1, s2, m, n-1))

    def LCS_Memoization(self, s1, s2, m, n):
        """
        Memoization approach
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        memo = [[-1] * (n+1) for _ in range(m+1)]
        return self.LCS_Memo_Helper(s1, s2, m, n, memo)

    def LCS_Memo_Helper(self, s1, s2, m, n, memo):
        if m == 0 or n == 0:
            return 0
        if memo[m][n] != -1:
            return memo[m][n]
        if s1[m-1] == s2[n-1]:
            memo[m][n] = 1 + self.LCS_Memo_Helper(s1, s2, m-1, n-1, memo)
        else:
            memo[m][n] = max(self.LCS_Memo_Helper(s1, s2, m-1, n, memo),
                            self.LCS_Memo_Helper(s1, s2, m, n-1, memo))
        return memo[m][n]

    def LCS_Tabulation(self, s1, s2, m, n):
        """
        Tabulation approach
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        dp = [[0] * (n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    def LCS_Space_Optimized(self, s1, s2, m, n):
        """
        Space optimized approach
        Time Complexity: O(m*n)
        Space Complexity: O(min(m,n))
        """
        if m < n:
            s1, s2 = s2, s1
            m, n = n, m
        prev = [0] * (n+1)
        curr = [0] * (n+1)
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    curr[j] = 1 + prev[j-1]
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev = curr[:]
        return curr[n]

def Test_Longest_Common_Subsequence():
    solution = Solution()
    s1 = "ABCBDAB"
    s2 = "BDCAB"
    
    print("Recursive:", solution.LCS_Recursive(s1, s2, len(s1), len(s2)))
    print("Memoization:", solution.LCS_Memoization(s1, s2, len(s1), len(s2)))
    print("Tabulation:", solution.LCS_Tabulation(s1, s2, len(s1), len(s2)))
    print("Space Optimized:", solution.LCS_Space_Optimized(s1, s2, len(s1), len(s2)))

if __name__ == "__main__":
    Test_Longest_Common_Subsequence()
