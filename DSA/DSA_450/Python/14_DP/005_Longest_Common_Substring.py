"""
Problem: Longest Common Substring
URL: https://practice.geeksforgeeks.org/problems/longest-common-substring1452/1

Problem Statement:
Given two strings X and Y. The task is to find the length of the longest common substring.

Sample Input/Output:
Input: "ABCDGH", "ACDGHR"
Output: 4
Explanation: Longest common substring is "CDGH"
"""

class Solution:
    def Longest_Common_Substring_DP_Tabulation(self, s1, s2, m, n):
        """
        DP Tabulation approach
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        dp = [[0] * (n+1) for _ in range(m+1)]
        result = 0
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                    result = max(result, dp[i][j])
                else:
                    dp[i][j] = 0
        return result

    def Longest_Common_Substring_Space_Optimized(self, s1, s2, m, n):
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
        result = 0
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    curr[j] = 1 + prev[j-1]
                    result = max(result, curr[j])
                else:
                    curr[j] = 0
            prev = curr[:]
        return result

def Test_Longest_Common_Substring():
    solution = Solution()
    s1 = "ABCDGH"
    s2 = "ACDGHR"
    
    print("DP Tabulation:", solution.Longest_Common_Substring_DP_Tabulation(s1, s2, len(s1), len(s2)))
    print("Space Optimized:", solution.Longest_Common_Substring_Space_Optimized(s1, s2, len(s1), len(s2)))

if __name__ == "__main__":
    Test_Longest_Common_Substring()
