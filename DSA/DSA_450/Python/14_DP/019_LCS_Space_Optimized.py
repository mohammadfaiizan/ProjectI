"""
Problem: LCS Space Optimized
URL: https://www.geeksforgeeks.org/space-optimized-solution-lcs/

Problem Statement:
Find the length of longest common subsequence of two strings using O(min(m,n)) space complexity.

Sample Input/Output:
Input: "AGGTAB", "GXTXAYB"
Output: 4
"""

class Solution:
    def LCS_Space_Two_Row(self, s1, s2):
        """
        Space optimized using two rows
        Time Complexity: O(m*n)
        Space Complexity: O(min(m,n))
        """
        m, n = len(s1), len(s2)
        str1, str2 = s1, s2
        if m < n:
            str1, str2 = str2, str1
            m, n = n, m
        dp = [[0] * (n+1) for _ in range(2)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if str1[i-1] == str2[j-1]:
                    dp[i%2][j] = 1 + dp[(i-1)%2][j-1]
                else:
                    dp[i%2][j] = max(dp[(i-1)%2][j], dp[i%2][j-1])
        return dp[m%2][n]

def Test_LCS_Space():
    solution = Solution()
    s1, s2 = "AGGTAB", "GXTXAYB"
    print("LCS Length:", solution.LCS_Space_Two_Row(s1, s2))

if __name__ == "__main__":
    Test_LCS_Space()
