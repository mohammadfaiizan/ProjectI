"""
Problem: Longest Repeated Subsequence
URL: https://practice.geeksforgeeks.org/problems/longest-repeating-subsequence2004/1

Problem Statement:
Given string str, find the length of the longest repeating subsequence such that it can be found twice in the given string. The two identified subsequences A and B can use the same characters from the string but the positions of the characters in A and B must be different.

Sample Input/Output:
Input: "axxzxy"
Output: 2
Explanation: The longest repeating subsequence is "xx" or "xy"
"""

class Solution:
    def Longest_Repeated_Subsequence_LRS_Tabulation(self, str_val, n):
        """
        LRS Tabulation approach
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        dp = [[0] * (n+1) for _ in range(n+1)]
        for i in range(1, n+1):
            for j in range(1, n+1):
                if str_val[i-1] == str_val[j-1] and i != j:
                    dp[i][j] = 1 + dp[i-1][j-1]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[n][n]

def Test_Longest_Repeated_Subsequence():
    solution = Solution()
    str_val = "axxzxy"
    
    print("LRS Tabulation:", solution.Longest_Repeated_Subsequence_LRS_Tabulation(str_val, len(str_val)))

if __name__ == "__main__":
    Test_Longest_Repeated_Subsequence()
