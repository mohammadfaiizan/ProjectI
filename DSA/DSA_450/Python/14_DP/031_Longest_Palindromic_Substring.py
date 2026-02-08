"""
Problem: Longest Palindromic Substring
URL: https://leetcode.com/problems/longest-palindromic-substring/

Problem Statement:
Given a string s, return the longest palindromic substring in s.

Sample Input/Output:
Input: "babad"
Output: "bab"
Input: "cbbd"
Output: "bb"
"""


class Solution:
    def LPS_Expand(self, s: str) -> str:
        """
        Expand Around Centers
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        n = len(s)
        if n == 0:
            return ""
        
        start = 0
        max_len = 1
        
        for i in range(n):
            len1 = self._expand_around_center(s, i, i)
            len2 = self._expand_around_center(s, i, i + 1)
            length = max(len1, len2)
            
            if length > max_len:
                max_len = length
                start = i - (length - 1) // 2
        
        return s[start:start + max_len]
    
    def LPS_DP(self, s: str) -> str:
        """
        Dynamic Programming
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        n = len(s)
        if n == 0:
            return ""
        
        dp = [[False] * n for _ in range(n)]
        start = 0
        max_len = 1
        
        for i in range(n):
            dp[i][i] = True
        
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
                start = i
                max_len = 2
        
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and dp[i + 1][j - 1]:
                    dp[i][j] = True
                    start = i
                    max_len = length
        
        return s[start:start + max_len]
    
    def _expand_around_center(self, s: str, left: int, right: int) -> int:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1


def Test_LongestPalindromicSubstring():
    solution = Solution()
    result1 = solution.LPS_Expand("babad")
    assert result1 == "bab" or result1 == "aba"
    assert solution.LPS_Expand("cbbd") == "bb"
    
    result2 = solution.LPS_DP("babad")
    assert result2 == "bab" or result2 == "aba"
    assert solution.LPS_DP("cbbd") == "bb"


if __name__ == "__main__":
    Test_LongestPalindromicSubstring()
