"""
Problem: Longest Common Subsequence
URL: https://leetcode.com/problems/longest-common-subsequence/

Problem Statement:
Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.
A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

Sample Input/Output:
Input: text1 = "abcde", text2 = "ace" 
Output: 3  
Explanation: The longest common subsequence is "ace" and its length is 3.

Input: text1 = "abc", text2 = "abc"
Output: 3
Explanation: The longest common subsequence is "abc" and its length is 3.

Input: text1 = "abc", text2 = "def"
Output: 0
Explanation: There is no such common subsequence, so the result is 0.
"""

from typing import List

class Solution:
    def Longest_Common_Subsequence_Recursive(self, text1: str, text2: str) -> int:
        """
        Recursive Brute Force - Try all possibilities
        Time Complexity: O(2^(m+n))
        Space Complexity: O(m+n)
        """
        def LCS_Helper(i: int, j: int) -> int:
            if i >= len(text1) or j >= len(text2):
                return 0
            
            if text1[i] == text2[j]:
                return 1 + LCS_Helper(i + 1, j + 1)
            else:
                return max(LCS_Helper(i + 1, j), LCS_Helper(i, j + 1))
        
        return LCS_Helper(0, 0)
    
    def Longest_Common_Subsequence_Memoized(self, text1: str, text2: str) -> int:
        """
        Memoized DP - Top-down with caching
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        memo = {}
        
        def LCS_Helper(i: int, j: int) -> int:
            if i >= len(text1) or j >= len(text2):
                return 0
            
            if (i, j) in memo:
                return memo[(i, j)]
            
            if text1[i] == text2[j]:
                result = 1 + LCS_Helper(i + 1, j + 1)
            else:
                result = max(LCS_Helper(i + 1, j), LCS_Helper(i, j + 1))
            
            memo[(i, j)] = result
            return result
        
        return LCS_Helper(0, 0)
    
    def Longest_Common_Subsequence_Tabulation_Optimal(self, text1: str, text2: str) -> int:
        """
        Tabulation DP Optimal - Bottom-up approach
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(text1), len(text2)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def Longest_Common_Subsequence_Space_Optimized(self, text1: str, text2: str) -> int:
        """
        Space Optimized DP - Use 1D array
        Time Complexity: O(m * n)
        Space Complexity: O(min(m, n))
        """
        if len(text1) < len(text2):
            text1, text2 = text2, text1
        
        m, n = len(text1), len(text2)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            
            prev, curr = curr, prev
        
        return prev[n]
    
    def Longest_Common_Subsequence_With_String(self, text1: str, text2: str) -> tuple:
        """
        With String Construction - Return length and actual LCS
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(text1), len(text2)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs = []
        i, j = m, n
        
        while i > 0 and j > 0:
            if text1[i-1] == text2[j-1]:
                lcs.append(text1[i-1])
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1
        
        return dp[m][n], ''.join(reversed(lcs))
    
    def Longest_Common_Subsequence_All_LCS(self, text1: str, text2: str) -> tuple:
        """
        All LCS - Find all longest common subsequences
        Time Complexity: O(m * n * 2^(m+n))
        Space Complexity: O(m * n * 2^(m+n))
        """
        m, n = len(text1), len(text2)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        def Find_All_LCS(i: int, j: int, current_lcs: str, all_lcs: set) -> None:
            if i == 0 or j == 0:
                all_lcs.add(current_lcs[::-1])
                return
            
            if text1[i-1] == text2[j-1]:
                Find_All_LCS(i-1, j-1, current_lcs + text1[i-1], all_lcs)
            else:
                if dp[i-1][j] == dp[i][j]:
                    Find_All_LCS(i-1, j, current_lcs, all_lcs)
                if dp[i][j-1] == dp[i][j]:
                    Find_All_LCS(i, j-1, current_lcs, all_lcs)
        
        all_lcs = set()
        if dp[m][n] > 0 and len(text1) <= 10 and len(text2) <= 10:
            Find_All_LCS(m, n, "", all_lcs)
        
        return dp[m][n], list(all_lcs)

def Test_Longest_Common_Subsequence():
    solution = Solution()
    
    test_cases = [
        ("abcde", "ace", 3),
        ("abc", "abc", 3),
        ("abc", "def", 0),
        ("ABCDGH", "AEDFHR", 3),
        ("AGGTAB", "GXTXAYB", 4)
    ]
    
    methods = [
        ("Recursive", solution.Longest_Common_Subsequence_Recursive),
        ("Memoized", solution.Longest_Common_Subsequence_Memoized),
        ("Tabulation Optimal", solution.Longest_Common_Subsequence_Tabulation_Optimal),
        ("Space Optimized", solution.Longest_Common_Subsequence_Space_Optimized)
    ]
    
    for text1, text2, expected in test_cases:
        print(f"Text1: '{text1}', Text2: '{text2}'")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            try:
                if method_name == "Recursive" and (len(text1) > 10 or len(text2) > 10):
                    print(f"{method_name}: Skipped (too slow)")
                    continue
                
                result = method(text1, text2)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        length, lcs_string = solution.Longest_Common_Subsequence_With_String(text1, text2)
        print(f"With String: Length={length}, LCS='{lcs_string}'")
        
        if len(text1) <= 8 and len(text2) <= 8:
            length, all_lcs = solution.Longest_Common_Subsequence_All_LCS(text1, text2)
            print(f"All LCS: Length={length}, Count={len(all_lcs)}")
            for lcs in all_lcs:
                print(f"  '{lcs}'")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Longest_Common_Subsequence()
