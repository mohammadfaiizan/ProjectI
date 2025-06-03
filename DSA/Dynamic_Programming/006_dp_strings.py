"""
Dynamic Programming - String Patterns
This module implements various DP problems on strings including LCS, LCSubstring,
edit distance, palindrome problems, and string manipulation with detailed analysis.
"""

from typing import List, Dict, Tuple, Optional
import time

# ==================== LONGEST COMMON SUBSEQUENCE (LCS) ====================

class LongestCommonSubsequence:
    """
    Longest Common Subsequence Problems
    
    LCS is a classic DP problem that finds the longest subsequence
    common to two sequences. A subsequence maintains relative order
    but doesn't need to be contiguous.
    """
    
    def lcs_length(self, text1: str, text2: str) -> int:
        """
        Find length of LCS between two strings
        
        LeetCode 1143 - Longest Common Subsequence
        
        Time Complexity: O(m * n)
        Space Complexity: O(min(m, n))
        
        Args:
            text1: First string
            text2: Second string
        
        Returns:
            Length of longest common subsequence
        """
        m, n = len(text1), len(text2)
        
        # Optimize space by using shorter string for columns
        if m < n:
            text1, text2 = text2, text1
            m, n = n, m
        
        # dp[j] represents LCS length for text1[0:i] and text2[0:j]
        dp = [0] * (n + 1)
        
        for i in range(1, m + 1):
            prev = 0  # dp[i-1][j-1]
            for j in range(1, n + 1):
                temp = dp[j]
                if text1[i - 1] == text2[j - 1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = temp
        
        return dp[n]
    
    def lcs_string(self, text1: str, text2: str) -> str:
        """
        Find actual LCS string between two strings
        
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        
        Args:
            text1: First string
            text2: Second string
        
        Returns:
            Longest common subsequence as string
        """
        m, n = len(text1), len(text2)
        
        # Create 2D DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        # Backtrack to find LCS string
        lcs = []
        i, j = m, n
        
        while i > 0 and j > 0:
            if text1[i - 1] == text2[j - 1]:
                lcs.append(text1[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        
        return ''.join(reversed(lcs))
    
    def lcs_three_strings(self, text1: str, text2: str, text3: str) -> int:
        """
        Find LCS of three strings
        
        Time Complexity: O(m * n * p)
        Space Complexity: O(m * n * p)
        
        Args:
            text1, text2, text3: Three input strings
        
        Returns:
            Length of LCS of all three strings
        """
        m, n, p = len(text1), len(text2), len(text3)
        
        # 3D DP table
        dp = [[[0 for _ in range(p + 1)] for _ in range(n + 1)] for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                for k in range(1, p + 1):
                    if text1[i - 1] == text2[j - 1] == text3[k - 1]:
                        dp[i][j][k] = dp[i - 1][j - 1][k - 1] + 1
                    else:
                        dp[i][j][k] = max(
                            dp[i - 1][j][k],
                            dp[i][j - 1][k],
                            dp[i][j][k - 1]
                        )
        
        return dp[m][n][p]
    
    def lcs_with_k_differences(self, text1: str, text2: str, k: int) -> int:
        """
        LCS allowing at most k character differences
        
        Args:
            text1, text2: Input strings
            k: Maximum allowed differences
        
        Returns:
            Length of LCS with at most k differences
        """
        m, n = len(text1), len(text2)
        
        # dp[i][j][d] = LCS length for text1[0:i], text2[0:j] with d differences
        dp = [[[0 for _ in range(k + 1)] for _ in range(n + 1)] for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                for d in range(k + 1):
                    if text1[i - 1] == text2[j - 1]:
                        dp[i][j][d] = dp[i - 1][j - 1][d] + 1
                    else:
                        dp[i][j][d] = max(dp[i - 1][j][d], dp[i][j - 1][d])
                        
                        # Allow difference if within limit
                        if d > 0:
                            dp[i][j][d] = max(dp[i][j][d], dp[i - 1][j - 1][d - 1] + 1)
        
        return max(dp[m][n][d] for d in range(k + 1))

# ==================== LONGEST COMMON SUBSTRING ====================

class LongestCommonSubstring:
    """
    Longest Common Substring Problems
    
    Unlike subsequence, substring must be contiguous.
    """
    
    def lcs_substring_length(self, text1: str, text2: str) -> int:
        """
        Find length of longest common substring
        
        Time Complexity: O(m * n)
        Space Complexity: O(min(m, n))
        
        Args:
            text1: First string
            text2: Second string
        
        Returns:
            Length of longest common substring
        """
        m, n = len(text1), len(text2)
        
        if m < n:
            text1, text2 = text2, text1
            m, n = n, m
        
        dp = [0] * (n + 1)
        max_length = 0
        
        for i in range(1, m + 1):
            prev = 0
            for j in range(1, n + 1):
                temp = dp[j]
                if text1[i - 1] == text2[j - 1]:
                    dp[j] = prev + 1
                    max_length = max(max_length, dp[j])
                else:
                    dp[j] = 0
                prev = temp
        
        return max_length
    
    def lcs_substring_with_position(self, text1: str, text2: str) -> Tuple[int, int, int]:
        """
        Find longest common substring with its position
        
        Args:
            text1: First string
            text2: Second string
        
        Returns:
            Tuple of (length, start_pos_in_text1, start_pos_in_text2)
        """
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        max_length = 0
        ending_pos_i = 0
        ending_pos_j = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > max_length:
                        max_length = dp[i][j]
                        ending_pos_i = i
                        ending_pos_j = j
                else:
                    dp[i][j] = 0
        
        start_i = ending_pos_i - max_length
        start_j = ending_pos_j - max_length
        
        return max_length, start_i, start_j
    
    def all_common_substrings(self, text1: str, text2: str, min_length: int = 1) -> List[str]:
        """
        Find all common substrings of at least min_length
        
        Args:
            text1: First string
            text2: Second string
            min_length: Minimum length of substrings to return
        
        Returns:
            List of common substrings
        """
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        common_substrings = set()
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] >= min_length:
                        start = i - dp[i][j]
                        substring = text1[start:i]
                        common_substrings.add(substring)
                else:
                    dp[i][j] = 0
        
        return list(common_substrings)

# ==================== EDIT DISTANCE ====================

class EditDistance:
    """
    Edit Distance Problems (Levenshtein Distance)
    
    Minimum number of operations to transform one string into another.
    Operations: insert, delete, replace.
    """
    
    def min_distance(self, word1: str, word2: str) -> int:
        """
        Minimum edit distance between two strings
        
        LeetCode 72 - Edit Distance
        
        Time Complexity: O(m * n)
        Space Complexity: O(min(m, n))
        
        Args:
            word1: Source string
            word2: Target string
        
        Returns:
            Minimum number of operations
        """
        m, n = len(word1), len(word2)
        
        # Optimize space by using shorter string for DP array
        if m < n:
            word1, word2 = word2, word1
            m, n = n, m
        
        dp = list(range(n + 1))  # Initialize with distances for empty word1
        
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i  # Distance for empty word2
            
            for j in range(1, n + 1):
                temp = dp[j]
                if word1[i - 1] == word2[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        
        return dp[n]
    
    def min_distance_with_operations(self, word1: str, word2: str) -> Tuple[int, List[str]]:
        """
        Find minimum edit distance and the actual operations
        
        Args:
            word1: Source string
            word2: Target string
        
        Returns:
            Tuple of (distance, operations_list)
        """
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # Delete
                        dp[i][j - 1],      # Insert
                        dp[i - 1][j - 1]   # Replace
                    )
        
        # Backtrack to find operations
        operations = []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and word1[i - 1] == word2[j - 1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                operations.append(f"Replace '{word1[i - 1]}' with '{word2[j - 1]}' at position {i - 1}")
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                operations.append(f"Delete '{word1[i - 1]}' at position {i - 1}")
                i -= 1
            elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
                operations.append(f"Insert '{word2[j - 1]}' at position {i}")
                j -= 1
        
        operations.reverse()
        return dp[m][n], operations
    
    def min_distance_with_costs(self, word1: str, word2: str, 
                               insert_cost: int, delete_cost: int, replace_cost: int) -> int:
        """
        Edit distance with custom operation costs
        
        Args:
            word1: Source string
            word2: Target string
            insert_cost: Cost of insert operation
            delete_cost: Cost of delete operation
            replace_cost: Cost of replace operation
        
        Returns:
            Minimum cost to transform word1 to word2
        """
        m, n = len(word1), len(word2)
        dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        dp[0][0] = 0
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + delete_cost
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + insert_cost
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j] + delete_cost,      # Delete
                        dp[i][j - 1] + insert_cost,      # Insert
                        dp[i - 1][j - 1] + replace_cost  # Replace
                    )
        
        return dp[m][n]
    
    def one_edit_distance(self, s: str, t: str) -> bool:
        """
        Check if strings are exactly one edit distance apart
        
        LeetCode 161 - One Edit Distance
        
        Time Complexity: O(min(m, n))
        Space Complexity: O(1)
        """
        m, n = len(s), len(t)
        
        # Ensure s is shorter
        if m > n:
            return self.one_edit_distance(t, s)
        
        # Length difference should be at most 1
        if n - m > 1:
            return False
        
        for i in range(m):
            if s[i] != t[i]:
                if m == n:
                    # Replace: rest should be identical
                    return s[i + 1:] == t[i + 1:]
                else:
                    # Insert: s should match t[i+1:]
                    return s[i:] == t[i + 1:]
        
        # All characters matched, check if exactly one insertion needed
        return n - m == 1

# ==================== PALINDROME PROBLEMS ====================

class PalindromeProblems:
    """
    Palindrome-related DP problems
    
    Various problems involving palindromes: detection, longest palindromic
    subsequence/substring, minimum insertions/deletions.
    """
    
    def longest_palindromic_subsequence(self, s: str) -> int:
        """
        Find length of longest palindromic subsequence
        
        LeetCode 516 - Longest Palindromic Subsequence
        
        Time Complexity: O(n²)
        Space Complexity: O(n)
        
        Args:
            s: Input string
        
        Returns:
            Length of longest palindromic subsequence
        """
        n = len(s)
        dp = [1] * n  # Single characters are palindromes
        
        for i in range(n - 2, -1, -1):
            prev = 0
            for j in range(i + 1, n):
                temp = dp[j]
                if s[i] == s[j]:
                    dp[j] = prev + 2
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = temp
        
        return dp[n - 1]
    
    def longest_palindromic_substring(self, s: str) -> str:
        """
        Find longest palindromic substring
        
        LeetCode 5 - Longest Palindromic Substring
        
        Time Complexity: O(n²)
        Space Complexity: O(1)
        
        Args:
            s: Input string
        
        Returns:
            Longest palindromic substring
        """
        if not s:
            return ""
        
        start = 0
        max_len = 1
        
        def expand_around_center(left: int, right: int) -> int:
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1
        
        for i in range(len(s)):
            # Odd length palindromes (center at i)
            len1 = expand_around_center(i, i)
            
            # Even length palindromes (center between i and i+1)
            len2 = expand_around_center(i, i + 1)
            
            current_max = max(len1, len2)
            if current_max > max_len:
                max_len = current_max
                start = i - (current_max - 1) // 2
        
        return s[start:start + max_len]
    
    def min_insertions_for_palindrome(self, s: str) -> int:
        """
        Minimum insertions to make string palindrome
        
        LeetCode 1312 - Minimum Insertion Steps to Make a String Palindrome
        
        Key insight: min_insertions = n - LPS(s)
        where LPS is longest palindromic subsequence
        
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        n = len(s)
        lps_length = self.longest_palindromic_subsequence(s)
        return n - lps_length
    
    def min_deletions_for_palindrome(self, s: str) -> int:
        """
        Minimum deletions to make string palindrome
        
        Same as minimum insertions for palindrome
        """
        return self.min_insertions_for_palindrome(s)
    
    def palindromic_substrings_count(self, s: str) -> int:
        """
        Count all palindromic substrings
        
        LeetCode 647 - Palindromic Substrings
        
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        count = 0
        
        def expand_around_center(left: int, right: int) -> int:
            local_count = 0
            while left >= 0 and right < len(s) and s[left] == s[right]:
                local_count += 1
                left -= 1
                right += 1
            return local_count
        
        for i in range(len(s)):
            # Odd length palindromes
            count += expand_around_center(i, i)
            
            # Even length palindromes
            count += expand_around_center(i, i + 1)
        
        return count
    
    def palindrome_partitioning_min_cuts(self, s: str) -> int:
        """
        Minimum cuts to partition string into palindromes
        
        LeetCode 132 - Palindrome Partitioning II
        
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        """
        n = len(s)
        
        # is_palindrome[i][j] = True if s[i:j+1] is palindrome
        is_palindrome = [[False] * n for _ in range(n)]
        
        # Every single character is a palindrome
        for i in range(n):
            is_palindrome[i][i] = True
        
        # Check for length 2
        for i in range(n - 1):
            is_palindrome[i][i + 1] = (s[i] == s[i + 1])
        
        # Check for lengths greater than 2
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                is_palindrome[i][j] = (s[i] == s[j] and is_palindrome[i + 1][j - 1])
        
        # dp[i] = minimum cuts needed for s[0:i+1]
        dp = [0] * n
        
        for i in range(n):
            if is_palindrome[0][i]:
                dp[i] = 0
            else:
                dp[i] = float('inf')
                for j in range(i):
                    if is_palindrome[j + 1][i]:
                        dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n - 1]

# ==================== SHORTEST COMMON SUPERSEQUENCE ====================

class ShortestCommonSupersequence:
    """
    Shortest Common Supersequence Problems
    
    Find the shortest string that contains both input strings as subsequences.
    """
    
    def scs_length(self, str1: str, str2: str) -> int:
        """
        Find length of shortest common supersequence
        
        Key insight: SCS_length = len(str1) + len(str2) - LCS_length
        
        Time Complexity: O(m * n)
        Space Complexity: O(min(m, n))
        """
        lcs_len = LongestCommonSubsequence().lcs_length(str1, str2)
        return len(str1) + len(str2) - lcs_len
    
    def scs_string(self, str1: str, str2: str) -> str:
        """
        Find actual shortest common supersequence string
        
        LeetCode 1092 - Shortest Common Supersequence
        
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(str1), len(str2)
        
        # First find LCS table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        # Build SCS by backtracking
        scs = []
        i, j = m, n
        
        while i > 0 and j > 0:
            if str1[i - 1] == str2[j - 1]:
                scs.append(str1[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                scs.append(str1[i - 1])
                i -= 1
            else:
                scs.append(str2[j - 1])
                j -= 1
        
        # Add remaining characters
        while i > 0:
            scs.append(str1[i - 1])
            i -= 1
        
        while j > 0:
            scs.append(str2[j - 1])
            j -= 1
        
        return ''.join(reversed(scs))

# ==================== ADVANCED STRING DP ====================

class AdvancedStringDP:
    """
    Advanced string DP problems
    """
    
    def wildcard_matching(self, s: str, p: str) -> bool:
        """
        Wildcard pattern matching
        
        LeetCode 44 - Wildcard Matching
        '?' matches any single character
        '*' matches any sequence of characters (including empty)
        
        Time Complexity: O(m * n)
        Space Complexity: O(n)
        """
        m, n = len(s), len(p)
        dp = [False] * (n + 1)
        dp[0] = True
        
        # Handle patterns like a*b* where * can match empty
        for j in range(1, n + 1):
            dp[j] = dp[j - 1] and p[j - 1] == '*'
        
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = False
            
            for j in range(1, n + 1):
                temp = dp[j]
                
                if p[j - 1] == '*':
                    dp[j] = dp[j - 1] or dp[j] or prev
                elif p[j - 1] == '?' or s[i - 1] == p[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = False
                
                prev = temp
        
        return dp[n]
    
    def regular_expression_matching(self, s: str, p: str) -> bool:
        """
        Regular expression matching
        
        LeetCode 10 - Regular Expression Matching
        '.' matches any single character
        '*' matches zero or more of the preceding element
        
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        
        dp[0][0] = True
        
        # Handle patterns like a*b*c* that can match empty string
        for j in range(2, n + 1):
            dp[0][j] = dp[0][j - 2] and p[j - 1] == '*'
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    # Zero occurrence of preceding character
                    dp[i][j] = dp[i][j - 2]
                    
                    # One or more occurrences
                    if p[j - 2] == '.' or s[i - 1] == p[j - 2]:
                        dp[i][j] = dp[i][j] or dp[i - 1][j]
                elif p[j - 1] == '.' or s[i - 1] == p[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
        
        return dp[m][n]
    
    def distinct_subsequences(self, s: str, t: str) -> int:
        """
        Count distinct subsequences
        
        LeetCode 115 - Distinct Subsequences
        Count how many ways t appears as subsequence in s
        
        Time Complexity: O(m * n)
        Space Complexity: O(n)
        """
        m, n = len(s), len(t)
        dp = [0] * (n + 1)
        dp[0] = 1  # Empty string is subsequence of any string
        
        for i in range(1, m + 1):
            # Process from right to left to avoid overwriting
            for j in range(min(i, n), 0, -1):
                if s[i - 1] == t[j - 1]:
                    dp[j] += dp[j - 1]
        
        return dp[n]
    
    def interleaving_string(self, s1: str, s2: str, s3: str) -> bool:
        """
        Check if s3 is interleaving of s1 and s2
        
        LeetCode 97 - Interleaving String
        
        Time Complexity: O(m * n)
        Space Complexity: O(n)
        """
        m, n = len(s1), len(s2)
        
        if len(s3) != m + n:
            return False
        
        dp = [False] * (n + 1)
        dp[0] = True
        
        # Initialize first row
        for j in range(1, n + 1):
            dp[j] = dp[j - 1] and s2[j - 1] == s3[j - 1]
        
        for i in range(1, m + 1):
            dp[0] = dp[0] and s1[i - 1] == s3[i - 1]
            
            for j in range(1, n + 1):
                dp[j] = ((dp[j] and s1[i - 1] == s3[i + j - 1]) or
                        (dp[j - 1] and s2[j - 1] == s3[i + j - 1]))
        
        return dp[n]

# ==================== PERFORMANCE ANALYSIS ====================

def performance_comparison():
    """Compare performance of different string DP approaches"""
    print("=== String DP Performance Analysis ===\n")
    
    # Test LCS
    lcs = LongestCommonSubsequence()
    text1, text2 = "abcdgh", "aedfhr"
    
    print(f"LCS for '{text1}' and '{text2}':")
    
    start_time = time.time()
    lcs_len = lcs.lcs_length(text1, text2)
    time_len = time.time() - start_time
    
    start_time = time.time()
    lcs_str = lcs.lcs_string(text1, text2)
    time_str = time.time() - start_time
    
    print(f"  Length only: {lcs_len} ({time_len:.6f}s)")
    print(f"  String: '{lcs_str}' ({time_str:.6f}s)")
    
    # Test Edit Distance
    edit = EditDistance()
    word1, word2 = "intention", "execution"
    
    start_time = time.time()
    edit_dist = edit.min_distance(word1, word2)
    time_edit = time.time() - start_time
    
    print(f"\nEdit distance for '{word1}' → '{word2}': {edit_dist} ({time_edit:.6f}s)")

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== String DP Demo ===\n")
    
    # LCS Problems
    print("1. Longest Common Subsequence:")
    lcs = LongestCommonSubsequence()
    
    text1, text2 = "abcde", "ace"
    lcs_len = lcs.lcs_length(text1, text2)
    lcs_str = lcs.lcs_string(text1, text2)
    
    print(f"  Strings: '{text1}', '{text2}'")
    print(f"  LCS length: {lcs_len}")
    print(f"  LCS string: '{lcs_str}'")
    
    # Three strings
    text3 = "aec"
    lcs_three = lcs.lcs_three_strings(text1, text2, text3)
    print(f"  LCS of three strings '{text1}', '{text2}', '{text3}': {lcs_three}")
    print()
    
    # Longest Common Substring
    print("2. Longest Common Substring:")
    lcs_sub = LongestCommonSubstring()
    
    str1, str2 = "GeeksforGeeks", "GeeksQuiz"
    sub_len = lcs_sub.lcs_substring_length(str1, str2)
    sub_len_pos, start1, start2 = lcs_sub.lcs_substring_with_position(str1, str2)
    
    print(f"  Strings: '{str1}', '{str2}'")
    print(f"  LCS substring length: {sub_len}")
    print(f"  Position: starts at {start1} in first, {start2} in second")
    print(f"  Substring: '{str1[start1:start1+sub_len_pos]}'")
    
    all_common = lcs_sub.all_common_substrings(str1, str2, 2)
    print(f"  All common substrings (length ≥ 2): {all_common}")
    print()
    
    # Edit Distance
    print("3. Edit Distance:")
    edit = EditDistance()
    
    word1, word2 = "horse", "ros"
    edit_dist = edit.min_distance(word1, word2)
    edit_dist_ops, operations = edit.min_distance_with_operations(word1, word2)
    
    print(f"  Transform '{word1}' → '{word2}'")
    print(f"  Minimum edit distance: {edit_dist}")
    print(f"  Operations:")
    for op in operations:
        print(f"    {op}")
    
    # Custom costs
    custom_dist = edit.min_distance_with_costs(word1, word2, 2, 3, 1)
    print(f"  With custom costs (insert=2, delete=3, replace=1): {custom_dist}")
    
    # One edit distance
    s, t = "ab", "acb"
    one_edit = edit.one_edit_distance(s, t)
    print(f"  '{s}' and '{t}' are one edit apart: {one_edit}")
    print()
    
    # Palindrome Problems
    print("4. Palindrome Problems:")
    palindrome = PalindromeProblems()
    
    s = "bbbab"
    lps_len = palindrome.longest_palindromic_subsequence(s)
    lps_str = palindrome.longest_palindromic_substring(s)
    min_insertions = palindrome.min_insertions_for_palindrome(s)
    pal_count = palindrome.palindromic_substrings_count(s)
    
    print(f"  String: '{s}'")
    print(f"  Longest palindromic subsequence length: {lps_len}")
    print(f"  Longest palindromic substring: '{lps_str}'")
    print(f"  Minimum insertions for palindrome: {min_insertions}")
    print(f"  Total palindromic substrings: {pal_count}")
    
    # Palindrome partitioning
    s2 = "aab"
    min_cuts = palindrome.palindrome_partitioning_min_cuts(s2)
    print(f"  Minimum cuts for '{s2}' palindrome partitioning: {min_cuts}")
    print()
    
    # Shortest Common Supersequence
    print("5. Shortest Common Supersequence:")
    scs = ShortestCommonSupersequence()
    
    str1, str2 = "abac", "cab"
    scs_len = scs.scs_length(str1, str2)
    scs_str = scs.scs_string(str1, str2)
    
    print(f"  Strings: '{str1}', '{str2}'")
    print(f"  SCS length: {scs_len}")
    print(f"  SCS string: '{scs_str}'")
    print()
    
    # Advanced String DP
    print("6. Advanced String DP:")
    advanced = AdvancedStringDP()
    
    # Wildcard matching
    s, p = "adceb", "*a*b*"
    wildcard_match = advanced.wildcard_matching(s, p)
    print(f"  Wildcard: '{s}' matches '{p}': {wildcard_match}")
    
    # Regular expression
    s, p = "aa", "a*"
    regex_match = advanced.regular_expression_matching(s, p)
    print(f"  Regex: '{s}' matches '{p}': {regex_match}")
    
    # Distinct subsequences
    s, t = "rabbbit", "rabbit"
    distinct_count = advanced.distinct_subsequences(s, t)
    print(f"  Distinct subsequences of '{t}' in '{s}': {distinct_count}")
    
    # Interleaving string
    s1, s2, s3 = "aabcc", "dbbca", "aadbbcbcac"
    is_interleaving = advanced.interleaving_string(s1, s2, s3)
    print(f"  '{s3}' is interleaving of '{s1}' and '{s2}': {is_interleaving}")
    print()
    
    # Performance comparison
    performance_comparison()
    print()
    
    # Pattern Recognition Guide
    print("=== String DP Pattern Recognition ===")
    print("Common String DP Patterns:")
    print("  1. LCS: Find longest common subsequence")
    print("  2. LCSubstring: Find longest common substring")
    print("  3. Edit Distance: Transform one string to another")
    print("  4. Palindromes: Problems involving palindromic properties")
    print("  5. Pattern Matching: Wildcard/regex matching")
    print("  6. String Transformation: SCS, interleaving, etc.")
    
    print("\nState Definition Tips:")
    print("  1. dp[i][j]: Consider substrings s1[0:i] and s2[0:j]")
    print("  2. dp[i]: Consider substring s[0:i]")
    print("  3. Add extra dimensions for additional constraints")
    print("  4. Think about what subproblems need to be solved")
    
    print("\nSpace Optimization:")
    print("  1. 2D → 1D when only previous row needed")
    print("  2. Use shorter string for DP array dimension")
    print("  3. Process diagonally for some problems")
    print("  4. Rolling arrays for multi-dimensional problems")
    
    print("\nReal-world Applications:")
    print("  1. DNA sequence analysis (bioinformatics)")
    print("  2. Text comparison and diff algorithms")
    print("  3. Spell checkers and autocorrect")
    print("  4. Version control systems")
    print("  5. Natural language processing")
    print("  6. Data compression algorithms")
    
    print("\n=== Demo Complete ===") 