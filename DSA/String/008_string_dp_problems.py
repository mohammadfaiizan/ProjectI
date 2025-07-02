"""
String Dynamic Programming - Advanced Problems
==============================================

Topics: Edit distance, word break, string matching DP
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Medium to Hard
"""

from typing import List, Set, Dict

class StringDPProblems:
    
    # ==========================================
    # 1. EDIT DISTANCE PROBLEMS
    # ==========================================
    
    def edit_distance(self, word1: str, word2: str) -> int:
        """LC 72: Edit Distance - Time: O(m*n), Space: O(m*n)"""
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )
        
        return dp[m][n]
    
    def min_distance_delete_only(self, word1: str, word2: str) -> int:
        """LC 583: Delete Operation for Two Strings"""
        m, n = len(word1), len(word2)
        
        # Find LCS length
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        return (m - lcs_length) + (n - lcs_length)
    
    def one_edit_distance(self, s: str, t: str) -> bool:
        """LC 161: One Edit Distance"""
        m, n = len(s), len(t)
        
        if abs(m - n) > 1:
            return False
        
        if m > n:
            return self.one_edit_distance(t, s)
        
        for i in range(m):
            if s[i] != t[i]:
                if m == n:
                    return s[i+1:] == t[i+1:]  # replace
                else:
                    return s[i:] == t[i+1:]    # insert
        
        return n - m == 1
    
    # ==========================================
    # 2. WORD BREAK PROBLEMS
    # ==========================================
    
    def word_break(self, s: str, wordDict: List[str]) -> bool:
        """LC 139: Word Break - Time: O(n²), Space: O(n)"""
        word_set = set(wordDict)
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        
        return dp[n]
    
    def word_break_ii(self, s: str, wordDict: List[str]) -> List[str]:
        """LC 140: Word Break II - Time: O(n²), Space: O(n²)"""
        word_set = set(wordDict)
        memo = {}
        
        def backtrack(start: int) -> List[str]:
            if start in memo:
                return memo[start]
            
            if start == len(s):
                memo[start] = [""]
                return [""]
            
            result = []
            for end in range(start + 1, len(s) + 1):
                word = s[start:end]
                if word in word_set:
                    suffixes = backtrack(end)
                    for suffix in suffixes:
                        result.append(word + (" " + suffix if suffix else ""))
            
            memo[start] = result
            return result
        
        return backtrack(0)
    
    def concatenated_words(self, words: List[str]) -> List[str]:
        """LC 472: Concatenated Words"""
        word_set = set(words)
        result = []
        
        def can_form(word: str) -> bool:
            n = len(word)
            dp = [False] * (n + 1)
            dp[0] = True
            
            for i in range(1, n + 1):
                for j in range(i):
                    if dp[j] and word[j:i] in word_set and word[j:i] != word:
                        dp[i] = True
                        break
            
            return dp[n]
        
        for word in words:
            if can_form(word):
                result.append(word)
        
        return result
    
    # ==========================================
    # 3. PALINDROME DP
    # ==========================================
    
    def count_palindromic_substrings(self, s: str) -> int:
        """LC 647: Palindromic Substrings - Time: O(n²), Space: O(n²)"""
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        count = 0
        
        # Every single character is palindrome
        for i in range(n):
            dp[i][i] = True
            count += 1
        
        # Check for palindromes of length 2
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
                count += 1
        
        # Check for palindromes of length 3 and more
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and dp[i + 1][j - 1]:
                    dp[i][j] = True
                    count += 1
        
        return count
    
    def longest_palindromic_subsequence(self, s: str) -> int:
        """LC 516: Longest Palindromic Subsequence"""
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        
        # Every single character is palindrome of length 1
        for i in range(n):
            dp[i][i] = 1
        
        # Check substrings of length 2 and more
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j]:
                    dp[i][j] = 2 + dp[i + 1][j - 1]
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        
        return dp[0][n - 1]
    
    # ==========================================
    # 4. PATTERN MATCHING DP
    # ==========================================
    
    def is_match_regex(self, s: str, p: str) -> bool:
        """LC 10: Regular Expression Matching"""
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        
        dp[0][0] = True
        
        # Handle patterns like a*b*c*
        for j in range(2, n + 1):
            if p[j-1] == '*':
                dp[0][j] = dp[0][j-2]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j-1] == s[i-1] or p[j-1] == '.':
                    dp[i][j] = dp[i-1][j-1]
                elif p[j-1] == '*':
                    dp[i][j] = dp[i][j-2]  # zero occurrences
                    if p[j-2] == s[i-1] or p[j-2] == '.':
                        dp[i][j] = dp[i][j] or dp[i-1][j]
        
        return dp[m][n]
    
    def is_match_wildcard(self, s: str, p: str) -> bool:
        """LC 44: Wildcard Pattern Matching"""
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        
        dp[0][0] = True
        
        # Handle patterns starting with *
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[0][j] = dp[0][j-1]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j-1] == '*':
                    dp[i][j] = dp[i-1][j] or dp[i][j-1]
                elif p[j-1] == '?' or s[i-1] == p[j-1]:
                    dp[i][j] = dp[i-1][j-1]
        
        return dp[m][n]
    
    # ==========================================
    # 5. SUBSEQUENCE PROBLEMS
    # ==========================================
    
    def num_distinct_subsequences(self, s: str, t: str) -> int:
        """LC 115: Distinct Subsequences"""
        m, n = len(s), len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Empty string t can be formed in one way
        for i in range(m + 1):
            dp[i][0] = 1
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j]
                if s[i-1] == t[j-1]:
                    dp[i][j] += dp[i-1][j-1]
        
        return dp[m][n]
    
    def is_subsequence_dp(self, s: str, t: str) -> bool:
        """LC 392: Is Subsequence using DP"""
        m, n = len(s), len(t)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        
        # Empty string is subsequence of any string
        for j in range(n + 1):
            dp[0][j] = True
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i-1] == t[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = dp[i][j-1]
        
        return dp[m][n]
    
    # ==========================================
    # 6. ADVANCED STRING DP
    # ==========================================
    
    def min_insertions_palindrome(self, s: str) -> int:
        """LC 1312: Minimum Insertion Steps to Make String Palindrome"""
        n = len(s)
        lps_length = self.longest_palindromic_subsequence(s)
        return n - lps_length
    
    def min_deletions_palindrome(self, s: str) -> int:
        """Minimum deletions to make string palindrome"""
        n = len(s)
        lps_length = self.longest_palindromic_subsequence(s)
        return n - lps_length
    
    def count_different_palindromic_subsequences(self, s: str) -> int:
        """LC 730: Count Different Palindromic Subsequences"""
        MOD = 10**9 + 7
        n = len(s)
        
        # dp[i][j] = number of different palindromic subsequences in s[i:j+1]
        dp = [[0] * n for _ in range(n)]
        
        # Every single character is a palindrome
        for i in range(n):
            dp[i][i] = 1
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                if s[i] == s[j]:
                    dp[i][j] = (2 * dp[i+1][j-1]) % MOD
                    
                    # Handle duplicates
                    left, right = i + 1, j - 1
                    while left <= right and s[left] != s[i]:
                        left += 1
                    while left <= right and s[right] != s[i]:
                        right -= 1
                    
                    if left > right:
                        dp[i][j] = (dp[i][j] + 2) % MOD
                    elif left == right:
                        dp[i][j] = (dp[i][j] + 1) % MOD
                    else:
                        dp[i][j] = (dp[i][j] - dp[left+1][right-1]) % MOD
                else:
                    dp[i][j] = (dp[i+1][j] + dp[i][j-1] - dp[i+1][j-1]) % MOD
        
        return dp[0][n-1] % MOD

# Test Examples
def run_examples():
    sdp = StringDPProblems()
    
    print("=== STRING DP EXAMPLES ===\n")
    
    # Edit distance
    print("1. EDIT DISTANCE:")
    print(f"Edit distance 'horse' -> 'ros': {sdp.edit_distance('horse', 'ros')}")
    print(f"One edit distance 'ab' vs 'acb': {sdp.one_edit_distance('ab', 'acb')}")
    
    # Word break
    print("\n2. WORD BREAK:")
    print(f"Word break 'leetcode': {sdp.word_break('leetcode', ['leet', 'code'])}")
    print(f"Word break II 'catsanddog': {sdp.word_break_ii('catsanddog', ['cat', 'cats', 'and', 'sand', 'dog'])}")
    
    # Palindrome DP
    print("\n3. PALINDROME DP:")
    print(f"Count palindromic substrings 'abc': {sdp.count_palindromic_substrings('abc')}")
    print(f"Longest palindromic subsequence 'bbbab': {sdp.longest_palindromic_subsequence('bbbab')}")
    
    # Pattern matching
    print("\n4. PATTERN MATCHING:")
    print(f"Regex match 'aa' with 'a*': {sdp.is_match_regex('aa', 'a*')}")
    print(f"Wildcard match 'adceb' with '*a*b*': {sdp.is_match_wildcard('adceb', '*a*b*')}")
    
    # Advanced problems
    print("\n5. ADVANCED:")
    print(f"Min insertions for palindrome 'zzazz': {sdp.min_insertions_palindrome('zzazz')}")

if __name__ == "__main__":
    run_examples() 