"""
String Subsequences and Substrings - Advanced Problems
=====================================================

Topics: LCS, LIS, substring problems, sequence matching
Companies: Google, Microsoft, Amazon, Facebook
Difficulty: Medium to Hard
"""

from typing import List

class StringSubsequencesSubstrings:
    
    # ==========================================
    # 1. LONGEST COMMON SUBSEQUENCE (LCS)
    # ==========================================
    
    def longest_common_subsequence(self, text1: str, text2: str) -> int:
        """LC 1143: Longest Common Subsequence - Time: O(m*n), Space: O(m*n)"""
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def lcs_string(self, text1: str, text2: str) -> str:
        """Return actual LCS string"""
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Build DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        # Reconstruct LCS
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
        
        return ''.join(reversed(lcs))
    
    # ==========================================
    # 2. LONGEST INCREASING SUBSEQUENCE
    # ==========================================
    
    def longest_increasing_subsequence(self, s: str) -> int:
        """LIS on string characters - Time: O(n²), Space: O(n)"""
        if not s:
            return 0
        
        n = len(s)
        dp = [1] * n
        
        for i in range(1, n):
            for j in range(i):
                if s[j] < s[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
    def lis_binary_search(self, s: str) -> int:
        """LIS using binary search - Time: O(nlogn), Space: O(n)"""
        from bisect import bisect_left
        
        tails = []
        for char in s:
            pos = bisect_left(tails, char)
            if pos == len(tails):
                tails.append(char)
            else:
                tails[pos] = char
        
        return len(tails)
    
    # ==========================================
    # 3. DISTINCT SUBSEQUENCES
    # ==========================================
    
    def num_distinct(self, s: str, t: str) -> int:
        """LC 115: Distinct Subsequences - Time: O(m*n), Space: O(m*n)"""
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
    
    # ==========================================
    # 4. SUBSTRING PROBLEMS
    # ==========================================
    
    def longest_common_substring(self, text1: str, text2: str) -> int:
        """Longest Common Substring - Time: O(m*n), Space: O(m*n)"""
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        max_length = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    max_length = max(max_length, dp[i][j])
                else:
                    dp[i][j] = 0
        
        return max_length
    
    def longest_substring_without_repeating(self, s: str) -> int:
        """LC 3: Longest Substring Without Repeating Characters"""
        char_index = {}
        max_length = 0
        left = 0
        
        for right, char in enumerate(s):
            if char in char_index and char_index[char] >= left:
                left = char_index[char] + 1
            
            char_index[char] = right
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    def longest_substring_k_distinct(self, s: str, k: int) -> int:
        """LC 340: Longest Substring with At Most K Distinct Characters"""
        if k == 0:
            return 0
        
        char_count = {}
        max_length = 0
        left = 0
        
        for right, char in enumerate(s):
            char_count[char] = char_count.get(char, 0) + 1
            
            while len(char_count) > k:
                left_char = s[left]
                char_count[left_char] -= 1
                if char_count[left_char] == 0:
                    del char_count[left_char]
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    # ==========================================
    # 5. INTERLEAVING STRINGS
    # ==========================================
    
    def is_interleave(self, s1: str, s2: str, s3: str) -> bool:
        """LC 97: Interleaving String - Time: O(m*n), Space: O(m*n)"""
        m, n, l = len(s1), len(s2), len(s3)
        
        if m + n != l:
            return False
        
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        
        # Fill first row
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
        
        # Fill first column
        for i in range(1, m + 1):
            dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
        
        # Fill rest of the table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = ((dp[i-1][j] and s1[i-1] == s3[i+j-1]) or
                           (dp[i][j-1] and s2[j-1] == s3[i+j-1]))
        
        return dp[m][n]
    
    # ==========================================
    # 6. SCRAMBLE STRING
    # ==========================================
    
    def is_scramble(self, s1: str, s2: str) -> bool:
        """LC 87: Scramble String - Time: O(n⁴), Space: O(n³)"""
        if len(s1) != len(s2):
            return False
        
        if s1 == s2:
            return True
        
        if sorted(s1) != sorted(s2):
            return False
        
        n = len(s1)
        
        for i in range(1, n):
            # Case 1: No swap
            if (self.is_scramble(s1[:i], s2[:i]) and 
                self.is_scramble(s1[i:], s2[i:])):
                return True
            
            # Case 2: Swap
            if (self.is_scramble(s1[:i], s2[n-i:]) and 
                self.is_scramble(s1[i:], s2[:n-i])):
                return True
        
        return False

# Test Examples
def run_examples():
    sss = StringSubsequencesSubstrings()
    
    print("=== SUBSEQUENCES AND SUBSTRINGS EXAMPLES ===\n")
    
    # LCS
    print("1. LONGEST COMMON SUBSEQUENCE:")
    lcs_len = sss.longest_common_subsequence("abcde", "ace")
    lcs_str = sss.lcs_string("abcde", "ace")
    print(f"LCS length: {lcs_len}, LCS string: '{lcs_str}'")
    
    # LIS
    print(f"LIS length of 'bdca': {sss.longest_increasing_subsequence('bdca')}")
    print(f"LIS binary search: {sss.lis_binary_search('bdca')}")
    
    # Distinct subsequences
    print(f"Distinct subsequences: {sss.num_distinct('rabbbit', 'rabbit')}")
    
    # Substring problems
    print(f"Longest common substring: {sss.longest_common_substring('GeeksforGeeks', 'GeeksQuiz')}")
    print(f"Longest without repeating: {sss.longest_substring_without_repeating('abcabcbb')}")
    print(f"Longest with k=2 distinct: {sss.longest_substring_k_distinct('eceba', 2)}")
    
    # Interleaving and scramble
    print(f"Is interleaving: {sss.is_interleave('aabcc', 'dbbca', 'aadbbcbcac')}")
    print(f"Is scramble: {sss.is_scramble('great', 'rgeat')}")

if __name__ == "__main__":
    run_examples() 