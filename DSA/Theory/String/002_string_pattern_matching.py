"""
String Pattern Matching - Advanced Algorithms
=============================================

Topics: KMP, Rabin-Karp, Z-Algorithm, Boyer-Moore
Companies: Google, Facebook, Microsoft, Amazon
Difficulty: Medium to Hard
"""

from typing import List, Dict
from collections import deque

class PatternMatching:
    
    def __init__(self):
        self.prime = 101
        self.base = 256
    
    # ==========================================
    # 1. KMP ALGORITHM
    # ==========================================
    
    def compute_lps_array(self, pattern: str) -> List[int]:
        """Compute LPS (Longest Proper Prefix Suffix) array"""
        m = len(pattern)
        lps = [0] * m
        length = 0
        i = 1
        
        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        return lps
    
    def kmp_search(self, text: str, pattern: str) -> List[int]:
        """KMP Pattern Matching - Time: O(n + m), Space: O(m)"""
        if not pattern:
            return []
        
        n, m = len(text), len(pattern)
        lps = self.compute_lps_array(pattern)
        
        matches = []
        i = j = 0
        
        while i < n:
            if pattern[j] == text[i]:
                i += 1
                j += 1
            
            if j == m:
                matches.append(i - j)
                j = lps[j - 1]
            elif i < n and pattern[j] != text[i]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        
        return matches
    
    def find_repeated_pattern(self, s: str) -> str:
        """Find shortest repeating pattern using KMP"""
        n = len(s)
        lps = self.compute_lps_array(s)
        
        if lps[n - 1] != 0 and n % (n - lps[n - 1]) == 0:
            return s[:n - lps[n - 1]]
        
        return s
    
    # ==========================================
    # 2. RABIN-KARP ALGORITHM
    # ==========================================
    
    def rabin_karp_search(self, text: str, pattern: str) -> List[int]:
        """Rabin-Karp with Rolling Hash - Avg: O(n + m), Worst: O(n*m)"""
        if not pattern:
            return []
        
        n, m = len(text), len(pattern)
        if m > n:
            return []
        
        matches = []
        pattern_hash = 0
        text_hash = 0
        h = 1
        
        # Calculate h = base^(m-1) % prime
        for i in range(m - 1):
            h = (h * self.base) % self.prime
        
        # Calculate initial hash values
        for i in range(m):
            pattern_hash = (self.base * pattern_hash + ord(pattern[i])) % self.prime
            text_hash = (self.base * text_hash + ord(text[i])) % self.prime
        
        # Slide pattern over text
        for i in range(n - m + 1):
            if pattern_hash == text_hash:
                # Verify character by character
                if text[i:i+m] == pattern:
                    matches.append(i)
            
            # Calculate next hash
            if i < n - m:
                text_hash = (self.base * (text_hash - ord(text[i]) * h) + 
                           ord(text[i + m])) % self.prime
                
                if text_hash < 0:
                    text_hash += self.prime
        
        return matches
    
    def longest_duplicate_substring(self, s: str) -> str:
        """LC 1044: Using Binary Search + Rabin-Karp"""
        n = len(s)
        if n <= 1:
            return ""
        
        def has_duplicate(length: int) -> str:
            if length == 0:
                return ""
            
            seen = set()
            base = 26
            prime = 2**63 - 1
            
            # Calculate base^(length-1) % prime
            base_pow = pow(base, length - 1, prime)
            
            # Calculate initial hash
            current_hash = 0
            for i in range(length):
                current_hash = (current_hash * base + ord(s[i]) - ord('a')) % prime
            
            seen.add(current_hash)
            
            # Rolling hash
            for i in range(1, n - length + 1):
                current_hash = (current_hash - (ord(s[i-1]) - ord('a')) * base_pow) % prime
                current_hash = (current_hash * base + ord(s[i + length - 1]) - ord('a')) % prime
                
                if current_hash in seen:
                    return s[i:i + length]
                seen.add(current_hash)
            
            return ""
        
        # Binary search on length
        left, right = 0, n - 1
        result = ""
        
        while left <= right:
            mid = (left + right) // 2
            duplicate = has_duplicate(mid)
            
            if duplicate:
                result = duplicate
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    # ==========================================
    # 3. Z ALGORITHM
    # ==========================================
    
    def z_algorithm(self, s: str) -> List[int]:
        """Z Algorithm - Time: O(n), Space: O(n)"""
        n = len(s)
        z = [0] * n
        l = r = 0
        
        for i in range(1, n):
            if i <= r:
                z[i] = min(r - i + 1, z[i - l])
            
            while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                z[i] += 1
            
            if i + z[i] - 1 > r:
                l, r = i, i + z[i] - 1
        
        return z
    
    def z_search(self, text: str, pattern: str) -> List[int]:
        """Pattern matching using Z Algorithm"""
        if not pattern:
            return []
        
        combined = pattern + "$" + text
        z = self.z_algorithm(combined)
        
        matches = []
        pattern_len = len(pattern)
        
        for i in range(pattern_len + 1, len(combined)):
            if z[i] == pattern_len:
                matches.append(i - pattern_len - 1)
        
        return matches
    
    # ==========================================
    # 4. LEETCODE PROBLEMS
    # ==========================================
    
    def str_str(self, haystack: str, needle: str) -> int:
        """LC 28: Implement strStr() using KMP"""
        if not needle:
            return 0
        
        matches = self.kmp_search(haystack, needle)
        return matches[0] if matches else -1
    
    def repeated_string_match(self, a: str, b: str) -> int:
        """LC 686: Repeated String Match"""
        if not b:
            return 1
        
        min_reps = len(b) // len(a)
        
        for reps in range(min_reps, min_reps + 3):
            if reps <= 0:
                continue
            repeated = a * reps
            if b in repeated:
                return reps
        
        return -1
    
    def find_anagrams(self, s: str, p: str) -> List[int]:
        """LC 438: Find All Anagrams"""
        if len(p) > len(s):
            return []
        
        from collections import Counter
        
        p_count = Counter(p)
        window_count = Counter()
        
        result = []
        left = 0
        
        for right in range(len(s)):
            window_count[s[right]] += 1
            
            if right - left + 1 > len(p):
                if window_count[s[left]] == 1:
                    del window_count[s[left]]
                else:
                    window_count[s[left]] -= 1
                left += 1
            
            if window_count == p_count:
                result.append(left)
        
        return result
    
    def is_match_wildcard(self, s: str, p: str) -> bool:
        """LC 44: Wildcard Pattern Matching"""
        m, n = len(s), len(p)
        
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        
        # Handle patterns with '*' at beginning
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

# Test Examples
def run_examples():
    pm = PatternMatching()
    
    print("=== PATTERN MATCHING EXAMPLES ===\n")
    
    # KMP Algorithm
    text = "ABABDABACDABABCABCABCABC"
    pattern = "ABABCABC"
    matches = pm.kmp_search(text, pattern)
    print(f"KMP search '{pattern}': {matches}")
    
    lps = pm.compute_lps_array("ABABCABAB")
    print(f"LPS array for 'ABABCABAB': {lps}")
    
    # Rabin-Karp
    rk_matches = pm.rabin_karp_search("GEEKS FOR GEEKS", "GEEK")
    print(f"Rabin-Karp 'GEEK': {rk_matches}")
    
    lds = pm.longest_duplicate_substring("banana")
    print(f"Longest duplicate in 'banana': '{lds}'")
    
    # Z Algorithm
    z_array = pm.z_algorithm("aabaaab")
    print(f"Z array for 'aabaaab': {z_array}")
    
    # LeetCode Problems
    print(f"strStr 'll' in 'hello': {pm.str_str('hello', 'll')}")
    print(f"Find anagrams 'ab' in 'abab': {pm.find_anagrams('abab', 'ab')}")
    print(f"Wildcard match 'adceb' with '*a*b*': {pm.is_match_wildcard('adceb', '*a*b*')}")

if __name__ == "__main__":
    run_examples() 