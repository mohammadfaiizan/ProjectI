"""
String Searching Algorithms
==========================

Topics: Pattern matching, string algorithms, text processing
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Medium to Hard
LeetCode: 28, 214, 686, 459
"""

from typing import List, Optional

class StringSearchingAlgorithms:
    
    # ==========================================
    # 1. NAIVE STRING MATCHING
    # ==========================================
    
    def naive_search(self, text: str, pattern: str) -> List[int]:
        """Naive string matching algorithm
        Time: O(n*m), Space: O(1)
        """
        positions = []
        n, m = len(text), len(pattern)
        
        for i in range(n - m + 1):
            j = 0
            while j < m and text[i + j] == pattern[j]:
                j += 1
            
            if j == m:
                positions.append(i)
        
        return positions
    
    # ==========================================
    # 2. KMP ALGORITHM
    # ==========================================
    
    def kmp_search(self, text: str, pattern: str) -> List[int]:
        """KMP (Knuth-Morris-Pratt) string matching
        Time: O(n + m), Space: O(m)
        """
        if not pattern:
            return []
        
        # Build failure function (LPS array)
        lps = self._compute_lps(pattern)
        
        positions = []
        n, m = len(text), len(pattern)
        i = j = 0
        
        while i < n:
            if text[i] == pattern[j]:
                i += 1
                j += 1
            
            if j == m:
                positions.append(i - j)
                j = lps[j - 1]
            elif i < n and text[i] != pattern[j]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        
        return positions
    
    def _compute_lps(self, pattern: str) -> List[int]:
        """Compute Longest Proper Prefix which is also Suffix"""
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
    
    # ==========================================
    # 3. RABIN-KARP ALGORITHM
    # ==========================================
    
    def rabin_karp_search(self, text: str, pattern: str, prime: int = 101) -> List[int]:
        """Rabin-Karp string matching using rolling hash
        Time: O(n + m) average, O(n*m) worst, Space: O(1)
        """
        positions = []
        n, m = len(text), len(pattern)
        d = 256  # Number of characters in alphabet
        
        if m > n:
            return positions
        
        # Calculate hash values
        pattern_hash = 0
        text_hash = 0
        h = 1
        
        # Calculate h = pow(d, m-1) % prime
        for i in range(m - 1):
            h = (h * d) % prime
        
        # Calculate hash of pattern and first window of text
        for i in range(m):
            pattern_hash = (d * pattern_hash + ord(pattern[i])) % prime
            text_hash = (d * text_hash + ord(text[i])) % prime
        
        # Slide the pattern over text
        for i in range(n - m + 1):
            # Check if hash values match
            if pattern_hash == text_hash:
                # Check characters one by one
                if text[i:i + m] == pattern:
                    positions.append(i)
            
            # Calculate hash for next window
            if i < n - m:
                text_hash = (d * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % prime
                if text_hash < 0:
                    text_hash += prime
        
        return positions
    
    # ==========================================
    # 4. BOYER-MOORE ALGORITHM
    # ==========================================
    
    def boyer_moore_search(self, text: str, pattern: str) -> List[int]:
        """Boyer-Moore string matching algorithm
        Time: O(n/m) best, O(n*m) worst, Space: O(m + σ)
        """
        positions = []
        n, m = len(text), len(pattern)
        
        if m > n:
            return positions
        
        # Build bad character heuristic table
        bad_char = self._build_bad_char_table(pattern)
        
        # Search for pattern
        s = 0  # shift of the pattern
        
        while s <= n - m:
            j = m - 1
            
            # Keep reducing j while characters match
            while j >= 0 and pattern[j] == text[s + j]:
                j -= 1
            
            if j < 0:
                # Pattern found
                positions.append(s)
                # Shift pattern to align with next possible match
                s += (m - bad_char.get(text[s + m], -1) - 1) if s + m < n else 1
            else:
                # Shift pattern based on bad character heuristic
                s += max(1, j - bad_char.get(text[s + j], -1))
        
        return positions
    
    def _build_bad_char_table(self, pattern: str) -> dict:
        """Build bad character heuristic table"""
        bad_char = {}
        m = len(pattern)
        
        for i in range(m):
            bad_char[pattern[i]] = i
        
        return bad_char
    
    # ==========================================
    # 5. STRING MATCHING APPLICATIONS
    # ==========================================
    
    def find_anagrams(self, s: str, p: str) -> List[int]:
        """LC 438: Find All Anagrams in a String
        Time: O(n), Space: O(1)
        """
        if len(p) > len(s):
            return []
        
        result = []
        p_count = [0] * 26
        window_count = [0] * 26
        
        # Count characters in pattern
        for char in p:
            p_count[ord(char) - ord('a')] += 1
        
        # Initialize window
        for i in range(len(p)):
            window_count[ord(s[i]) - ord('a')] += 1
        
        # Check first window
        if window_count == p_count:
            result.append(0)
        
        # Slide window
        for i in range(len(p), len(s)):
            # Add new character
            window_count[ord(s[i]) - ord('a')] += 1
            # Remove old character
            window_count[ord(s[i - len(p)]) - ord('a')] -= 1
            
            if window_count == p_count:
                result.append(i - len(p) + 1)
        
        return result
    
    def repeated_substring_pattern(self, s: str) -> bool:
        """LC 459: Repeated Substring Pattern
        Time: O(n), Space: O(n)
        """
        # If s is made of repeated pattern, then s + s will contain s starting at index 1
        return s in (s + s)[1:-1]
    
    def shortest_palindrome(self, s: str) -> str:
        """LC 214: Shortest Palindrome
        Time: O(n), Space: O(n)
        """
        if not s:
            return s
        
        # Create a string: s + '#' + reverse(s)
        combined = s + '#' + s[::-1]
        
        # Find LPS of combined string
        lps = self._compute_lps(combined)
        
        # The LPS value at the end gives us the length of longest palindromic prefix
        palindrome_length = lps[-1]
        
        # Add the non-palindromic suffix in reverse to the beginning
        to_add = s[palindrome_length:][::-1]
        return to_add + s
    
    def is_rotation(self, s1: str, s2: str) -> bool:
        """Check if s2 is rotation of s1
        Time: O(n), Space: O(n)
        """
        if len(s1) != len(s2):
            return False
        
        return s2 in s1 + s1
    
    def longest_common_prefix(self, strs: List[str]) -> str:
        """LC 14: Longest Common Prefix
        Time: O(S) where S is sum of all characters, Space: O(1)
        """
        if not strs:
            return ""
        
        # Find minimum length
        min_len = min(len(s) for s in strs)
        
        for i in range(min_len):
            char = strs[0][i]
            for s in strs[1:]:
                if s[i] != char:
                    return strs[0][:i]
        
        return strs[0][:min_len]

# Test Examples
def run_examples():
    string_search = StringSearchingAlgorithms()
    
    print("=== STRING SEARCHING ALGORITHMS ===\n")
    
    text = "ABABDABACDABABCABCABCABCABCABC"
    pattern = "ABABCAB"
    
    print(f"Text: {text}")
    print(f"Pattern: {pattern}")
    
    # Test different algorithms
    print("\n1. NAIVE SEARCH:")
    naive_result = string_search.naive_search(text, pattern)
    print(f"Found at positions: {naive_result}")
    
    print("\n2. KMP SEARCH:")
    kmp_result = string_search.kmp_search(text, pattern)
    print(f"Found at positions: {kmp_result}")
    
    print("\n3. RABIN-KARP SEARCH:")
    rk_result = string_search.rabin_karp_search(text, pattern)
    print(f"Found at positions: {rk_result}")
    
    print("\n4. BOYER-MOORE SEARCH:")
    bm_result = string_search.boyer_moore_search(text, pattern)
    print(f"Found at positions: {bm_result}")
    
    # Test applications
    print("\n5. STRING APPLICATIONS:")
    
    # Find anagrams
    s = "abab"
    p = "ab"
    anagrams = string_search.find_anagrams(s, p)
    print(f"Anagrams of '{p}' in '{s}': {anagrams}")
    
    # Repeated pattern
    repeated = string_search.repeated_substring_pattern("abcabcabcabc")
    print(f"'abcabcabcabc' has repeated pattern: {repeated}")
    
    # String rotation
    rotation = string_search.is_rotation("waterbottle", "erbottlewat")
    print(f"'erbottlewat' is rotation of 'waterbottle': {rotation}")

if __name__ == "__main__":
    run_examples() 