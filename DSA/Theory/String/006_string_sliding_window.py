"""
String Sliding Window - Advanced Techniques
==========================================

Topics: Fixed/variable window, character frequency, optimization
Companies: Facebook, Google, Amazon, Microsoft
Difficulty: Medium to Hard
"""

from typing import List, Dict
from collections import Counter, defaultdict

class StringSlidingWindow:
    
    # ==========================================
    # 1. FIXED SIZE SLIDING WINDOW
    # ==========================================
    
    def find_anagrams_fixed_window(self, s: str, p: str) -> List[int]:
        """LC 438: Find All Anagrams - Fixed Window Size"""
        if len(p) > len(s):
            return []
        
        p_count = Counter(p)
        window_count = Counter(s[:len(p)])
        
        result = []
        if window_count == p_count:
            result.append(0)
        
        for i in range(len(p), len(s)):
            # Add new character
            window_count[s[i]] += 1
            
            # Remove old character
            left_char = s[i - len(p)]
            window_count[left_char] -= 1
            if window_count[left_char] == 0:
                del window_count[left_char]
            
            # Check if current window is anagram
            if window_count == p_count:
                result.append(i - len(p) + 1)
        
        return result
    
    def max_vowels_in_substring(self, s: str, k: int) -> int:
        """LC 1456: Maximum Number of Vowels in Substring of Length k"""
        vowels = set('aeiou')
        
        # Count vowels in first window
        vowel_count = sum(1 for c in s[:k] if c in vowels)
        max_vowels = vowel_count
        
        # Slide the window
        for i in range(k, len(s)):
            # Add new character
            if s[i] in vowels:
                vowel_count += 1
            
            # Remove old character
            if s[i - k] in vowels:
                vowel_count -= 1
            
            max_vowels = max(max_vowels, vowel_count)
        
        return max_vowels
    
    def contains_nearby_duplicate(self, s: str, k: int) -> bool:
        """Check if string contains duplicate characters within distance k"""
        window = set()
        
        for i, char in enumerate(s):
            if i > k:
                window.remove(s[i - k - 1])
            
            if char in window:
                return True
            
            window.add(char)
        
        return False
    
    # ==========================================
    # 2. VARIABLE SIZE SLIDING WINDOW
    # ==========================================
    
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
    
    def longest_substring_exactly_k_distinct(self, s: str, k: int) -> int:
        """Longest substring with exactly k distinct characters"""
        def at_most_k_distinct(k: int) -> int:
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
        
        return at_most_k_distinct(k) - at_most_k_distinct(k - 1)
    
    # ==========================================
    # 3. MINIMUM WINDOW PROBLEMS
    # ==========================================
    
    def min_window_substring(self, s: str, t: str) -> str:
        """LC 76: Minimum Window Substring"""
        if len(t) > len(s):
            return ""
        
        t_count = Counter(t)
        window_count = Counter()
        
        left = 0
        min_len = float('inf')
        min_start = 0
        required_matches = len(t_count)
        formed_matches = 0
        
        for right in range(len(s)):
            char = s[right]
            window_count[char] += 1
            
            if char in t_count and window_count[char] == t_count[char]:
                formed_matches += 1
            
            while formed_matches == required_matches and left <= right:
                if right - left + 1 < min_len:
                    min_len = right - left + 1
                    min_start = left
                
                char = s[left]
                window_count[char] -= 1
                if char in t_count and window_count[char] < t_count[char]:
                    formed_matches -= 1
                
                left += 1
        
        return s[min_start:min_start + min_len] if min_len != float('inf') else ""
    
    def min_window_with_characters(self, s: str, chars: str) -> str:
        """Find minimum window containing all characters from chars"""
        if not chars:
            return ""
        
        char_count = Counter(chars)
        window_count = Counter()
        
        left = 0
        min_len = float('inf')
        min_start = 0
        required = len(char_count)
        formed = 0
        
        for right in range(len(s)):
            char = s[right]
            window_count[char] += 1
            
            if char in char_count and window_count[char] == char_count[char]:
                formed += 1
            
            while formed == required:
                if right - left + 1 < min_len:
                    min_len = right - left + 1
                    min_start = left
                
                char = s[left]
                window_count[char] -= 1
                if char in char_count and window_count[char] < char_count[char]:
                    formed -= 1
                
                left += 1
        
        return s[min_start:min_start + min_len] if min_len != float('inf') else ""
    
    # ==========================================
    # 4. CHARACTER REPLACEMENT PROBLEMS
    # ==========================================
    
    def character_replacement(self, s: str, k: int) -> int:
        """LC 424: Longest Repeating Character Replacement"""
        char_count = {}
        max_count = 0
        max_length = 0
        left = 0
        
        for right in range(len(s)):
            char_count[s[right]] = char_count.get(s[right], 0) + 1
            max_count = max(max_count, char_count[s[right]])
            
            if right - left + 1 - max_count > k:
                char_count[s[left]] -= 1
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    def max_consecutive_ones_with_flips(self, s: str, k: int) -> int:
        """LC 1004: Max Consecutive Ones III (adapted for strings)"""
        zero_count = 0
        max_length = 0
        left = 0
        
        for right in range(len(s)):
            if s[right] == '0':
                zero_count += 1
            
            while zero_count > k:
                if s[left] == '0':
                    zero_count -= 1
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    # ==========================================
    # 5. FREQUENCY-BASED PROBLEMS
    # ==========================================
    
    def find_all_anagrams_optimized(self, s: str, p: str) -> List[int]:
        """Optimized anagram finding using frequency matching"""
        if len(p) > len(s):
            return []
        
        p_count = [0] * 26
        window_count = [0] * 26
        
        # Count characters in p
        for char in p:
            p_count[ord(char) - ord('a')] += 1
        
        result = []
        
        for i in range(len(s)):
            # Add current character
            window_count[ord(s[i]) - ord('a')] += 1
            
            # Remove character that falls out of window
            if i >= len(p):
                window_count[ord(s[i - len(p)]) - ord('a')] -= 1
            
            # Check if current window is anagram
            if i >= len(p) - 1 and window_count == p_count:
                result.append(i - len(p) + 1)
        
        return result
    
    def checkInclusion_optimized(self, s1: str, s2: str) -> bool:
        """LC 567: Permutation in String - Optimized"""
        if len(s1) > len(s2):
            return False
        
        s1_count = [0] * 26
        window_count = [0] * 26
        
        # Count characters in s1
        for char in s1:
            s1_count[ord(char) - ord('a')] += 1
        
        for i in range(len(s2)):
            # Add current character
            window_count[ord(s2[i]) - ord('a')] += 1
            
            # Remove character that falls out of window
            if i >= len(s1):
                window_count[ord(s2[i - len(s1)]) - ord('a')] -= 1
            
            # Check if current window matches s1
            if i >= len(s1) - 1 and window_count == s1_count:
                return True
        
        return False
    
    # ==========================================
    # 6. ADVANCED SLIDING WINDOW PROBLEMS
    # ==========================================
    
    def longest_substring_with_same_letters(self, s: str, k: int) -> int:
        """Longest substring with same letters after at most k changes"""
        char_count = {}
        max_count = 0
        max_length = 0
        left = 0
        
        for right in range(len(s)):
            char_count[s[right]] = char_count.get(s[right], 0) + 1
            max_count = max(max_count, char_count[s[right]])
            
            if right - left + 1 - max_count > k:
                char_count[s[left]] -= 1
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    def min_window_all_characters(self, s: str) -> str:
        """Find minimum window containing all unique characters"""
        unique_chars = len(set(s))
        
        char_count = {}
        left = 0
        min_len = float('inf')
        min_start = 0
        
        for right in range(len(s)):
            char_count[s[right]] = char_count.get(s[right], 0) + 1
            
            while len(char_count) == unique_chars:
                if right - left + 1 < min_len:
                    min_len = right - left + 1
                    min_start = left
                
                char_count[s[left]] -= 1
                if char_count[s[left]] == 0:
                    del char_count[s[left]]
                left += 1
        
        return s[min_start:min_start + min_len] if min_len != float('inf') else ""

# Test Examples
def run_examples():
    ssw = StringSlidingWindow()
    
    print("=== SLIDING WINDOW EXAMPLES ===\n")
    
    # Fixed window
    print("1. FIXED WINDOW:")
    print(f"Find anagrams 'ab' in 'abab': {ssw.find_anagrams_fixed_window('abab', 'ab')}")
    print(f"Max vowels in 'abciiidef' k=3: {ssw.max_vowels_in_substring('abciiidef', 3)}")
    
    # Variable window
    print("\n2. VARIABLE WINDOW:")
    print(f"Longest without repeating 'abcabcbb': {ssw.longest_substring_without_repeating('abcabcbb')}")
    print(f"Longest with k=2 distinct 'eceba': {ssw.longest_substring_k_distinct('eceba', 2)}")
    
    # Minimum window
    print("\n3. MINIMUM WINDOW:")
    print(f"Min window 'ADOBECODEBANC' 'ABC': '{ssw.min_window_substring('ADOBECODEBANC', 'ABC')}'")
    
    # Character replacement
    print("\n4. CHARACTER REPLACEMENT:")
    print(f"Longest repeating replacement 'ABAB' k=2: {ssw.character_replacement('ABAB', 2)}")
    
    # Advanced problems
    print("\n5. ADVANCED:")
    print(f"Min window all characters 'aabbcc': '{ssw.min_window_all_characters('aabbcc')}'")

if __name__ == "__main__":
    run_examples() 