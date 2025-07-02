"""
String Anagrams and Permutations - All Variants
==============================================

Topics: Anagram detection, permutation problems, group anagrams
Companies: Facebook, Google, Amazon, Microsoft
Difficulty: Easy to Hard
"""

from typing import List, Dict
from collections import Counter, defaultdict

class StringAnagramsPermutations:
    
    # ==========================================
    # 1. BASIC ANAGRAM DETECTION
    # ==========================================
    
    def is_anagram_sorting(self, s: str, t: str) -> bool:
        """LC 242: Valid Anagram (Sorting) - Time: O(nlogn), Space: O(1)"""
        return sorted(s) == sorted(t)
    
    def is_anagram_counting(self, s: str, t: str) -> bool:
        """LC 242: Valid Anagram (Counting) - Time: O(n), Space: O(1)"""
        if len(s) != len(t):
            return False
        
        char_count = [0] * 26
        for i in range(len(s)):
            char_count[ord(s[i]) - ord('a')] += 1
            char_count[ord(t[i]) - ord('a')] -= 1
        
        return all(count == 0 for count in char_count)
    
    def is_anagram_counter(self, s: str, t: str) -> bool:
        """Using Counter - Time: O(n), Space: O(1)"""
        return Counter(s) == Counter(t)
    
    # ==========================================
    # 2. GROUP ANAGRAMS
    # ==========================================
    
    def group_anagrams(self, strs: List[str]) -> List[List[str]]:
        """LC 49: Group Anagrams - Time: O(n*mlogm), Space: O(n*m)"""
        anagram_map = defaultdict(list)
        
        for s in strs:
            key = ''.join(sorted(s))
            anagram_map[key].append(s)
        
        return list(anagram_map.values())
    
    def group_anagrams_counting(self, strs: List[str]) -> List[List[str]]:
        """Using character counting - Time: O(n*m), Space: O(n*m)"""
        anagram_map = defaultdict(list)
        
        for s in strs:
            count = [0] * 26
            for char in s:
                count[ord(char) - ord('a')] += 1
            key = tuple(count)
            anagram_map[key].append(s)
        
        return list(anagram_map.values())
    
    # ==========================================
    # 3. ANAGRAM SUBSEQUENCES
    # ==========================================
    
    def find_anagrams(self, s: str, p: str) -> List[int]:
        """LC 438: Find All Anagrams in String - Time: O(n), Space: O(1)"""
        if len(p) > len(s):
            return []
        
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
    
    def check_inclusion(self, s1: str, s2: str) -> bool:
        """LC 567: Permutation in String - Time: O(n), Space: O(1)"""
        if len(s1) > len(s2):
            return False
        
        s1_count = Counter(s1)
        window_count = Counter()
        
        left = 0
        for right in range(len(s2)):
            window_count[s2[right]] += 1
            
            if right - left + 1 > len(s1):
                if window_count[s2[left]] == 1:
                    del window_count[s2[left]]
                else:
                    window_count[s2[left]] -= 1
                left += 1
            
            if window_count == s1_count:
                return True
        
        return False
    
    # ==========================================
    # 4. PERMUTATION GENERATION
    # ==========================================
    
    def permutations_recursive(self, s: str) -> List[str]:
        """Generate all permutations recursively - Time: O(n!*n)"""
        if len(s) <= 1:
            return [s]
        
        result = []
        for i, char in enumerate(s):
            remaining = s[:i] + s[i+1:]
            for perm in self.permutations_recursive(remaining):
                result.append(char + perm)
        
        return result
    
    def permutations_iterative(self, s: str) -> List[str]:
        """Generate all permutations iteratively"""
        from itertools import permutations
        
        perms = permutations(s)
        return [''.join(p) for p in perms]
    
    def unique_permutations(self, s: str) -> List[str]:
        """Generate unique permutations for strings with duplicate characters"""
        def backtrack(path: str, remaining: List[str]):
            if not remaining:
                result.append(path)
                return
            
            seen = set()
            for i, char in enumerate(remaining):
                if char not in seen:
                    seen.add(char)
                    new_remaining = remaining[:i] + remaining[i+1:]
                    backtrack(path + char, new_remaining)
        
        result = []
        backtrack("", list(s))
        return result
    
    def next_permutation(self, s: str) -> str:
        """LC 31: Next Permutation - Time: O(n), Space: O(n)"""
        chars = list(s)
        n = len(chars)
        
        # Find first decreasing element from right
        i = n - 2
        while i >= 0 and chars[i] >= chars[i + 1]:
            i -= 1
        
        if i == -1:
            return ''.join(sorted(chars))
        
        # Find smallest element greater than chars[i]
        j = n - 1
        while chars[j] <= chars[i]:
            j -= 1
        
        # Swap and reverse suffix
        chars[i], chars[j] = chars[j], chars[i]
        chars[i + 1:] = reversed(chars[i + 1:])
        
        return ''.join(chars)
    
    # ==========================================
    # 5. ADVANCED ANAGRAM PROBLEMS
    # ==========================================
    
    def min_steps_anagram(self, s: str, t: str) -> int:
        """LC 1347: Minimum Steps to Make Two Strings Anagram"""
        if len(s) != len(t):
            return -1
        
        s_count = Counter(s)
        t_count = Counter(t)
        
        steps = 0
        for char in s_count:
            if s_count[char] > t_count.get(char, 0):
                steps += s_count[char] - t_count.get(char, 0)
        
        return steps
    
    def find_anagram_mappings(self, A: List[int], B: List[int]) -> List[int]:
        """LC 760: Find Anagram Mappings"""
        index_map = defaultdict(list)
        
        # Build index map for B
        for i, num in enumerate(B):
            index_map[num].append(i)
        
        result = []
        for num in A:
            result.append(index_map[num].pop())
        
        return result
    
    def check_anagram_with_deletions(self, s: str, t: str) -> bool:
        """Check if t is anagram of subsequence of s"""
        s_count = Counter(s)
        t_count = Counter(t)
        
        for char, count in t_count.items():
            if s_count[char] < count:
                return False
        
        return True
    
    # ==========================================
    # 6. PALINDROME ANAGRAMS
    # ==========================================
    
    def can_permute_palindrome(self, s: str) -> bool:
        """LC 266: Palindrome Permutation - Time: O(n), Space: O(1)"""
        char_count = Counter(s)
        odd_count = sum(1 for count in char_count.values() if count % 2 == 1)
        return odd_count <= 1
    
    def generate_palindrome_permutations(self, s: str) -> List[str]:
        """LC 267: Palindrome Permutation II"""
        char_count = Counter(s)
        odd_chars = [char for char, count in char_count.items() if count % 2 == 1]
        
        if len(odd_chars) > 1:
            return []
        
        # Build half of palindrome
        half = []
        middle = ""
        
        for char, count in char_count.items():
            half.extend([char] * (count // 2))
            if count % 2 == 1:
                middle = char
        
        def backtrack(path: List[str], remaining: List[str]):
            if not remaining:
                palindrome = ''.join(path) + middle + ''.join(reversed(path))
                result.append(palindrome)
                return
            
            seen = set()
            for i, char in enumerate(remaining):
                if char not in seen:
                    seen.add(char)
                    path.append(char)
                    new_remaining = remaining[:i] + remaining[i+1:]
                    backtrack(path, new_remaining)
                    path.pop()
        
        result = []
        backtrack([], half)
        return result
    
    # ==========================================
    # 7. SPECIAL ANAGRAM PROBLEMS
    # ==========================================
    
    def longest_anagram_substring(self, s1: str, s2: str) -> int:
        """Find length of longest substring that can form anagram with s2"""
        if not s1 or not s2:
            return 0
        
        s2_count = Counter(s2)
        max_length = 0
        
        # Try all substrings of s1
        for i in range(len(s1)):
            window_count = Counter()
            
            for j in range(i, len(s1)):
                window_count[s1[j]] += 1
                
                # Check if current window can form anagram with s2
                if self.can_form_anagram(window_count, s2_count):
                    max_length = max(max_length, j - i + 1)
        
        return max_length
    
    def can_form_anagram(self, count1: Counter, count2: Counter) -> bool:
        """Check if count1 can form anagram with count2"""
        for char, count in count1.items():
            if count > count2.get(char, 0):
                return False
        return True
    
    def min_window_anagram(self, s: str, t: str) -> str:
        """Find minimum window in s that contains anagram of t"""
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

# Test Examples
def run_examples():
    sap = StringAnagramsPermutations()
    
    print("=== ANAGRAMS AND PERMUTATIONS EXAMPLES ===\n")
    
    # Basic anagrams
    print("1. BASIC ANAGRAMS:")
    print(f"Is 'listen' anagram of 'silent': {sap.is_anagram_counting('listen', 'silent')}")
    
    # Group anagrams
    print(f"Grouped anagrams: {sap.group_anagrams(['eat', 'tea', 'tan', 'ate', 'nat', 'bat'])}")
    
    # Find anagrams
    print(f"Find anagrams 'ab' in 'abab': {sap.find_anagrams('abab', 'ab')}")
    print(f"Permutation in string: {sap.check_inclusion('ab', 'eidbaooo')}")
    
    # Permutations
    print(f"Permutations of 'abc': {sap.permutations_recursive('abc')}")
    print(f"Next permutation of '123': '{sap.next_permutation('123')}'")
    
    # Palindrome permutations
    print(f"Can 'aab' form palindrome: {sap.can_permute_palindrome('aab')}")
    
    # Advanced problems
    print(f"Min window anagram: '{sap.min_window_anagram('ADOBECODEBANC', 'ABC')}'")

if __name__ == "__main__":
    run_examples() 