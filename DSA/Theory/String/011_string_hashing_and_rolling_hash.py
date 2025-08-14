"""
String Hashing and Rolling Hash - Advanced Techniques
====================================================

Topics: Rolling hash, polynomial hashing, hash collisions
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Medium to Hard
"""

from typing import List, Set, Dict, Tuple
import random

class StringHashingRollingHash:
    
    def __init__(self):
        self.base = 256
        self.prime = 1000000007  # Large prime
        self.base2 = 31  # Alternative base
        self.prime2 = 1000000009  # Alternative prime
    
    # ==========================================
    # 1. BASIC POLYNOMIAL HASHING
    # ==========================================
    
    def polynomial_hash(self, s: str, base: int = None, prime: int = None) -> int:
        """Compute polynomial hash of string"""
        if base is None:
            base = self.base
        if prime is None:
            prime = self.prime
        
        hash_value = 0
        power = 1
        
        for char in s:
            hash_value = (hash_value + ord(char) * power) % prime
            power = (power * base) % prime
        
        return hash_value
    
    def polynomial_hash_prefix(self, s: str) -> List[int]:
        """Compute prefix hashes for all prefixes"""
        n = len(s)
        prefix_hash = [0] * (n + 1)
        power = [1] * (n + 1)
        
        for i in range(n):
            prefix_hash[i + 1] = (prefix_hash[i] + ord(s[i]) * power[i]) % self.prime
            power[i + 1] = (power[i] * self.base) % self.prime
        
        return prefix_hash, power
    
    def substring_hash(self, prefix_hash: List[int], power: List[int], 
                      left: int, right: int) -> int:
        """Get hash of substring s[left:right+1] using precomputed values"""
        hash_val = (prefix_hash[right + 1] - prefix_hash[left]) % self.prime
        hash_val = (hash_val * pow(power[left], self.prime - 2, self.prime)) % self.prime
        return hash_val
    
    # ==========================================
    # 2. ROLLING HASH IMPLEMENTATION
    # ==========================================
    
    class RollingHash:
        def __init__(self, s: str, window_size: int, base: int = 256, prime: int = 1000000007):
            self.s = s
            self.window_size = window_size
            self.base = base
            self.prime = prime
            self.base_power = pow(base, window_size - 1, prime)
            
            # Compute initial hash
            self.current_hash = 0
            for i in range(window_size):
                self.current_hash = (self.current_hash * base + ord(s[i])) % prime
            
            self.start = 0
        
        def roll(self) -> int:
            """Roll the hash window by one position"""
            if self.start + self.window_size >= len(self.s):
                return None
            
            # Remove leftmost character
            old_char = ord(self.s[self.start])
            self.current_hash = (self.current_hash - old_char * self.base_power) % self.prime
            
            # Add new character
            new_char = ord(self.s[self.start + self.window_size])
            self.current_hash = (self.current_hash * self.base + new_char) % self.prime
            
            self.start += 1
            return self.current_hash
        
        def get_hash(self) -> int:
            return self.current_hash
    
    # ==========================================
    # 3. DUPLICATE SUBSTRING DETECTION
    # ==========================================
    
    def find_all_duplicates(self, s: str, length: int) -> List[str]:
        """Find all duplicate substrings of given length using rolling hash"""
        if length > len(s):
            return []
        
        seen_hashes = set()
        duplicates = set()
        
        rolling_hash = self.RollingHash(s, length, self.base, self.prime)
        
        # Check first window
        hash_val = rolling_hash.get_hash()
        if hash_val in seen_hashes:
            duplicates.add(s[0:length])
        seen_hashes.add(hash_val)
        
        # Roll through remaining windows
        for i in range(len(s) - length):
            hash_val = rolling_hash.roll()
            substring = s[i + 1:i + 1 + length]
            
            if hash_val in seen_hashes:
                duplicates.add(substring)
            seen_hashes.add(hash_val)
        
        return list(duplicates)
    
    def longest_duplicate_substring(self, s: str) -> str:
        """LC 1044: Longest Duplicate Substring using binary search + rolling hash"""
        def has_duplicate(length: int) -> str:
            if length == 0:
                return ""
            
            seen = {}
            rolling_hash = self.RollingHash(s, length, self.base, self.prime)
            
            # Check first window
            hash_val = rolling_hash.get_hash()
            seen[hash_val] = 0
            
            # Roll through remaining windows
            for i in range(len(s) - length):
                hash_val = rolling_hash.roll()
                if hash_val in seen:
                    # Verify to avoid hash collision
                    start1 = seen[hash_val]
                    start2 = i + 1
                    if s[start1:start1 + length] == s[start2:start2 + length]:
                        return s[start1:start1 + length]
                seen[hash_val] = i + 1
            
            return ""
        
        # Binary search on length
        left, right = 0, len(s) - 1
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
    # 4. DOUBLE HASHING FOR COLLISION AVOIDANCE
    # ==========================================
    
    def double_hash(self, s: str) -> Tuple[int, int]:
        """Compute double hash to reduce collision probability"""
        hash1 = self.polynomial_hash(s, self.base, self.prime)
        hash2 = self.polynomial_hash(s, self.base2, self.prime2)
        return (hash1, hash2)
    
    def find_duplicates_double_hash(self, s: str, length: int) -> List[str]:
        """Find duplicates using double hashing"""
        if length > len(s):
            return []
        
        seen_hashes = set()
        duplicates = set()
        
        for i in range(len(s) - length + 1):
            substring = s[i:i + length]
            hash_pair = self.double_hash(substring)
            
            if hash_pair in seen_hashes:
                duplicates.add(substring)
            seen_hashes.add(hash_pair)
        
        return list(duplicates)
    
    # ==========================================
    # 5. PATTERN MATCHING WITH HASHING
    # ==========================================
    
    def rabin_karp_multiple_patterns(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """Rabin-Karp for multiple patterns"""
        if not patterns:
            return {}
        
        # Compute pattern hashes
        pattern_hashes = {}
        max_len = max(len(p) for p in patterns)
        
        for pattern in patterns:
            pattern_hash = self.polynomial_hash(pattern)
            if pattern_hash not in pattern_hashes:
                pattern_hashes[pattern_hash] = []
            pattern_hashes[pattern_hash].append(pattern)
        
        result = {pattern: [] for pattern in patterns}
        
        # Check all substrings
        for length in set(len(p) for p in patterns):
            if length > len(text):
                continue
            
            rolling_hash = self.RollingHash(text, length, self.base, self.prime)
            
            # Check first window
            hash_val = rolling_hash.get_hash()
            if hash_val in pattern_hashes:
                for pattern in pattern_hashes[hash_val]:
                    if len(pattern) == length and text[0:length] == pattern:
                        result[pattern].append(0)
            
            # Roll through remaining windows
            for i in range(len(text) - length):
                hash_val = rolling_hash.roll()
                if hash_val in pattern_hashes:
                    for pattern in pattern_hashes[hash_val]:
                        if len(pattern) == length and text[i+1:i+1+length] == pattern:
                            result[pattern].append(i + 1)
        
        return result
    
    # ==========================================
    # 6. ADVANCED APPLICATIONS
    # ==========================================
    
    def count_distinct_substrings(self, s: str) -> int:
        """Count number of distinct substrings using hashing"""
        seen_hashes = set()
        
        for length in range(1, len(s) + 1):
            for i in range(len(s) - length + 1):
                substring = s[i:i + length]
                hash_val = self.polynomial_hash(substring)
                seen_hashes.add(hash_val)
        
        return len(seen_hashes)
    
    def longest_common_substring_hash(self, s1: str, s2: str) -> str:
        """Find longest common substring using binary search + hashing"""
        def has_common_substring(length: int) -> str:
            if length == 0:
                return ""
            
            # Get all substrings of s1 with given length
            s1_hashes = set()
            for i in range(len(s1) - length + 1):
                substring = s1[i:i + length]
                hash_val = self.polynomial_hash(substring)
                s1_hashes.add((hash_val, substring))
            
            # Check substrings of s2
            for i in range(len(s2) - length + 1):
                substring = s2[i:i + length]
                hash_val = self.polynomial_hash(substring)
                
                for h, s1_sub in s1_hashes:
                    if h == hash_val and s1_sub == substring:
                        return substring
            
            return ""
        
        # Binary search on length
        left, right = 0, min(len(s1), len(s2))
        result = ""
        
        while left <= right:
            mid = (left + right) // 2
            common = has_common_substring(mid)
            
            if common:
                result = common
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    def find_anagram_groups_hash(self, words: List[str]) -> List[List[str]]:
        """Group anagrams using character frequency hashing"""
        def char_hash(word: str) -> int:
            # Use prime numbers for each character
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
                     53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
            
            hash_val = 1
            for char in word:
                hash_val *= primes[ord(char) - ord('a')]
            
            return hash_val
        
        groups = {}
        for word in words:
            hash_val = char_hash(word)
            if hash_val not in groups:
                groups[hash_val] = []
            groups[hash_val].append(word)
        
        return list(groups.values())
    
    def is_rotation_hash(self, s1: str, s2: str) -> bool:
        """Check if s2 is rotation of s1 using hashing"""
        if len(s1) != len(s2):
            return False
        
        if len(s1) == 0:
            return True
        
        # s2 is rotation of s1 if s2 is substring of s1+s1
        combined = s1 + s1
        
        # Use rolling hash to find s2 in combined
        if len(s2) > len(combined):
            return False
        
        target_hash = self.polynomial_hash(s2)
        rolling_hash = self.RollingHash(combined, len(s2), self.base, self.prime)
        
        # Check first window
        if rolling_hash.get_hash() == target_hash:
            if combined[0:len(s2)] == s2:
                return True
        
        # Roll through remaining windows
        for i in range(len(combined) - len(s2)):
            hash_val = rolling_hash.roll()
            if hash_val == target_hash:
                if combined[i+1:i+1+len(s2)] == s2:
                    return True
        
        return False

# Test Examples
def run_examples():
    shrh = StringHashingRollingHash()
    
    print("=== STRING HASHING AND ROLLING HASH EXAMPLES ===\n")
    
    # Basic hashing
    print("1. BASIC HASHING:")
    hash_val = shrh.polynomial_hash("hello")
    print(f"Polynomial hash of 'hello': {hash_val}")
    
    double_hash = shrh.double_hash("hello")
    print(f"Double hash of 'hello': {double_hash}")
    
    # Rolling hash
    print("\n2. ROLLING HASH:")
    rolling = shrh.RollingHash("abcdef", 3)
    print(f"Initial hash ('abc'): {rolling.get_hash()}")
    print(f"After roll ('bcd'): {rolling.roll()}")
    print(f"After roll ('cde'): {rolling.roll()}")
    
    # Duplicate detection
    print("\n3. DUPLICATE DETECTION:")
    duplicates = shrh.find_all_duplicates("abcabc", 3)
    print(f"Duplicates of length 3 in 'abcabc': {duplicates}")
    
    lds = shrh.longest_duplicate_substring("banana")
    print(f"Longest duplicate in 'banana': '{lds}'")
    
    # Pattern matching
    print("\n4. PATTERN MATCHING:")
    patterns = ["ab", "bc", "ca"]
    matches = shrh.rabin_karp_multiple_patterns("abcabc", patterns)
    print(f"Multiple pattern matches: {matches}")
    
    # Advanced applications
    print("\n5. ADVANCED APPLICATIONS:")
    distinct_count = shrh.count_distinct_substrings("abc")
    print(f"Distinct substrings in 'abc': {distinct_count}")
    
    lcs = shrh.longest_common_substring_hash("abcdef", "cdefgh")
    print(f"Longest common substring: '{lcs}'")
    
    anagram_groups = shrh.find_anagram_groups_hash(["eat", "tea", "tan", "ate", "nat", "bat"])
    print(f"Anagram groups: {anagram_groups}")
    
    is_rotation = shrh.is_rotation_hash("abcde", "cdeab")
    print(f"Is 'cdeab' rotation of 'abcde': {is_rotation}")

if __name__ == "__main__":
    run_examples() 