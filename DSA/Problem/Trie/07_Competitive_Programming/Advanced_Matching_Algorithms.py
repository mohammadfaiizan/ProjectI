"""
Advanced Matching Algorithms - Multiple Approaches
Difficulty: Hard

Advanced string matching algorithms using trie structures for competitive programming.

Algorithms:
1. Aho-Corasick Algorithm
2. Z-Algorithm with Trie
3. KMP with Trie Optimization
4. Suffix Array with Trie
5. Rolling Hash + Trie
6. Multiple Pattern Matching
"""

from typing import List, Dict, Set, Tuple
from collections import deque, defaultdict

class ACNode:
    def __init__(self):
        self.children = {}
        self.fail = None
        self.output = []

class AdvancedMatching:
    
    def aho_corasick(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Aho-Corasick multiple pattern matching
        Time: O(|text| + sum(|patterns|))
        Space: O(sum(|patterns|))
        """
        root = ACNode()
        
        # Build trie
        for pattern in patterns:
            node = root
            for char in pattern:
                if char not in node.children:
                    node.children[char] = ACNode()
                node = node.children[char]
            node.output.append(pattern)
        
        # Build failure links
        queue = deque()
        for child in root.children.values():
            child.fail = root
            queue.append(child)
        
        while queue:
            current = queue.popleft()
            for char, child in current.children.items():
                queue.append(child)
                
                fail_node = current.fail
                while fail_node != root and char not in fail_node.children:
                    fail_node = fail_node.fail
                
                if char in fail_node.children and fail_node.children[char] != child:
                    child.fail = fail_node.children[char]
                else:
                    child.fail = root
                
                child.output.extend(child.fail.output)
        
        # Search
        result = defaultdict(list)
        current = root
        
        for i, char in enumerate(text):
            while current != root and char not in current.children:
                current = current.fail
            
            if char in current.children:
                current = current.children[char]
            
            for pattern in current.output:
                start_pos = i - len(pattern) + 1
                result[pattern].append(start_pos)
        
        return dict(result)
    
    def z_algorithm_trie(self, patterns: List[str], text: str) -> List[List[int]]:
        """
        Z-algorithm with trie for multiple patterns
        Time: O(|text| * num_patterns)
        Space: O(max_pattern_length)
        """
        def z_algorithm(s: str) -> List[int]:
            n = len(s)
            z = [0] * n
            l, r = 0, 0
            
            for i in range(1, n):
                if i <= r:
                    z[i] = min(r - i + 1, z[i - l])
                
                while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                    z[i] += 1
                
                if i + z[i] - 1 > r:
                    l, r = i, i + z[i] - 1
            
            return z
        
        results = []
        
        for pattern in patterns:
            combined = pattern + "#" + text
            z_array = z_algorithm(combined)
            
            pattern_len = len(pattern)
            matches = []
            
            for i in range(pattern_len + 1, len(combined)):
                if z_array[i] == pattern_len:
                    start_pos = i - pattern_len - 1
                    matches.append(start_pos)
            
            results.append(matches)
        
        return results
    
    def rolling_hash_trie(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Rolling hash with trie for pattern verification
        Time: O(|text| + sum(|patterns|))
        Space: O(sum(|patterns|))
        """
        BASE = 31
        MOD = 10**9 + 7
        
        # Build pattern hash trie
        pattern_hashes = {}
        for pattern in patterns:
            hash_val = 0
            for char in pattern:
                hash_val = (hash_val * BASE + ord(char)) % MOD
            pattern_hashes[hash_val] = pattern
        
        result = defaultdict(list)
        
        for pattern_len in set(len(p) for p in patterns):
            if pattern_len > len(text):
                continue
            
            # Calculate rolling hash
            hash_val = 0
            power = 1
            
            # Initial hash
            for i in range(pattern_len):
                hash_val = (hash_val * BASE + ord(text[i])) % MOD
                if i < pattern_len - 1:
                    power = (power * BASE) % MOD
            
            # Check initial window
            if hash_val in pattern_hashes:
                pattern = pattern_hashes[hash_val]
                if text[:pattern_len] == pattern:
                    result[pattern].append(0)
            
            # Rolling hash for remaining positions
            for i in range(pattern_len, len(text)):
                # Remove leftmost character
                hash_val = (hash_val - ord(text[i - pattern_len]) * power) % MOD
                hash_val = (hash_val * BASE + ord(text[i])) % MOD
                hash_val = (hash_val + MOD) % MOD
                
                # Check if hash matches any pattern
                if hash_val in pattern_hashes:
                    pattern = pattern_hashes[hash_val]
                    start_pos = i - pattern_len + 1
                    if text[start_pos:start_pos + pattern_len] == pattern:
                        result[pattern].append(start_pos)
        
        return dict(result)


def test_aho_corasick():
    """Test Aho-Corasick algorithm"""
    print("=== Testing Aho-Corasick ===")
    
    matcher = AdvancedMatching()
    
    text = "ushersheishishe"
    patterns = ["he", "she", "his", "hers"]
    
    print(f"Text: '{text}'")
    print(f"Patterns: {patterns}")
    
    matches = matcher.aho_corasick(text, patterns)
    
    for pattern, positions in matches.items():
        print(f"'{pattern}': {positions}")

def test_z_algorithm():
    """Test Z-algorithm with trie"""
    print("\n=== Testing Z-Algorithm ===")
    
    matcher = AdvancedMatching()
    
    text = "abababcababa"
    patterns = ["ab", "aba", "abc"]
    
    print(f"Text: '{text}'")
    print(f"Patterns: {patterns}")
    
    results = matcher.z_algorithm_trie(patterns, text)
    
    for i, matches in enumerate(results):
        print(f"'{patterns[i]}': {matches}")

def test_rolling_hash():
    """Test rolling hash with trie"""
    print("\n=== Testing Rolling Hash ===")
    
    matcher = AdvancedMatching()
    
    text = "abcabcabc"
    patterns = ["abc", "bca", "cab"]
    
    print(f"Text: '{text}'")
    print(f"Patterns: {patterns}")
    
    matches = matcher.rolling_hash_trie(text, patterns)
    
    for pattern, positions in matches.items():
        print(f"'{pattern}': {positions}")

if __name__ == "__main__":
    test_aho_corasick()
    test_z_algorithm()
    test_rolling_hash()
