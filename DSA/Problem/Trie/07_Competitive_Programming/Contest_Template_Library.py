"""
Contest Template Library - Multiple Approaches
Difficulty: Hard

Template library for competitive programming with trie-based solutions.
Ready-to-use implementations for common contest problems.

Templates:
1. Fast Trie Implementation
2. Aho-Corasick Template
3. String Hashing with Trie
4. Suffix Array Template
5. Contest I/O Templates
6. Common Algorithm Templates
"""

import sys
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque

class FastTrie:
    """Fast trie implementation for contests"""
    __slots__ = ['children', 'is_end', 'count']
    
    def __init__(self):
        self.children = [None] * 26  # For lowercase letters
        self.is_end = False
        self.count = 0
    
    def insert(self, word: str) -> None:
        node = self
        for char in word:
            idx = ord(char) - ord('a')
            if node.children[idx] is None:
                node.children[idx] = FastTrie()
            node = node.children[idx]
            node.count += 1
        node.is_end = True
    
    def search(self, word: str) -> bool:
        node = self
        for char in word:
            idx = ord(char) - ord('a')
            if node.children[idx] is None:
                return False
            node = node.children[idx]
        return node.is_end
    
    def count_prefix(self, prefix: str) -> int:
        node = self
        for char in prefix:
            idx = ord(char) - ord('a')
            if node.children[idx] is None:
                return 0
            node = node.children[idx]
        return node.count

class AhoCorasick:
    """Aho-Corasick template for multiple pattern matching"""
    
    def __init__(self):
        self.trie = [{}]  # trie[node][char] = next_node
        self.fail = [0]   # failure links
        self.output = [[]]  # output for each node
        self.node_count = 1
    
    def add_pattern(self, pattern: str, pattern_id: int) -> None:
        node = 0
        for char in pattern:
            if char not in self.trie[node]:
                self.trie.append({})
                self.fail.append(0)
                self.output.append([])
                self.trie[node][char] = self.node_count
                self.node_count += 1
            node = self.trie[node][char]
        self.output[node].append(pattern_id)
    
    def build_failure_links(self) -> None:
        queue = deque()
        
        # Initialize first level
        for char, child in self.trie[0].items():
            queue.append(child)
        
        while queue:
            current = queue.popleft()
            
            for char, child in self.trie[current].items():
                queue.append(child)
                
                # Find failure link
                fail_node = self.fail[current]
                while fail_node != 0 and char not in self.trie[fail_node]:
                    fail_node = self.fail[fail_node]
                
                if char in self.trie[fail_node] and self.trie[fail_node][char] != child:
                    self.fail[child] = self.trie[fail_node][char]
                
                # Add output from failure link
                self.output[child].extend(self.output[self.fail[child]])
    
    def search_text(self, text: str) -> List[Tuple[int, int]]:
        """Returns list of (position, pattern_id) pairs"""
        results = []
        node = 0
        
        for i, char in enumerate(text):
            # Follow failure links
            while node != 0 and char not in self.trie[node]:
                node = self.fail[node]
            
            if char in self.trie[node]:
                node = self.trie[node][char]
            
            # Record all patterns ending at this position
            for pattern_id in self.output[node]:
                results.append((i, pattern_id))
        
        return results

class StringHash:
    """Rolling hash template"""
    
    def __init__(self, text: str, base: int = 31, mod: int = 10**9 + 7):
        self.text = text
        self.base = base
        self.mod = mod
        self.n = len(text)
        
        # Precompute hashes and powers
        self.hash_vals = [0] * (self.n + 1)
        self.powers = [1] * (self.n + 1)
        
        for i in range(self.n):
            self.hash_vals[i + 1] = (self.hash_vals[i] * base + ord(text[i])) % mod
            self.powers[i + 1] = (self.powers[i] * base) % mod
    
    def get_hash(self, left: int, right: int) -> int:
        """Get hash of substring text[left:right+1]"""
        result = (self.hash_vals[right + 1] - self.hash_vals[left] * self.powers[right - left + 1]) % self.mod
        return (result + self.mod) % self.mod

class ContestIO:
    """Fast I/O template for contests"""
    
    def __init__(self):
        self.input_lines = []
        self.input_index = 0
    
    def read_all_input(self) -> None:
        self.input_lines = sys.stdin.read().strip().split('\n')
    
    def next_line(self) -> str:
        if self.input_index < len(self.input_lines):
            line = self.input_lines[self.input_index]
            self.input_index += 1
            return line
        return ""
    
    def next_int(self) -> int:
        return int(self.next_line())
    
    def next_ints(self) -> List[int]:
        return list(map(int, self.next_line().split()))
    
    def next_strings(self) -> List[str]:
        return self.next_line().split()

class ContestTemplates:
    """Collection of contest templates"""
    
    @staticmethod
    def max_xor_subarray(arr: List[int]) -> int:
        """Template: Maximum XOR subarray"""
        class XORTrie:
            def __init__(self):
                self.root = {}
            
            def insert(self, num: int) -> None:
                node = self.root
                for i in range(31, -1, -1):
                    bit = (num >> i) & 1
                    if bit not in node:
                        node[bit] = {}
                    node = node[bit]
            
            def max_xor(self, num: int) -> int:
                node = self.root
                result = 0
                for i in range(31, -1, -1):
                    bit = (num >> i) & 1
                    opposite = 1 - bit
                    
                    if opposite in node:
                        result |= (1 << i)
                        node = node[opposite]
                    else:
                        node = node[bit]
                
                return result
        
        trie = XORTrie()
        trie.insert(0)
        
        max_xor = 0
        prefix_xor = 0
        
        for num in arr:
            prefix_xor ^= num
            max_xor = max(max_xor, trie.max_xor(prefix_xor))
            trie.insert(prefix_xor)
        
        return max_xor
    
    @staticmethod
    def multiple_string_matching(text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """Template: Multiple string matching using Aho-Corasick"""
        ac = AhoCorasick()
        
        # Add patterns
        for i, pattern in enumerate(patterns):
            ac.add_pattern(pattern, i)
        
        ac.build_failure_links()
        
        # Search
        matches = ac.search_text(text)
        
        # Group by pattern
        result = defaultdict(list)
        for pos, pattern_id in matches:
            pattern = patterns[pattern_id]
            start_pos = pos - len(pattern) + 1
            result[pattern].append(start_pos)
        
        return dict(result)
    
    @staticmethod
    def longest_common_prefix_array(strings: List[str]) -> List[int]:
        """Template: LCP array using trie"""
        if not strings:
            return []
        
        # Build trie
        trie = FastTrie()
        for s in strings:
            trie.insert(s)
        
        # Calculate LCP
        lcp = []
        for i in range(len(strings) - 1):
            s1, s2 = strings[i], strings[i + 1]
            common_len = 0
            
            for j in range(min(len(s1), len(s2))):
                if s1[j] == s2[j]:
                    common_len += 1
                else:
                    break
            
            lcp.append(common_len)
        
        return lcp
    
    @staticmethod
    def digit_dp_no_adjacent_same(n: str) -> int:
        """Template: Digit DP - count numbers <= n with no adjacent same digits"""
        MOD = 10**9 + 7
        memo = {}
        
        def dp(pos: int, prev_digit: int, tight: bool, started: bool) -> int:
            if pos == len(n):
                return 1 if started else 0
            
            state = (pos, prev_digit, tight, started)
            if state in memo:
                return memo[state]
            
            limit = int(n[pos]) if tight else 9
            result = 0
            
            for digit in range(0, limit + 1):
                if not started and digit == 0:
                    # Leading zero
                    result += dp(pos + 1, -1, tight and digit == limit, False)
                elif started and digit == prev_digit:
                    # Adjacent same digit - skip
                    continue
                else:
                    result += dp(pos + 1, digit, tight and digit == limit, True)
            
            memo[state] = result % MOD
            return memo[state]
        
        return dp(0, -1, True, False)
    
    @staticmethod
    def z_algorithm(s: str) -> List[int]:
        """Template: Z-algorithm for string matching"""
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

def contest_main():
    """Main template for contest problems"""
    # Fast I/O setup
    io = ContestIO()
    io.read_all_input()
    
    # Example usage:
    # n = io.next_int()
    # arr = io.next_ints()
    # strings = io.next_strings()
    
    # Example problem: Maximum XOR subarray
    n = io.next_int()
    arr = io.next_ints()
    
    result = ContestTemplates.max_xor_subarray(arr)
    print(result)

def demonstrate_templates():
    """Demonstrate template usage"""
    print("=== Contest Template Demonstrations ===")
    
    # 1. Fast Trie
    print("1. Fast Trie:")
    trie = FastTrie()
    words = ["cat", "car", "card", "care", "careful"]
    
    for word in words:
        trie.insert(word)
    
    print(f"   Words with prefix 'car': {trie.count_prefix('car')}")
    print(f"   Search 'card': {trie.search('card')}")
    
    # 2. Aho-Corasick
    print("\n2. Multiple Pattern Matching:")
    text = "ushersheishishe"
    patterns = ["he", "she", "his", "hers"]
    
    matches = ContestTemplates.multiple_string_matching(text, patterns)
    for pattern, positions in matches.items():
        print(f"   '{pattern}': {positions}")
    
    # 3. String Hashing
    print("\n3. String Hashing:")
    text = "abcdefg"
    hasher = StringHash(text)
    
    print(f"   Hash of 'abc': {hasher.get_hash(0, 2)}")
    print(f"   Hash of 'def': {hasher.get_hash(3, 5)}")
    
    # 4. Maximum XOR Subarray
    print("\n4. Maximum XOR Subarray:")
    arr = [1, 2, 3, 4]
    max_xor = ContestTemplates.max_xor_subarray(arr)
    print(f"   Array {arr}: max XOR = {max_xor}")
    
    # 5. Z-Algorithm
    print("\n5. Z-Algorithm:")
    s = "ababcababa"
    z_array = ContestTemplates.z_algorithm(s)
    print(f"   String '{s}': Z = {z_array}")

if __name__ == "__main__":
    demonstrate_templates()
    # Uncomment for actual contest:
    # contest_main()
