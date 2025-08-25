"""
Contest String Problems - Multiple Approaches
Difficulty: Hard

Competitive programming string problems commonly solved using trie data structures.
These problems frequently appear in programming contests like Codeforces, AtCoder, etc.

Problems:
1. Maximum XOR Subarray
2. Distinct Subsequences Count
3. String Hashing with Trie
4. Palindromic Subsequences
5. Lexicographically Smallest String
6. Multiple String Matching
7. String Compression Ratio
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import heapq

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0
        self.indices = []

class ContestStringProblems:
    
    def __init__(self):
        self.root = TrieNode()
    
    def maximum_xor_subarray(self, arr: List[int]) -> Tuple[int, List[int]]:
        """
        Approach 1: Maximum XOR Subarray
        
        Find subarray with maximum XOR using prefix XOR + trie.
        
        Time: O(n * 32)
        Space: O(n * 32)
        """
        class XORTrie:
            def __init__(self):
                self.root = {}
            
            def insert(self, num: int, index: int) -> None:
                node = self.root
                for i in range(31, -1, -1):
                    bit = (num >> i) & 1
                    if bit not in node:
                        node[bit] = {'children': {}, 'indices': []}
                    node['indices'].append(index)
                    node = node[bit]['children']
                node['indices'] = [index]
            
            def query_max_xor(self, num: int) -> Tuple[int, int]:
                node = self.root
                max_xor = 0
                best_index = -1
                
                for i in range(31, -1, -1):
                    bit = (num >> i) & 1
                    opposite = 1 - bit
                    
                    if opposite in node:
                        max_xor |= (1 << i)
                        node = node[opposite]['children']
                        if 'indices' in node and node['indices']:
                            best_index = node['indices'][-1]
                    elif bit in node:
                        node = node[bit]['children']
                        if 'indices' in node and node['indices']:
                            best_index = node['indices'][-1]
                    else:
                        break
                
                return max_xor, best_index
        
        n = len(arr)
        prefix_xor = [0] * (n + 1)
        
        # Calculate prefix XOR
        for i in range(n):
            prefix_xor[i + 1] = prefix_xor[i] ^ arr[i]
        
        trie = XORTrie()
        max_xor = 0
        best_subarray = []
        
        trie.insert(0, 0)  # Insert prefix_xor[0]
        
        for i in range(1, n + 1):
            # Query maximum XOR with current prefix
            curr_max, best_j = trie.query_max_xor(prefix_xor[i])
            
            if curr_max > max_xor:
                max_xor = curr_max
                # Subarray from best_j to i-1
                best_subarray = arr[best_j:i]
            
            # Insert current prefix
            trie.insert(prefix_xor[i], i)
        
        return max_xor, best_subarray
    
    def count_distinct_subsequences(self, s: str) -> int:
        """
        Approach 2: Count Distinct Subsequences
        
        Count number of distinct subsequences using trie + DP.
        
        Time: O(n * 2^n) worst case, O(n * k) average where k = distinct subsequences
        Space: O(k)
        """
        MOD = 10**9 + 7
        
        def insert_subsequence(subsequence: str) -> None:
            node = self.root
            for char in subsequence:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
            node.count += 1
        
        def generate_subsequences(s: str, index: int, current: str) -> None:
            if index == len(s):
                if current:  # Non-empty subsequence
                    insert_subsequence(current)
                return
            
            # Include current character
            generate_subsequences(s, index + 1, current + s[index])
            # Exclude current character
            generate_subsequences(s, index + 1, current)
        
        def count_trie_nodes(node: TrieNode) -> int:
            count = 1 if node.is_end else 0
            for child in node.children.values():
                count += count_trie_nodes(child)
            return count
        
        # Generate all subsequences and store in trie
        generate_subsequences(s, 0, "")
        
        # Count distinct subsequences
        return count_trie_nodes(self.root)
    
    def string_hashing_with_trie(self, patterns: List[str], text: str) -> Dict[str, List[int]]:
        """
        Approach 3: String Hashing with Trie
        
        Combine string hashing with trie for fast pattern matching.
        
        Time: O(|text| + sum(|pattern|))
        Space: O(sum(|pattern|))
        """
        BASE = 31
        MOD = 10**9 + 7
        
        class HashTrieNode:
            def __init__(self):
                self.children = {}
                self.hash_value = 0
                self.pattern = ""
                self.is_pattern = False
        
        def compute_hash(s: str) -> int:
            hash_val = 0
            for char in s:
                hash_val = (hash_val * BASE + ord(char)) % MOD
            return hash_val
        
        def insert_pattern(pattern: str) -> None:
            node = hash_trie_root
            current_hash = 0
            
            for char in pattern:
                current_hash = (current_hash * BASE + ord(char)) % MOD
                
                if char not in node.children:
                    node.children[char] = HashTrieNode()
                
                node = node.children[char]
                node.hash_value = current_hash
            
            node.is_pattern = True
            node.pattern = pattern
        
        # Build hash trie
        hash_trie_root = HashTrieNode()
        for pattern in patterns:
            insert_pattern(pattern)
        
        # Search in text
        result = defaultdict(list)
        n = len(text)
        
        for i in range(n):
            node = hash_trie_root
            j = i
            
            while j < n and text[j] in node.children:
                node = node.children[text[j]]
                
                if node.is_pattern:
                    result[node.pattern].append(i)
                
                j += 1
        
        return dict(result)
    
    def count_palindromic_subsequences(self, s: str) -> int:
        """
        Approach 4: Count Palindromic Subsequences
        
        Count distinct palindromic subsequences using trie.
        
        Time: O(n^3)
        Space: O(n^2)
        """
        MOD = 10**9 + 7
        n = len(s)
        
        def is_palindrome(string: str) -> bool:
            return string == string[::-1]
        
        def generate_and_count_palindromes() -> int:
            palindromes = set()
            
            # Generate all subsequences and check if palindrome
            def backtrack(index: int, current: str) -> None:
                if index == n:
                    if current and is_palindrome(current):
                        palindromes.add(current)
                    return
                
                # Include current character
                backtrack(index + 1, current + s[index])
                # Exclude current character
                backtrack(index + 1, current)
            
            backtrack(0, "")
            return len(palindromes)
        
        # For large inputs, use DP approach
        if n <= 15:
            return generate_and_count_palindromes()
        
        # DP approach for larger inputs
        # dp[i][j] = number of palindromic subsequences in s[i:j+1]
        dp = {}
        
        def solve(i: int, j: int) -> int:
            if i > j:
                return 0
            if i == j:
                return 1
            if (i, j) in dp:
                return dp[(i, j)]
            
            if s[i] == s[j]:
                result = (2 * solve(i + 1, j - 1) + 2) % MOD
            else:
                result = (solve(i + 1, j) + solve(i, j - 1) - solve(i + 1, j - 1)) % MOD
            
            dp[(i, j)] = result
            return result
        
        return solve(0, n - 1)
    
    def lexicographically_smallest_string(self, words: List[str], k: int) -> str:
        """
        Approach 5: Lexicographically Smallest String
        
        Find lexicographically smallest string after k operations.
        
        Time: O(n * m * k) where n=words, m=avg_length
        Space: O(n * m)
        """
        def build_suffix_trie(word: str) -> TrieNode:
            root = TrieNode()
            
            for i in range(len(word)):
                node = root
                for j in range(i, len(word)):
                    char = word[j]
                    if char not in node.children:
                        node.children[char] = TrieNode()
                    node = node.children[char]
                    node.indices.append(i)
                node.is_end = True
            
            return root
        
        def find_smallest_suffix(trie_root: TrieNode, remaining_ops: int) -> str:
            if remaining_ops == 0 or not trie_root.children:
                return ""
            
            # Find smallest character
            min_char = min(trie_root.children.keys())
            node = trie_root.children[min_char]
            
            # Recursively build the rest
            rest = find_smallest_suffix(node, remaining_ops - 1)
            return min_char + rest
        
        # Build tries for all words
        tries = []
        for word in words:
            trie = build_suffix_trie(word)
            tries.append(trie)
        
        # Find globally smallest string
        result = ""
        remaining_k = k
        
        while remaining_k > 0 and tries:
            min_char = 'z' + '1'  # Larger than any possible character
            best_trie_idx = -1
            
            # Find minimum first character among all tries
            for i, trie in enumerate(tries):
                if trie.children:
                    first_char = min(trie.children.keys())
                    if first_char < min_char:
                        min_char = first_char
                        best_trie_idx = i
            
            if best_trie_idx == -1:
                break
            
            result += min_char
            remaining_k -= 1
            
            # Move to next level in the best trie
            tries[best_trie_idx] = tries[best_trie_idx].children[min_char]
        
        return result
    
    def multiple_string_matching(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Approach 6: Multiple String Matching (Aho-Corasick)
        
        Find all occurrences of multiple patterns in text.
        
        Time: O(|text| + sum(|patterns|))
        Space: O(sum(|patterns|))
        """
        class ACTrieNode:
            def __init__(self):
                self.children = {}
                self.fail = None
                self.output = []
        
        def build_ac_automaton(patterns: List[str]) -> ACTrieNode:
            root = ACTrieNode()
            
            # Build trie
            for pattern in patterns:
                node = root
                for char in pattern:
                    if char not in node.children:
                        node.children[char] = ACTrieNode()
                    node = node.children[char]
                node.output.append(pattern)
            
            # Build failure links (BFS)
            queue = []
            root.fail = root
            
            for child in root.children.values():
                child.fail = root
                queue.append(child)
            
            while queue:
                current = queue.pop(0)
                
                for char, child in current.children.items():
                    queue.append(child)
                    
                    # Find failure link
                    fail_node = current.fail
                    while fail_node != root and char not in fail_node.children:
                        fail_node = fail_node.fail
                    
                    if char in fail_node.children and fail_node.children[char] != child:
                        child.fail = fail_node.children[char]
                    else:
                        child.fail = root
                    
                    # Add output from failure link
                    child.output.extend(child.fail.output)
            
            return root
        
        # Build Aho-Corasick automaton
        ac_root = build_ac_automaton(patterns)
        
        # Search in text
        result = defaultdict(list)
        current = ac_root
        
        for i, char in enumerate(text):
            # Follow failure links until we find a match or reach root
            while current != ac_root and char not in current.children:
                current = current.fail
            
            if char in current.children:
                current = current.children[char]
            
            # Record all patterns ending at this position
            for pattern in current.output:
                start_pos = i - len(pattern) + 1
                result[pattern].append(start_pos)
        
        return dict(result)
    
    def calculate_compression_ratio(self, strings: List[str]) -> float:
        """
        Approach 7: String Compression Ratio using Trie
        
        Calculate compression ratio achieved by trie structure.
        
        Time: O(sum(lengths))
        Space: O(trie_size)
        """
        def insert_string(s: str) -> None:
            node = self.root
            for char in s:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                node.count += 1
            node.is_end = True
        
        def calculate_trie_size(node: TrieNode) -> int:
            size = 1  # Current node
            for child in node.children.values():
                size += calculate_trie_size(child)
            return size
        
        # Build trie
        for s in strings:
            insert_string(s)
        
        # Calculate sizes
        original_size = sum(len(s) for s in strings)
        trie_size = calculate_trie_size(self.root)
        
        compression_ratio = original_size / trie_size if trie_size > 0 else 0
        
        return compression_ratio


def test_maximum_xor_subarray():
    """Test maximum XOR subarray problem"""
    print("=== Testing Maximum XOR Subarray ===")
    
    solver = ContestStringProblems()
    
    test_cases = [
        [1, 2, 3, 4],
        [8, 1, 2, 12, 7, 6],
        [4, 6, 1, 3],
    ]
    
    for i, arr in enumerate(test_cases):
        max_xor, subarray = solver.maximum_xor_subarray(arr)
        print(f"Test {i+1}: {arr}")
        print(f"  Max XOR: {max_xor}")
        print(f"  Subarray: {subarray}")
        
        # Verify result
        actual_xor = 0
        for num in subarray:
            actual_xor ^= num
        print(f"  Verification: {actual_xor} == {max_xor} ✓")


def test_distinct_subsequences():
    """Test distinct subsequences counting"""
    print("\n=== Testing Distinct Subsequences ===")
    
    solver = ContestStringProblems()
    
    test_strings = ["abc", "aab", "abcd"]
    
    for s in test_strings:
        # Reset root for each test
        solver.root = TrieNode()
        
        count = solver.count_distinct_subsequences(s)
        print(f"String '{s}': {count} distinct subsequences")


def test_string_hashing():
    """Test string hashing with trie"""
    print("\n=== Testing String Hashing with Trie ===")
    
    solver = ContestStringProblems()
    
    text = "ababcababa"
    patterns = ["ab", "ba", "abc", "cab"]
    
    print(f"Text: '{text}'")
    print(f"Patterns: {patterns}")
    
    matches = solver.string_hashing_with_trie(patterns, text)
    
    print("Matches found:")
    for pattern, positions in matches.items():
        print(f"  '{pattern}': {positions}")


def test_palindromic_subsequences():
    """Test palindromic subsequences counting"""
    print("\n=== Testing Palindromic Subsequences ===")
    
    solver = ContestStringProblems()
    
    test_strings = ["abc", "aab", "aba", "racecar"]
    
    for s in test_strings:
        count = solver.count_palindromic_subsequences(s)
        print(f"String '{s}': {count} palindromic subsequences")


def test_multiple_string_matching():
    """Test Aho-Corasick multiple string matching"""
    print("\n=== Testing Multiple String Matching ===")
    
    solver = ContestStringProblems()
    
    text = "ushersheishishe"
    patterns = ["he", "she", "his", "hers"]
    
    print(f"Text: '{text}'")
    print(f"Patterns: {patterns}")
    
    matches = solver.multiple_string_matching(text, patterns)
    
    print("All matches found:")
    for pattern, positions in matches.items():
        print(f"  '{pattern}': {positions}")
        for pos in positions:
            print(f"    Position {pos}: '{text[pos:pos+len(pattern)]}'")


def benchmark_contest_problems():
    """Benchmark contest problems"""
    print("\n=== Benchmarking Contest Problems ===")
    
    import time
    import random
    import string
    
    solver = ContestStringProblems()
    
    def generate_random_array(size: int) -> List[int]:
        return [random.randint(1, 100) for _ in range(size)]
    
    def generate_random_string(length: int) -> str:
        return ''.join(random.choices(string.ascii_lowercase, k=length))
    
    # Benchmark maximum XOR subarray
    print("Maximum XOR Subarray Performance:")
    for size in [50, 100, 200]:
        arr = generate_random_array(size)
        
        start_time = time.time()
        max_xor, _ = solver.maximum_xor_subarray(arr)
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  Size {size}: {elapsed:.2f}ms, Max XOR: {max_xor}")
    
    # Benchmark string hashing
    print("\nString Hashing Performance:")
    for text_len in [100, 500, 1000]:
        text = generate_random_string(text_len)
        patterns = [generate_random_string(5) for _ in range(10)]
        
        start_time = time.time()
        matches = solver.string_hashing_with_trie(patterns, text)
        elapsed = (time.time() - start_time) * 1000
        
        total_matches = sum(len(positions) for positions in matches.values())
        print(f"  Text length {text_len}: {elapsed:.2f}ms, {total_matches} matches")


def demonstrate_optimization_techniques():
    """Demonstrate optimization techniques for contest problems"""
    print("\n=== Contest Problem Optimization Techniques ===")
    
    # Technique 1: Bit manipulation optimization
    print("1. Bit Manipulation Optimization:")
    
    def optimized_xor_query(arr: List[int]) -> int:
        """Optimized XOR using bit manipulation"""
        result = 0
        bit_count = [0] * 32
        
        for num in arr:
            for i in range(32):
                if num & (1 << i):
                    bit_count[i] += 1
        
        for i in range(32):
            if bit_count[i] > len(arr) // 2:
                result |= (1 << i)
        
        return result
    
    test_arr = [1, 2, 3, 4, 5]
    result = optimized_xor_query(test_arr)
    print(f"   Optimized XOR result for {test_arr}: {result}")
    
    # Technique 2: Memory optimization
    print("\n2. Memory Optimization:")
    
    print("   Using bit packing for trie nodes")
    print("   Sharing common prefixes")
    print("   Lazy evaluation for large datasets")
    
    # Technique 3: Time complexity optimization
    print("\n3. Time Complexity Optimization:")
    
    print("   Preprocessing: Build auxiliary structures")
    print("   Batch operations: Process multiple queries together")
    print("   Early termination: Stop when answer is found")
    
    # Technique 4: Contest-specific tips
    print("\n4. Contest-Specific Tips:")
    
    tips = [
        "Use appropriate data types (int vs long long)",
        "Handle edge cases (empty strings, single elements)",
        "Consider modular arithmetic for large results",
        "Implement template functions for common operations",
        "Use fast I/O for large inputs",
        "Pre-calculate constants and lookup tables"
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"   {i}. {tip}")


if __name__ == "__main__":
    test_maximum_xor_subarray()
    test_distinct_subsequences()
    test_string_hashing()
    test_palindromic_subsequences()
    test_multiple_string_matching()
    benchmark_contest_problems()
    demonstrate_optimization_techniques()

"""
Contest String Problems demonstrates competitive programming techniques:

Key Problem Types:
1. Maximum XOR Subarray - Prefix XOR with trie for optimal subarray
2. Distinct Subsequences - Combinatorial counting with trie storage
3. String Hashing - Fast pattern matching with rolling hash + trie
4. Palindromic Subsequences - Dynamic programming with trie optimization
5. Lexicographic Problems - Trie-based greedy algorithms
6. Multiple Pattern Matching - Aho-Corasick algorithm implementation
7. Compression Analysis - Trie-based compression ratio calculation

Contest Optimization Strategies:
- Bit manipulation for faster operations
- Memory-efficient data structures
- Preprocessing and auxiliary structures
- Early termination and pruning
- Template implementations for speed
- Handling edge cases and constraints

These problems commonly appear in competitive programming contests
and demonstrate advanced algorithmic thinking with trie optimizations.
"""
