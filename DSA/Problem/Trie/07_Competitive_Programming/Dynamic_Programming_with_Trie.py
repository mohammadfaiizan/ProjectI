"""
Dynamic Programming with Trie - Multiple Approaches
Difficulty: Hard

Advanced dynamic programming problems that utilize trie data structures
for optimization and state management in competitive programming.

Problems:
1. Longest Common Subsequence with Trie
2. Edit Distance with Trie Optimization
3. Substring DP with Trie States
4. Palindrome Partitioning with Trie
5. String Reconstruction DP
6. Trie-based State Compression
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import functools

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.word_id = -1
        self.fail_link = None  # For Aho-Corasick
        self.dp_value = 0
        self.visited = False

class TrieDPSolver:
    
    def __init__(self):
        self.root = TrieNode()
        self.memo = {}
    
    def longest_common_subsequence_trie(self, strings: List[str]) -> int:
        """
        Approach 1: LCS with Trie for Multiple Strings
        
        Find longest common subsequence among multiple strings using trie.
        
        Time: O(n * m^k) where n=strings, m=avg_length, k=num_strings
        Space: O(trie_size + states)
        """
        if not strings:
            return 0
        
        # Build trie of all suffixes
        suffix_trie = TrieNode()
        
        def build_suffix_trie():
            for string_idx, string in enumerate(strings):
                for start_pos in range(len(string)):
                    node = suffix_trie
                    for char in string[start_pos:]:
                        if char not in node.children:
                            node.children[char] = TrieNode()
                        node = node.children[char]
                    node.is_end = True
        
        build_suffix_trie()
        
        # DP with trie states
        @functools.lru_cache(maxsize=None)
        def dp(positions: Tuple[int, ...], trie_node_id: int) -> int:
            """
            positions: current position in each string
            trie_node_id: current position in trie (simulated)
            """
            # Base case: reached end of any string
            if any(pos >= len(strings[i]) for i, pos in enumerate(positions)):
                return 0
            
            max_lcs = 0
            
            # Try each possible character
            all_chars = set()
            for i, pos in enumerate(positions):
                if pos < len(strings[i]):
                    all_chars.add(strings[i][pos])
            
            for char in all_chars:
                # Check if all strings can use this character
                new_positions = []
                can_use = True
                
                for i, pos in enumerate(positions):
                    found_pos = -1
                    for j in range(pos, len(strings[i])):
                        if strings[i][j] == char:
                            found_pos = j + 1
                            break
                    
                    if found_pos == -1:
                        can_use = False
                        break
                    new_positions.append(found_pos)
                
                if can_use:
                    new_positions_tuple = tuple(new_positions)
                    result = 1 + dp(new_positions_tuple, trie_node_id + 1)
                    max_lcs = max(max_lcs, result)
            
            return max_lcs
        
        # Start DP from beginning of all strings
        initial_positions = tuple(0 for _ in strings)
        return dp(initial_positions, 0)
    
    def edit_distance_trie_optimized(self, source: str, targets: List[str]) -> Dict[str, int]:
        """
        Approach 2: Edit Distance with Trie Optimization
        
        Calculate edit distance from source to multiple targets efficiently.
        
        Time: O(|source| * trie_size)
        Space: O(trie_size)
        """
        # Build trie of target strings
        target_trie = TrieNode()
        
        for target_idx, target in enumerate(targets):
            node = target_trie
            for char in target:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
            node.word_id = target_idx
        
        # DP on trie
        results = {}
        
        def dfs_edit_distance(trie_node: TrieNode, source_idx: int, 
                            current_cost: int, path: str) -> None:
            """DFS through trie calculating edit distances"""
            
            # If we've processed the entire source
            if source_idx == len(source):
                if trie_node.is_end:
                    target = targets[trie_node.word_id]
                    results[target] = min(results.get(target, float('inf')), current_cost)
                
                # Cost to insert remaining characters in trie path
                def calculate_remaining_cost(node: TrieNode, depth: int) -> int:
                    if node.is_end:
                        target = targets[node.word_id]
                        total_cost = current_cost + depth
                        results[target] = min(results.get(target, float('inf')), total_cost)
                    
                    for child in node.children.values():
                        calculate_remaining_cost(child, depth + 1)
                
                calculate_remaining_cost(trie_node, 0)
                return
            
            current_char = source[source_idx]
            
            # Option 1: Match character (if exists in trie)
            if current_char in trie_node.children:
                dfs_edit_distance(trie_node.children[current_char], source_idx + 1, 
                                current_cost, path + current_char)
            
            # Option 2: Insert character (move in trie only)
            for char, child in trie_node.children.items():
                dfs_edit_distance(child, source_idx, current_cost + 1, path + char)
            
            # Option 3: Delete character (move in source only)
            dfs_edit_distance(trie_node, source_idx + 1, current_cost + 1, path)
            
            # Option 4: Substitute character
            for char, child in trie_node.children.items():
                if char != current_char:
                    dfs_edit_distance(child, source_idx + 1, current_cost + 1, path + char)
        
        dfs_edit_distance(target_trie, 0, 0, "")
        return results
    
    def substring_dp_with_trie(self, text: str, patterns: List[str]) -> int:
        """
        Approach 3: Substring DP with Trie States
        
        Count ways to partition text using given patterns.
        
        Time: O(|text|^2 + trie_size)
        Space: O(|text| + trie_size)
        """
        n = len(text)
        
        # Build pattern trie
        pattern_trie = TrieNode()
        for pattern in patterns:
            node = pattern_trie
            for char in pattern:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
        
        # DP array: dp[i] = number of ways to partition text[0:i]
        dp = [0] * (n + 1)
        dp[0] = 1  # Empty string has 1 way to partition
        
        for i in range(1, n + 1):
            # Try all possible last patterns ending at position i
            node = pattern_trie
            
            for j in range(i - 1, -1, -1):  # Go backwards from position i-1
                char = text[j]
                
                if char not in node.children:
                    break  # No pattern can extend further back
                
                node = node.children[char]
                
                # If we found a complete pattern
                if node.is_end:
                    dp[i] = (dp[i] + dp[j]) % (10**9 + 7)
        
        return dp[n]
    
    def palindrome_partitioning_trie(self, s: str) -> List[List[str]]:
        """
        Approach 4: Palindrome Partitioning with Trie
        
        Find all palindromic partitions using trie for efficient lookup.
        
        Time: O(n^2 + result_size)
        Space: O(trie_size + result_size)
        """
        n = len(s)
        
        # Precompute all palindromes and store in trie
        palindrome_trie = TrieNode()
        
        def is_palindrome(start: int, end: int) -> bool:
            while start < end:
                if s[start] != s[end]:
                    return False
                start += 1
                end -= 1
            return True
        
        # Build trie of all palindromic substrings
        for i in range(n):
            for j in range(i, n):
                if is_palindrome(i, j):
                    substring = s[i:j+1]
                    node = palindrome_trie
                    for char in substring:
                        if char not in node.children:
                            node.children[char] = TrieNode()
                        node = node.children[char]
                    node.is_end = True
        
        # DP to find all partitions
        def backtrack(start_idx: int, current_partition: List[str]) -> None:
            if start_idx == n:
                result.append(current_partition[:])
                return
            
            # Try all palindromes starting from start_idx
            node = palindrome_trie
            
            for end_idx in range(start_idx, n):
                char = s[end_idx]
                
                if char not in node.children:
                    break
                
                node = node.children[char]
                
                if node.is_end:
                    # Found palindrome from start_idx to end_idx
                    palindrome = s[start_idx:end_idx + 1]
                    current_partition.append(palindrome)
                    backtrack(end_idx + 1, current_partition)
                    current_partition.pop()
        
        result = []
        backtrack(0, [])
        return result
    
    def string_reconstruction_dp(self, fragments: List[str], target_length: int) -> int:
        """
        Approach 5: String Reconstruction DP
        
        Count ways to reconstruct strings of given length using fragments.
        
        Time: O(target_length * trie_size)
        Space: O(target_length + trie_size)
        """
        # Build fragment trie
        fragment_trie = TrieNode()
        
        for fragment in fragments:
            node = fragment_trie
            for char in fragment:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
        
        # DP with trie traversal
        @functools.lru_cache(maxsize=None)
        def dp(remaining_length: int, trie_node: TrieNode) -> int:
            """
            remaining_length: characters left to place
            trie_node: current position in trie
            """
            if remaining_length == 0:
                return 1 if trie_node == fragment_trie else 0
            
            if remaining_length < 0:
                return 0
            
            total_ways = 0
            
            # Option 1: Start new fragment
            if trie_node == fragment_trie:
                for char, child in fragment_trie.children.items():
                    total_ways += dp(remaining_length - 1, child)
            
            # Option 2: Continue current fragment
            else:
                for char, child in trie_node.children.items():
                    total_ways += dp(remaining_length - 1, child)
                
                # Option 3: End current fragment (if possible) and start new one
                if trie_node.is_end:
                    for char, child in fragment_trie.children.items():
                        total_ways += dp(remaining_length - 1, child)
            
            return total_ways % (10**9 + 7)
        
        return dp(target_length, fragment_trie)
    
    def trie_state_compression(self, states: List[str]) -> Dict[str, int]:
        """
        Approach 6: Trie-based State Compression
        
        Compress DP states using trie for memory efficiency.
        
        Time: O(states * avg_length)
        Space: O(compressed_size)
        """
        # Build state trie
        state_trie = TrieNode()
        state_to_id = {}
        state_counter = 0
        
        for state in states:
            node = state_trie
            for char in state:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            
            if not node.is_end:
                node.is_end = True
                node.word_id = state_counter
                state_to_id[state] = state_counter
                state_counter += 1
        
        # Calculate compression statistics
        def calculate_trie_size(node: TrieNode) -> int:
            size = 1
            for child in node.children.values():
                size += calculate_trie_size(child)
            return size
        
        original_size = sum(len(state) for state in states)
        compressed_size = calculate_trie_size(state_trie)
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': original_size / compressed_size if compressed_size > 0 else 0,
            'state_mapping': state_to_id
        }


def test_lcs_with_trie():
    """Test LCS with multiple strings using trie"""
    print("=== Testing LCS with Trie ===")
    
    solver = TrieDPSolver()
    
    test_cases = [
        (["ABCDGH", "AEDFHR", "ABCDFG"], "Expected LCS length around 3-4"),
        (["ABC", "BCD", "CDE"], "Expected LCS length around 1-2"),
        (["HELLO", "HELP", "HELD"], "Expected LCS length around 3"),
    ]
    
    for i, (strings, expected) in enumerate(test_cases):
        print(f"\nTest case {i+1}: {strings}")
        print(f"Expected: {expected}")
        
        lcs_length = solver.longest_common_subsequence_trie(strings)
        print(f"LCS Length: {lcs_length}")

def test_edit_distance_optimization():
    """Test edit distance with trie optimization"""
    print("\n=== Testing Edit Distance with Trie ===")
    
    solver = TrieDPSolver()
    
    source = "kitten"
    targets = ["sitting", "mitten", "bitten", "written"]
    
    print(f"Source: '{source}'")
    print(f"Targets: {targets}")
    
    distances = solver.edit_distance_trie_optimized(source, targets)
    
    print("Edit distances:")
    for target, distance in distances.items():
        print(f"  '{source}' -> '{target}': {distance}")

def test_substring_dp():
    """Test substring DP with trie"""
    print("\n=== Testing Substring DP with Trie ===")
    
    solver = TrieDPSolver()
    
    text = "catsanddog"
    patterns = ["cat", "cats", "and", "sand", "dog"]
    
    print(f"Text: '{text}'")
    print(f"Patterns: {patterns}")
    
    ways = solver.substring_dp_with_trie(text, patterns)
    print(f"Number of ways to partition: {ways}")
    
    # Test another case
    text2 = "abcd"
    patterns2 = ["a", "ab", "abc", "cd"]
    
    print(f"\nText: '{text2}'")
    print(f"Patterns: {patterns2}")
    
    ways2 = solver.substring_dp_with_trie(text2, patterns2)
    print(f"Number of ways to partition: {ways2}")

def test_palindrome_partitioning():
    """Test palindrome partitioning with trie"""
    print("\n=== Testing Palindrome Partitioning with Trie ===")
    
    solver = TrieDPSolver()
    
    test_strings = ["aab", "raceacar", "abccba"]
    
    for s in test_strings:
        print(f"\nString: '{s}'")
        
        partitions = solver.palindrome_partitioning_trie(s)
        print(f"Palindromic partitions ({len(partitions)}):")
        
        for i, partition in enumerate(partitions[:5]):  # Show first 5
            print(f"  {i+1}: {partition}")
        
        if len(partitions) > 5:
            print(f"  ... and {len(partitions) - 5} more")

def test_string_reconstruction():
    """Test string reconstruction DP"""
    print("\n=== Testing String Reconstruction DP ===")
    
    solver = TrieDPSolver()
    
    fragments = ["a", "aa", "aaa"]
    target_lengths = [1, 2, 3, 4, 5]
    
    print(f"Fragments: {fragments}")
    print("Ways to reconstruct strings of different lengths:")
    
    for length in target_lengths:
        ways = solver.string_reconstruction_dp(fragments, length)
        print(f"  Length {length}: {ways} ways")

def test_state_compression():
    """Test trie-based state compression"""
    print("\n=== Testing Trie State Compression ===")
    
    solver = TrieDPSolver()
    
    # Generate sample DP states
    states = [
        "state_000_001",
        "state_000_002", 
        "state_001_001",
        "state_001_002",
        "state_002_001",
        "prefix_common_a",
        "prefix_common_b",
        "different_state_x"
    ]
    
    print(f"Original states: {states}")
    
    compression_info = solver.trie_state_compression(states)
    
    print("Compression analysis:")
    for key, value in compression_info.items():
        if key != 'state_mapping':
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    print(f"State mappings:")
    for state, state_id in compression_info['state_mapping'].items():
        print(f"  '{state}' -> ID {state_id}")

def benchmark_dp_approaches():
    """Benchmark different DP approaches"""
    print("\n=== Benchmarking DP Approaches ===")
    
    import time
    
    solver = TrieDPSolver()
    
    # Test substring DP performance
    print("Substring DP Performance:")
    
    test_cases = [
        ("short text", ["a", "b", "c"]),
        ("medium text here", ["med", "text", "here", "ium"]),
        ("this is a longer text for testing", ["this", "is", "a", "for", "test"])
    ]
    
    for text, patterns in test_cases:
        start_time = time.time()
        ways = solver.substring_dp_with_trie(text, patterns)
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  Text length {len(text)}, {len(patterns)} patterns: {elapsed:.2f}ms, {ways} ways")
    
    # Test state compression performance
    print("\nState Compression Performance:")
    
    import random
    import string
    
    for num_states in [100, 500, 1000]:
        states = []
        for _ in range(num_states):
            # Generate states with common prefixes
            prefix = random.choice(["state", "config", "data"])
            suffix = ''.join(random.choices(string.ascii_lowercase, k=random.randint(5, 10)))
            states.append(f"{prefix}_{suffix}")
        
        start_time = time.time()
        compression_info = solver.trie_state_compression(states)
        elapsed = (time.time() - start_time) * 1000
        
        ratio = compression_info['compression_ratio']
        print(f"  {num_states} states: {elapsed:.2f}ms, compression ratio: {ratio:.2f}x")

def demonstrate_dp_optimizations():
    """Demonstrate DP optimization techniques"""
    print("\n=== DP Optimization Techniques ===")
    
    optimizations = [
        ("Memoization", "Cache results of expensive recursive calls"),
        ("State Compression", "Use trie to reduce memory usage for similar states"),
        ("Bottom-up DP", "Avoid recursion overhead with iterative approach"),
        ("Space Optimization", "Reduce space complexity by keeping only necessary states"),
        ("Preprocessing", "Build auxiliary structures (tries) before main DP"),
        ("State Pruning", "Eliminate impossible or suboptimal states early"),
        ("Batch Processing", "Process multiple similar subproblems together"),
        ("Rolling Array", "Use circular buffer for states that depend on recent history")
    ]
    
    print("Key DP optimization techniques:")
    for i, (technique, description) in enumerate(optimizations, 1):
        print(f"{i}. {technique}: {description}")
    
    print(f"\nWhen to use trie with DP:")
    print(f"  • String-based state representation")
    print(f"  • Many states with common prefixes/suffixes")
    print(f"  • Pattern matching in DP transitions")
    print(f"  • Multiple string processing problems")
    print(f"  • State space compression is beneficial")
    
    print(f"\nComplexity considerations:")
    print(f"  • Trie construction: O(total_string_length)")
    print(f"  • DP transitions: O(states * transitions)")
    print(f"  • Memory usage: O(trie_size + DP_table)")
    print(f"  • Cache performance: Better locality with trie")

if __name__ == "__main__":
    test_lcs_with_trie()
    test_edit_distance_optimization()
    test_substring_dp()
    test_palindrome_partitioning()
    test_string_reconstruction()
    test_state_compression()
    benchmark_dp_approaches()
    demonstrate_dp_optimizations()

"""
Dynamic Programming with Trie demonstrates advanced DP optimization:

Key Techniques:
1. LCS with Trie - Multiple string LCS using trie-based state management
2. Edit Distance Optimization - Single source to multiple targets efficiently
3. Substring DP - String partitioning problems with trie pattern matching
4. Palindrome Partitioning - Efficient palindrome detection using trie
5. String Reconstruction - Count reconstruction ways with fragment trie
6. State Compression - Reduce memory usage for string-based DP states

DP Optimization Strategies:
- Trie-based state representation for string problems
- Memoization with trie node states
- Bottom-up DP with trie traversal
- State space compression using common prefixes
- Preprocessing with auxiliary trie structures

Applications in Competitive Programming:
- String matching and manipulation problems
- Sequence alignment and edit distance variants
- Pattern counting and reconstruction problems
- State space reduction in complex DP
- Memory optimization for large state spaces

These techniques provide significant optimizations for string-based DP problems,
often reducing both time and space complexity compared to naive approaches.
"""
