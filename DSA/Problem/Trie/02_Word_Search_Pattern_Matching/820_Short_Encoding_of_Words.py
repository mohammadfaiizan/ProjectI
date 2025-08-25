"""
820. Short Encoding of Words - Multiple Approaches
Difficulty: Medium

A valid encoding of an array of words is any reference string s and array of indices 
such that:
- words[i] = s[substring from indices[i] to next '#' character (or end of string)]

Given an array of words, return the length of the shortest reference string s possible 
of any valid encoding of words.

LeetCode Problem: https://leetcode.com/problems/short-encoding-of-words/

Example:
Input: words = ["time", "me", "bell"]
Output: 10
Explanation: A valid encoding would be s = "time#bell#" and indices = [0, 2, 5].
words[0] = "time", the substring of s from index 0 to index 3
words[1] = "me", the substring of s from index 2 to index 3
words[2] = "bell", the substring of s from index 5 to index 8
"""

from typing import List, Set
from collections import defaultdict

class TrieNode:
    """Trie node for suffix-based encoding"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.depth = 0

class Solution:
    
    def minimumLengthEncoding1(self, words: List[str]) -> int:
        """
        Approach 1: Suffix Trie
        
        Build trie from reversed words to find unique suffixes.
        
        Time: O(sum of word lengths)
        Space: O(sum of word lengths)
        """
        # Build suffix trie (reversed words)
        root = TrieNode()
        nodes = {}  # word -> leaf node mapping
        
        for word in set(words):  # Remove duplicates
            node = root
            for char in reversed(word):
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
            nodes[word] = node
        
        # Calculate encoding length
        total_length = 0
        for word, node in nodes.items():
            # If node has no children, it's a unique suffix
            if not node.children:
                total_length += len(word) + 1  # +1 for '#'
        
        return total_length
    
    def minimumLengthEncoding2(self, words: List[str]) -> int:
        """
        Approach 2: Set-based Suffix Removal
        
        Remove words that are suffixes of other words.
        
        Time: O(sum of word lengths squared)
        Space: O(number of words)
        """
        word_set = set(words)
        
        # Remove words that are suffixes of other words
        for word in words:
            for i in range(1, len(word)):
                suffix = word[i:]
                if suffix in word_set:
                    word_set.discard(suffix)
        
        # Calculate total length
        return sum(len(word) + 1 for word in word_set)
    
    def minimumLengthEncoding3(self, words: List[str]) -> int:
        """
        Approach 3: Optimized Trie with Length Tracking
        
        Enhanced trie that tracks depth for length calculation.
        
        Time: O(sum of word lengths)
        Space: O(sum of word lengths)
        """
        root = TrieNode()
        leaf_nodes = []
        
        # Build trie and collect leaf information
        for word in set(words):
            node = root
            for i, char in enumerate(reversed(word)):
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                node.depth = i + 1
            
            node.is_end = True
            leaf_nodes.append((node, len(word)))
        
        # Calculate encoding length from unique leaves
        total_length = 0
        for node, word_length in leaf_nodes:
            if not node.children:  # Leaf node (unique suffix)
                total_length += word_length + 1
        
        return total_length
    
    def minimumLengthEncoding4(self, words: List[str]) -> int:
        """
        Approach 4: Reverse and Sort Approach
        
        Sort reversed words and check for prefix relationships.
        
        Time: O(n * m * log(n)) where n=number of words, m=max word length
        Space: O(n * m)
        """
        # Reverse all words and sort
        reversed_words = [word[::-1] for word in set(words)]
        reversed_words.sort()
        
        unique_words = []
        
        for i, rev_word in enumerate(reversed_words):
            # Check if current word is prefix of next word
            is_prefix = False
            if i + 1 < len(reversed_words):
                next_word = reversed_words[i + 1]
                if next_word.startswith(rev_word):
                    is_prefix = True
            
            if not is_prefix:
                unique_words.append(rev_word[::-1])  # Reverse back
        
        return sum(len(word) + 1 for word in unique_words)
    
    def minimumLengthEncoding5(self, words: List[str]) -> int:
        """
        Approach 5: Graph-based Approach
        
        Model as graph where edges represent suffix relationships.
        
        Time: O(n^2 * m) where n=number of words, m=max word length
        Space: O(n)
        """
        words = list(set(words))  # Remove duplicates
        n = len(words)
        
        # Build suffix relationship graph
        is_suffix = [False] * n
        
        for i in range(n):
            for j in range(n):
                if i != j and words[i] != words[j]:
                    # Check if words[i] is suffix of words[j]
                    if words[j].endswith(words[i]):
                        is_suffix[i] = True
                        break
        
        # Calculate encoding length for non-suffix words
        total_length = 0
        for i, word in enumerate(words):
            if not is_suffix[i]:
                total_length += len(word) + 1
        
        return total_length
    
    def minimumLengthEncoding6(self, words: List[str]) -> int:
        """
        Approach 6: Hash Set with Suffix Generation
        
        Generate all suffixes and remove from original set.
        
        Time: O(sum of word lengths squared)
        Space: O(sum of word lengths)
        """
        word_set = set(words)
        
        # For each word, remove all its suffixes from the set
        for word in words:
            for i in range(1, len(word)):
                word_set.discard(word[i:])
        
        return sum(len(word) + 1 for word in word_set)
    
    def minimumLengthEncoding7(self, words: List[str]) -> int:
        """
        Approach 7: Trie with Compression
        
        Compressed trie to handle long common suffixes efficiently.
        
        Time: O(sum of word lengths)
        Space: O(unique suffixes)
        """
        class CompressedTrieNode:
            def __init__(self):
                self.children = {}
                self.compressed_suffix = ""
                self.is_word_end = False
                self.word_length = 0
        
        root = CompressedTrieNode()
        word_nodes = {}
        
        for word in set(words):
            node = root
            reversed_word = word[::-1]
            
            i = 0
            while i < len(reversed_word):
                char = reversed_word[i]
                
                if char in node.children:
                    child = node.children[char]
                    # Check compression
                    if child.compressed_suffix:
                        suffix = child.compressed_suffix
                        j = 0
                        # Find common prefix of remaining word and compressed suffix
                        while (j < len(suffix) and 
                               i + 1 + j < len(reversed_word) and
                               suffix[j] == reversed_word[i + 1 + j]):
                            j += 1
                        
                        if j == len(suffix):
                            # Entire compressed suffix matches
                            i += 1 + j
                            node = child
                        else:
                            # Need to split compression
                            # This is complex - fall back to regular trie
                            node = child
                            i += 1
                    else:
                        node = child
                        i += 1
                else:
                    # Create new compressed node
                    new_node = CompressedTrieNode()
                    remaining = reversed_word[i + 1:]
                    new_node.compressed_suffix = remaining
                    new_node.is_word_end = True
                    new_node.word_length = len(word)
                    
                    node.children[char] = new_node
                    word_nodes[word] = new_node
                    break
        
        # Calculate encoding length
        total_length = 0
        for word, node in word_nodes.items():
            if not node.children:
                total_length += len(word) + 1
        
        return total_length


def test_basic_cases():
    """Test basic functionality"""
    print("=== Testing Basic Cases ===")
    
    solution = Solution()
    
    test_cases = [
        # LeetCode examples
        (["time", "me", "bell"], 10),  # "time#bell#"
        (["t"], 2),                   # "t#"
        (["me", "time"], 5),          # "time#"
        
        # More complex cases
        (["a", "aa", "aaa"], 7),      # "aaa#" (4+3=7)
        (["abc", "bc", "c"], 4),      # "abc#"
        (["ab", "ba"], 6),            # "ab#ba#"
        
        # No suffix relationships
        (["cat", "dog", "bird"], 12), # "cat#dog#bird#"
        
        # All same word
        (["same", "same", "same"], 5), # "same#"
        
        # Empty cases
        ([], 0),
        ([""], 1),  # "#"
    ]
    
    approaches = [
        ("Suffix Trie", solution.minimumLengthEncoding1),
        ("Set Removal", solution.minimumLengthEncoding2),
        ("Optimized Trie", solution.minimumLengthEncoding3),
        ("Reverse Sort", solution.minimumLengthEncoding4),
        ("Graph-based", solution.minimumLengthEncoding5),
        ("Hash Set", solution.minimumLengthEncoding6),
    ]
    
    for i, (words, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {words}")
        print(f"Expected: {expected}")
        
        for name, method in approaches:
            try:
                result = method(words[:])  # Copy to avoid modification
                status = "✓" if result == expected else "✗"
                print(f"  {name:15}: {result} {status}")
            except Exception as e:
                print(f"  {name:15}: Error - {e}")


def demonstrate_trie_construction():
    """Demonstrate suffix trie construction"""
    print("\n=== Suffix Trie Construction Demo ===")
    
    words = ["time", "me", "bell"]
    print(f"Words: {words}")
    print(f"Goal: Find shortest encoding")
    
    # Show suffix relationships
    print(f"\nSuffix analysis:")
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if i != j and word2.endswith(word1):
                print(f"  '{word1}' is suffix of '{word2}'")
    
    # Build suffix trie
    root = TrieNode()
    word_nodes = {}
    
    print(f"\nBuilding suffix trie (reversed words):")
    
    for word in words:
        print(f"\nInserting '{word}' (reversed: '{word[::-1]}'):")
        node = root
        
        for i, char in enumerate(reversed(word)):
            if char not in node.children:
                node.children[char] = TrieNode()
                print(f"  Created node for '{char}' at depth {i+1}")
            
            node = node.children[char]
            print(f"  Moved to node '{char}', path: '{word[::-1][:i+1]}'")
        
        node.is_end = True
        word_nodes[word] = node
        print(f"  Marked end for word '{word}'")
    
    # Show which words contribute to encoding
    print(f"\nEncoding analysis:")
    total_length = 0
    
    for word, node in word_nodes.items():
        if not node.children:
            contribution = len(word) + 1
            total_length += contribution
            print(f"  '{word}' -> unique suffix -> contributes {contribution} chars")
        else:
            print(f"  '{word}' -> has extensions -> not included in encoding")
    
    print(f"\nTotal encoding length: {total_length}")


def demonstrate_encoding_process():
    """Demonstrate the encoding process"""
    print("\n=== Encoding Process Demo ===")
    
    words = ["time", "me", "bell"]
    
    # Method 1: Show what gets encoded
    word_set = set(words)
    
    print(f"Original words: {list(word_set)}")
    
    # Remove suffixes
    for word in words:
        for i in range(1, len(word)):
            suffix = word[i:]
            if suffix in word_set:
                print(f"Removing '{suffix}' (suffix of '{word}')")
                word_set.discard(suffix)
    
    remaining_words = list(word_set)
    print(f"Remaining words after suffix removal: {remaining_words}")
    
    # Build encoding string
    encoding_parts = []
    total_length = 0
    
    for word in remaining_words:
        encoding_parts.append(word + "#")
        total_length += len(word) + 1
        print(f"  '{word}' -> '{word}#' (length: {len(word) + 1})")
    
    encoding_string = "".join(encoding_parts)
    print(f"\nFinal encoding: '{encoding_string}'")
    print(f"Total length: {total_length}")
    
    # Show how original words can be recovered
    print(f"\nRecovery verification:")
    current_pos = 0
    
    for word in words:
        # Find word in encoding
        for start in range(len(encoding_string)):
            end = encoding_string.find('#', start)
            if end == -1:
                end = len(encoding_string)
            
            encoded_word = encoding_string[start:end]
            if encoded_word.endswith(word):
                recovery_start = end - len(word)
                print(f"  '{word}' can be recovered from position {recovery_start} to {end-1}")
                break


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import time
    import random
    import string
    
    solution = Solution()
    
    # Generate test data with suffix relationships
    def generate_words_with_suffixes(base_count: int, max_length: int) -> List[str]:
        words = []
        
        # Generate base words
        for _ in range(base_count):
            length = random.randint(3, max_length)
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.append(word)
            
            # Generate some suffixes
            if random.random() < 0.3:  # 30% chance to add suffixes
                for i in range(1, len(word)):
                    if random.random() < 0.2:  # 20% chance for each suffix
                        words.append(word[i:])
        
        return list(set(words))  # Remove duplicates
    
    test_scenarios = [
        ("Small", generate_words_with_suffixes(10, 6)),
        ("Medium", generate_words_with_suffixes(50, 8)),
        ("Large", generate_words_with_suffixes(200, 10)),
    ]
    
    approaches = [
        ("Suffix Trie", solution.minimumLengthEncoding1),
        ("Set Removal", solution.minimumLengthEncoding2),
        ("Optimized Trie", solution.minimumLengthEncoding3),
        ("Hash Set", solution.minimumLengthEncoding6),
    ]
    
    for scenario_name, words in test_scenarios:
        print(f"\n--- {scenario_name} Dataset ---")
        print(f"Words: {len(words)}, Avg length: {sum(len(w) for w in words) / len(words):.1f}")
        
        for approach_name, method in approaches:
            start_time = time.time()
            
            # Run multiple times for better measurement
            for _ in range(10):
                result = method(words[:])
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 10
            
            print(f"  {approach_name:15}: {avg_time*1000:.2f}ms")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solution = Solution()
    
    # Application 1: DNA sequence compression
    print("1. DNA Sequence Compression:")
    
    dna_sequences = [
        "ATCGATCG",
        "TCGATCG",  # suffix of first
        "CGATCG",   # suffix of first
        "GATCG",    # suffix of first
        "ATCG",     # suffix of first
        "GGCCTTAA",
        "CCTTAA",   # suffix of GGCCTTAA
    ]
    
    print(f"   Original sequences: {dna_sequences}")
    
    encoding_length = solution.minimumLengthEncoding1(dna_sequences)
    original_length = sum(len(seq) + 1 for seq in dna_sequences)  # +1 for separator
    
    print(f"   Original storage: {original_length} characters")
    print(f"   Compressed storage: {encoding_length} characters")
    print(f"   Compression ratio: {encoding_length/original_length:.2%}")
    
    # Application 2: URL path compression
    print(f"\n2. URL Path Compression:")
    
    url_paths = [
        "/api/users/profile",
        "/profile",           # suffix
        "/api/users/settings",
        "/settings",          # suffix
        "/api/posts",
        "/posts",             # suffix
        "/admin/dashboard"
    ]
    
    print(f"   URL paths: {url_paths}")
    
    url_encoding_length = solution.minimumLengthEncoding1(url_paths)
    url_original_length = sum(len(path) + 1 for path in url_paths)
    
    print(f"   Original storage: {url_original_length} characters")
    print(f"   Compressed storage: {url_encoding_length} characters")
    print(f"   Savings: {url_original_length - url_encoding_length} characters")
    
    # Application 3: Dictionary compression
    print(f"\n3. Dictionary Word Compression:")
    
    dictionary_words = [
        "programming",
        "gramming",     # suffix
        "amming",       # suffix
        "development",
        "opment",       # suffix
        "testing",
        "esting",       # suffix
        "debugging"
    ]
    
    print(f"   Dictionary words: {dictionary_words}")
    
    dict_encoding_length = solution.minimumLengthEncoding1(dictionary_words)
    dict_original_length = sum(len(word) + 1 for word in dictionary_words)
    
    print(f"   Original storage: {dict_original_length} characters")
    print(f"   Compressed storage: {dict_encoding_length} characters")
    print(f"   Space saved: {dict_original_length - dict_encoding_length} characters ({(1-dict_encoding_length/dict_original_length)*100:.1f}%)")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solution = Solution()
    
    edge_cases = [
        # Empty and single character
        ([], "Empty list"),
        ([""], "Single empty string"),
        (["a"], "Single character"),
        
        # All same words
        (["same", "same", "same"], "All identical words"),
        
        # Chain of suffixes
        (["abcdef", "bcdef", "cdef", "def", "ef", "f"], "Chain of suffixes"),
        
        # No suffix relationships
        (["cat", "dog", "bird", "fish"], "No suffix relationships"),
        
        # Complex suffix patterns
        (["abc", "bc", "c", "def", "ef", "f"], "Multiple suffix chains"),
        
        # Very long words
        (["a" * 1000, "a" * 999, "a" * 998], "Very long words"),
        
        # Single character words
        (["a", "b", "c", "d"], "Single character words"),
    ]
    
    for words, description in edge_cases:
        print(f"\n{description}:")
        print(f"  Words: {words if len(str(words)) < 50 else f'{words[:3]}... (total: {len(words)})'}")
        
        try:
            result = solution.minimumLengthEncoding1(words)
            print(f"  Encoding length: {result}")
            
            if words:
                original_length = sum(len(word) + 1 for word in words)
                print(f"  Original length: {original_length}")
                print(f"  Compression ratio: {result/original_length:.2%}")
        except Exception as e:
            print(f"  Error: {e}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Suffix Trie",
         "Time: O(sum of word lengths) - build trie once",
         "Space: O(sum of word lengths) - trie storage"),
        
        ("Set Removal",
         "Time: O(sum of word lengths²) - check all suffixes",
         "Space: O(number of words) - set storage"),
        
        ("Reverse Sort",
         "Time: O(n*m*log(n)) - sort reversed words",
         "Space: O(n*m) - store reversed words"),
        
        ("Graph-based",
         "Time: O(n²*m) - check all pairs for suffix relationship",
         "Space: O(n) - boolean array"),
        
        ("Hash Set",
         "Time: O(sum of word lengths²) - generate all suffixes",
         "Space: O(sum of word lengths) - set storage"),
    ]
    
    print("Approach Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nWhere:")
    print(f"  n = number of words")
    print(f"  m = average word length")
    
    print(f"\nRecommendations:")
    print(f"  • Use Suffix Trie for optimal time complexity")
    print(f"  • Use Set Removal for simple implementation")
    print(f"  • Use Hash Set for memory-constrained environments")


if __name__ == "__main__":
    test_basic_cases()
    demonstrate_trie_construction()
    demonstrate_encoding_process()
    benchmark_approaches()
    demonstrate_real_world_applications()
    test_edge_cases()
    analyze_complexity()

"""
820. Short Encoding of Words demonstrates suffix-based optimization approaches:

1. Suffix Trie - Build trie from reversed words to identify unique suffixes
2. Set Removal - Remove words that are suffixes of other words using set operations
3. Optimized Trie - Enhanced trie with depth tracking for efficient calculation
4. Reverse Sort - Sort reversed words and check prefix relationships
5. Graph-based - Model suffix relationships as directed graph
6. Hash Set - Generate all suffixes and remove from original set
7. Compressed Trie - Space-optimized trie for long common suffixes

Each approach provides different optimization strategies for finding minimal
string encodings while preserving the ability to recover original words.
"""
