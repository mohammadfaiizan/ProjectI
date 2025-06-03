"""
Trie (Prefix Tree) - String Search and Processing Data Structure
This module implements comprehensive Trie algorithms for string operations and advanced applications.
"""

from typing import List, Optional, Dict, Set, Tuple
from collections import defaultdict, deque

class TrieNode:
    """Standard Trie node structure"""
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.word_count = 0  # For tracking word frequencies
        self.prefix_count = 0  # For tracking prefix frequencies

class Trie:
    """Basic Trie (Prefix Tree) implementation"""
    
    def __init__(self):
        """Initialize empty Trie"""
        self.root = TrieNode()
        self.word_count = 0
    
    def insert(self, word: str) -> None:
        """
        Insert word into Trie
        
        Time Complexity: O(m) where m is length of word
        Space Complexity: O(m) in worst case
        
        Args:
            word: Word to insert
        """
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.prefix_count += 1
        
        if not node.is_end_of_word:
            self.word_count += 1
        
        node.is_end_of_word = True
        node.word_count += 1
    
    def search(self, word: str) -> bool:
        """
        Search for word in Trie
        
        Time Complexity: O(m)
        
        Args:
            word: Word to search
        
        Returns:
            bool: True if word exists
        """
        node = self._find_node(word)
        return node is not None and node.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        """
        Check if any word starts with prefix
        
        Time Complexity: O(m)
        
        Args:
            prefix: Prefix to check
        
        Returns:
            bool: True if prefix exists
        """
        return self._find_node(prefix) is not None
    
    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """Find node corresponding to prefix"""
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        
        return node
    
    def delete(self, word: str) -> bool:
        """
        Delete word from Trie
        
        Time Complexity: O(m)
        
        Args:
            word: Word to delete
        
        Returns:
            bool: True if word was deleted
        """
        def _delete_helper(node: TrieNode, word: str, index: int) -> bool:
            if index == len(word):
                if not node.is_end_of_word:
                    return False
                
                node.is_end_of_word = False
                node.word_count = 0
                # Return True if node has no children (can be deleted)
                return len(node.children) == 0
            
            char = word[index]
            if char not in node.children:
                return False
            
            should_delete_child = _delete_helper(node.children[char], word, index + 1)
            
            if should_delete_child:
                del node.children[char]
                # Return True if current node can be deleted
                return not node.is_end_of_word and len(node.children) == 0
            
            return False
        
        if self.search(word):
            _delete_helper(self.root, word, 0)
            self.word_count -= 1
            return True
        
        return False
    
    def get_all_words(self) -> List[str]:
        """
        Get all words in Trie
        
        Returns:
            List of all words
        """
        words = []
        
        def dfs(node: TrieNode, current_word: str):
            if node.is_end_of_word:
                words.append(current_word)
            
            for char, child_node in node.children.items():
                dfs(child_node, current_word + char)
        
        dfs(self.root, "")
        return words
    
    def count_words_with_prefix(self, prefix: str) -> int:
        """
        Count words that start with prefix
        
        Args:
            prefix: Prefix to count
        
        Returns:
            int: Number of words with prefix
        """
        node = self._find_node(prefix)
        if not node:
            return 0
        
        count = 0
        
        def dfs(node: TrieNode):
            nonlocal count
            if node.is_end_of_word:
                count += node.word_count
            
            for child in node.children.values():
                dfs(child)
        
        dfs(node)
        return count

class AdvancedTrie:
    """Advanced Trie with additional features"""
    
    def __init__(self):
        """Initialize advanced Trie"""
        self.trie = Trie()
        self.word_frequencies = {}
    
    def insert_with_frequency(self, word: str, frequency: int = 1) -> None:
        """
        Insert word with frequency
        
        Args:
            word: Word to insert
            frequency: Frequency of the word
        """
        self.trie.insert(word)
        self.word_frequencies[word] = self.word_frequencies.get(word, 0) + frequency
    
    def search_with_frequency(self, word: str) -> Tuple[bool, int]:
        """
        Search word and return frequency
        
        Args:
            word: Word to search
        
        Returns:
            Tuple of (exists, frequency)
        """
        exists = self.trie.search(word)
        frequency = self.word_frequencies.get(word, 0) if exists else 0
        return exists, frequency
    
    def get_words_with_prefix(self, prefix: str) -> List[str]:
        """
        Get all words that start with prefix
        
        Time Complexity: O(p + n) where p is prefix length, n is number of results
        
        Args:
            prefix: Prefix to search
        
        Returns:
            List of words starting with prefix
        """
        node = self.trie._find_node(prefix)
        if not node:
            return []
        
        words = []
        
        def dfs(node: TrieNode, current_word: str):
            if node.is_end_of_word:
                words.append(current_word)
            
            for char, child_node in node.children.items():
                dfs(child_node, current_word + char)
        
        dfs(node, prefix)
        return words
    
    def auto_complete(self, prefix: str, max_suggestions: int = 10) -> List[Tuple[str, int]]:
        """
        Auto-complete functionality with frequency-based ranking
        
        Args:
            prefix: Prefix to complete
            max_suggestions: Maximum number of suggestions
        
        Returns:
            List of (word, frequency) tuples sorted by frequency
        """
        words = self.get_words_with_prefix(prefix)
        
        # Add frequency information and sort
        word_freq_pairs = []
        for word in words:
            frequency = self.word_frequencies.get(word, 0)
            word_freq_pairs.append((word, frequency))
        
        # Sort by frequency (descending) then alphabetically
        word_freq_pairs.sort(key=lambda x: (-x[1], x[0]))
        
        return word_freq_pairs[:max_suggestions]
    
    def longest_prefix_matching(self, text: str) -> str:
        """
        Find longest prefix in Trie that matches beginning of text
        
        Args:
            text: Text to match against
        
        Returns:
            str: Longest matching prefix
        """
        longest_match = ""
        node = self.trie.root
        current_prefix = ""
        
        for char in text:
            if char not in node.children:
                break
            
            node = node.children[char]
            current_prefix += char
            
            if node.is_end_of_word:
                longest_match = current_prefix
        
        return longest_match
    
    def word_break_possible(self, text: str) -> bool:
        """
        Check if text can be segmented using words in Trie
        
        Time Complexity: O(n^2) where n is length of text
        
        Args:
            text: Text to segment
        
        Returns:
            bool: True if text can be segmented
        """
        n = len(text)
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and self.trie.search(text[j:i]):
                    dp[i] = True
                    break
        
        return dp[n]
    
    def word_break_segments(self, text: str) -> List[List[str]]:
        """
        Find all possible ways to segment text using words in Trie
        
        Args:
            text: Text to segment
        
        Returns:
            List of all possible segmentations
        """
        def backtrack(start: int, current_segment: List[str]) -> List[List[str]]:
            if start == len(text):
                return [current_segment[:]]
            
            results = []
            for end in range(start + 1, len(text) + 1):
                word = text[start:end]
                if self.trie.search(word):
                    current_segment.append(word)
                    results.extend(backtrack(end, current_segment))
                    current_segment.pop()
            
            return results
        
        return backtrack(0, [])
    
    def find_words_by_pattern(self, pattern: str, wildcard: str = '.') -> List[str]:
        """
        Find words matching pattern with wildcards
        
        Args:
            pattern: Pattern with wildcards
            wildcard: Wildcard character
        
        Returns:
            List of matching words
        """
        results = []
        
        def dfs(node: TrieNode, pattern_index: int, current_word: str):
            if pattern_index == len(pattern):
                if node.is_end_of_word:
                    results.append(current_word)
                return
            
            char = pattern[pattern_index]
            
            if char == wildcard:
                # Try all possible characters
                for child_char, child_node in node.children.items():
                    dfs(child_node, pattern_index + 1, current_word + child_char)
            else:
                # Exact character match
                if char in node.children:
                    dfs(node.children[char], pattern_index + 1, current_word + char)
        
        dfs(self.trie.root, 0, "")
        return results

class XORTrie:
    """Specialized Trie for XOR operations on integers"""
    
    class XORTrieNode:
        def __init__(self):
            self.children = {}
            self.count = 0
    
    def __init__(self, max_bits: int = 32):
        """
        Initialize XOR Trie
        
        Args:
            max_bits: Maximum number of bits to consider
        """
        self.root = self.XORTrieNode()
        self.max_bits = max_bits
    
    def insert(self, num: int) -> None:
        """
        Insert number into XOR Trie
        
        Time Complexity: O(max_bits)
        
        Args:
            num: Number to insert
        """
        node = self.root
        
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            
            if bit not in node.children:
                node.children[bit] = self.XORTrieNode()
            
            node = node.children[bit]
            node.count += 1
    
    def remove(self, num: int) -> None:
        """
        Remove number from XOR Trie
        
        Args:
            num: Number to remove
        """
        node = self.root
        
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            
            if bit in node.children:
                node = node.children[bit]
                node.count -= 1
                
                if node.count == 0:
                    # Clean up empty nodes
                    break
    
    def find_max_xor(self, num: int) -> int:
        """
        Find maximum XOR with any number in Trie
        
        Time Complexity: O(max_bits)
        
        Args:
            num: Number to find max XOR for
        
        Returns:
            int: Maximum XOR value
        """
        node = self.root
        max_xor = 0
        
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            
            # Try to go to opposite bit for maximum XOR
            desired_bit = 1 - bit
            
            if desired_bit in node.children and node.children[desired_bit].count > 0:
                max_xor |= (1 << i)
                node = node.children[desired_bit]
            elif bit in node.children and node.children[bit].count > 0:
                node = node.children[bit]
            else:
                # No valid path
                break
        
        return max_xor
    
    def find_max_xor_pair(self, nums: List[int]) -> int:
        """
        Find maximum XOR of any two numbers in the list
        
        Time Complexity: O(n * max_bits)
        
        Args:
            nums: List of numbers
        
        Returns:
            int: Maximum XOR of any pair
        """
        # Insert all numbers
        for num in nums:
            self.insert(num)
        
        max_xor = 0
        
        # Find maximum XOR for each number
        for num in nums:
            max_xor = max(max_xor, self.find_max_xor(num))
        
        return max_xor

class SuffixTrie:
    """Trie for suffix-based operations"""
    
    def __init__(self, text: str):
        """
        Build suffix Trie for given text
        
        Args:
            text: Text to build suffix Trie for
        """
        self.text = text
        self.trie = Trie()
        self._build_suffix_trie()
    
    def _build_suffix_trie(self):
        """Build suffix Trie from text"""
        for i in range(len(self.text)):
            suffix = self.text[i:] + "$"  # Add end marker
            self.trie.insert(suffix)
    
    def search_pattern(self, pattern: str) -> List[int]:
        """
        Search for pattern in text using suffix Trie
        
        Args:
            pattern: Pattern to search
        
        Returns:
            List of starting positions where pattern occurs
        """
        positions = []
        
        # Find all suffixes that start with pattern
        suffixes = []
        node = self.trie._find_node(pattern)
        
        if not node:
            return positions
        
        def collect_suffixes(node: TrieNode, current_suffix: str):
            if node.is_end_of_word:
                suffixes.append(current_suffix)
            
            for char, child_node in node.children.items():
                collect_suffixes(child_node, current_suffix + char)
        
        collect_suffixes(node, pattern)
        
        # Extract positions
        for suffix in suffixes:
            if suffix.endswith("$"):
                suffix = suffix[:-1]  # Remove end marker
                position = len(self.text) - len(suffix)
                positions.append(position)
        
        return sorted(positions)
    
    def count_occurrences(self, pattern: str) -> int:
        """
        Count occurrences of pattern in text
        
        Args:
            pattern: Pattern to count
        
        Returns:
            int: Number of occurrences
        """
        return len(self.search_pattern(pattern))
    
    def longest_repeated_substring(self) -> str:
        """
        Find longest repeated substring using suffix Trie
        
        Returns:
            str: Longest repeated substring
        """
        longest = ""
        
        def dfs(node: TrieNode, current_string: str):
            nonlocal longest
            
            # If node has multiple children or is visited multiple times,
            # it represents a repeated substring
            if len(node.children) > 1 or (len(node.children) == 1 and node.is_end_of_word):
                if len(current_string) > len(longest):
                    longest = current_string
            
            for char, child_node in node.children.items():
                if char != "$":  # Skip end marker
                    dfs(child_node, current_string + char)
        
        dfs(self.trie.root, "")
        return longest

class CompressedTrie:
    """Compressed Trie (Patricia Tree) for space efficiency"""
    
    class CompressedTrieNode:
        def __init__(self):
            self.children = {}
            self.is_end_of_word = False
            self.edge_label = ""
    
    def __init__(self):
        """Initialize compressed Trie"""
        self.root = self.CompressedTrieNode()
    
    def insert(self, word: str) -> None:
        """
        Insert word into compressed Trie
        
        Args:
            word: Word to insert
        """
        node = self.root
        i = 0
        
        while i < len(word):
            found_edge = False
            
            for edge_char, child_node in node.children.items():
                # Check if current position matches edge label
                edge_label = child_node.edge_label
                j = 0
                
                while (j < len(edge_label) and 
                       i + j < len(word) and 
                       edge_label[j] == word[i + j]):
                    j += 1
                
                if j > 0:  # Found matching edge
                    found_edge = True
                    
                    if j == len(edge_label):
                        # Entire edge matches, continue traversal
                        node = child_node
                        i += j
                    else:
                        # Partial match, need to split edge
                        # Create new internal node
                        new_internal = self.CompressedTrieNode()
                        new_internal.edge_label = edge_label[:j]
                        
                        # Update existing child
                        child_node.edge_label = edge_label[j:]
                        new_internal.children[edge_label[j]] = child_node
                        
                        # Update parent's reference
                        node.children[edge_char] = new_internal
                        
                        node = new_internal
                        i += j
                    
                    break
            
            if not found_edge:
                # Create new edge
                new_node = self.CompressedTrieNode()
                new_node.edge_label = word[i:]
                new_node.is_end_of_word = True
                
                node.children[word[i]] = new_node
                return
        
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """
        Search for word in compressed Trie
        
        Args:
            word: Word to search
        
        Returns:
            bool: True if word exists
        """
        node = self.root
        i = 0
        
        while i < len(word):
            found = False
            
            for edge_char, child_node in node.children.items():
                edge_label = child_node.edge_label
                
                if i < len(word) and word[i] == edge_char:
                    # Check if word matches edge label
                    if word[i:i+len(edge_label)] == edge_label:
                        node = child_node
                        i += len(edge_label)
                        found = True
                        break
                    else:
                        return False
            
            if not found:
                return False
        
        return node.is_end_of_word
    
    def get_all_words(self) -> List[str]:
        """Get all words in compressed Trie"""
        words = []
        
        def dfs(node: self.CompressedTrieNode, current_word: str):
            if node.is_end_of_word:
                words.append(current_word)
            
            for child_node in node.children.values():
                dfs(child_node, current_word + child_node.edge_label)
        
        dfs(self.root, "")
        return words

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Trie (Prefix Tree) Demo ===\n")
    
    # Example 1: Basic Trie Operations
    print("1. Basic Trie Operations:")
    
    basic_trie = Trie()
    words = ["apple", "app", "application", "apply", "banana", "band", "bandana"]
    
    # Insert words
    for word in words:
        basic_trie.insert(word)
    
    print(f"Inserted words: {words}")
    print(f"Total word count: {basic_trie.word_count}")
    
    # Search operations
    search_tests = ["app", "apple", "appl", "application", "ban", "banana"]
    for word in search_tests:
        exists = basic_trie.search(word)
        has_prefix = basic_trie.starts_with(word)
        print(f"  '{word}': exists={exists}, has_prefix={has_prefix}")
    
    # Get all words
    all_words = basic_trie.get_all_words()
    print(f"All words in Trie: {sorted(all_words)}")
    
    # Delete operation
    print(f"Deleting 'app': {basic_trie.delete('app')}")
    print(f"Search 'app' after deletion: {basic_trie.search('app')}")
    print(f"Total word count after deletion: {basic_trie.word_count}")
    print()
    
    # Example 2: Advanced Trie with Frequencies
    print("2. Advanced Trie with Auto-complete:")
    
    advanced_trie = AdvancedTrie()
    
    # Insert words with frequencies (simulating search frequency)
    word_frequencies = [
        ("apple", 100), ("application", 80), ("apply", 60), ("app", 120),
        ("appreciate", 40), ("approach", 90), ("appropriate", 30),
        ("banana", 70), ("band", 50), ("bandana", 20)
    ]
    
    for word, freq in word_frequencies:
        advanced_trie.insert_with_frequency(word, freq)
    
    # Auto-complete functionality
    print("Auto-complete suggestions:")
    prefixes = ["app", "ban", "appr"]
    
    for prefix in prefixes:
        suggestions = advanced_trie.auto_complete(prefix, max_suggestions=5)
        print(f"  '{prefix}': {suggestions}")
    
    # Longest prefix matching
    test_texts = ["application_form", "band_practice", "approximately"]
    print("Longest prefix matching:")
    
    for text in test_texts:
        longest_match = advanced_trie.longest_prefix_matching(text)
        print(f"  '{text}' -> '{longest_match}'")
    print()
    
    # Example 3: Word Break Problem
    print("3. Word Break Problem:")
    
    # Create Trie with dictionary words
    word_break_trie = AdvancedTrie()
    dictionary = ["cats", "dog", "sand", "and", "cat", "dogs", "doggy"]
    
    for word in dictionary:
        word_break_trie.insert_with_frequency(word)
    
    print(f"Dictionary: {dictionary}")
    
    test_sentences = ["catsanddog", "catsanddogs", "catsdoggy", "catdog"]
    
    for sentence in test_sentences:
        can_break = word_break_trie.word_break_possible(sentence)
        segments = word_break_trie.word_break_segments(sentence) if can_break else []
        
        print(f"  '{sentence}': can_break={can_break}")
        if segments:
            print(f"    Possible segmentations: {segments}")
    print()
    
    # Example 4: Pattern Matching with Wildcards
    print("4. Pattern Matching with Wildcards:")
    
    pattern_trie = AdvancedTrie()
    pattern_words = ["cat", "car", "card", "care", "careful", "cut", "cute"]
    
    for word in pattern_words:
        pattern_trie.insert_with_frequency(word)
    
    print(f"Words: {pattern_words}")
    
    patterns = ["c.t", "ca.", "c.r.", "cu.."]
    
    for pattern in patterns:
        matches = pattern_trie.find_words_by_pattern(pattern)
        print(f"  Pattern '{pattern}': {matches}")
    print()
    
    # Example 5: XOR Trie for Maximum XOR
    print("5. XOR Trie for Maximum XOR Pair:")
    
    xor_trie = XORTrie(max_bits=8)  # Using 8 bits for demo
    
    numbers = [3, 10, 5, 25, 2, 8]
    print(f"Numbers: {numbers}")
    
    # Find maximum XOR pair
    max_xor = xor_trie.find_max_xor_pair(numbers)
    print(f"Maximum XOR of any pair: {max_xor}")
    
    # Find maximum XOR for each number
    print("Maximum XOR for each number:")
    for num in numbers:
        xor_trie_temp = XORTrie(max_bits=8)
        for other_num in numbers:
            if other_num != num:
                xor_trie_temp.insert(other_num)
        
        max_xor_individual = xor_trie_temp.find_max_xor(num)
        print(f"  {num}: {max_xor_individual}")
    print()
    
    # Example 6: Suffix Trie for Pattern Searching
    print("6. Suffix Trie for Pattern Searching:")
    
    text = "banana"
    suffix_trie = SuffixTrie(text)
    
    print(f"Text: '{text}'")
    
    patterns = ["an", "ana", "ban", "na"]
    
    for pattern in patterns:
        positions = suffix_trie.search_pattern(pattern)
        count = suffix_trie.count_occurrences(pattern)
        print(f"  Pattern '{pattern}': positions={positions}, count={count}")
    
    # Find longest repeated substring
    longest_repeated = suffix_trie.longest_repeated_substring()
    print(f"Longest repeated substring: '{longest_repeated}'")
    print()
    
    # Example 7: Compressed Trie (Patricia Tree)
    print("7. Compressed Trie (Patricia Tree):")
    
    compressed_trie = CompressedTrie()
    compressed_words = ["apple", "application", "apply", "banana", "band"]
    
    for word in compressed_words:
        compressed_trie.insert(word)
    
    print(f"Inserted words: {compressed_words}")
    
    # Search in compressed Trie
    search_words = ["apple", "app", "application", "band", "bandana"]
    
    for word in search_words:
        found = compressed_trie.search(word)
        print(f"  Search '{word}': {found}")
    
    # Get all words from compressed Trie
    all_compressed_words = compressed_trie.get_all_words()
    print(f"All words in compressed Trie: {sorted(all_compressed_words)}")
    print()
    
    # Example 8: Performance Comparison
    print("8. Performance Analysis:")
    
    # Large dataset for performance testing
    import random
    import string
    
    def generate_random_words(count: int, min_length: int = 3, max_length: int = 10) -> List[str]:
        words = set()
        while len(words) < count:
            length = random.randint(min_length, max_length)
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.add(word)
        return list(words)
    
    large_word_list = generate_random_words(1000)
    
    # Test basic Trie
    large_trie = Trie()
    for word in large_word_list:
        large_trie.insert(word)
    
    print(f"Inserted {len(large_word_list)} random words into Trie")
    print(f"Total words in Trie: {large_trie.word_count}")
    
    # Test search performance
    sample_words = random.sample(large_word_list, 10)
    print(f"Sample search results:")
    
    for word in sample_words:
        exists = large_trie.search(word)
        print(f"  '{word[:8]}...': {exists}")
    
    # Test prefix operations
    sample_prefixes = [word[:3] for word in sample_words[:5]]
    print(f"Prefix count tests:")
    
    advanced_large_trie = AdvancedTrie()
    for word in large_word_list:
        advanced_large_trie.insert_with_frequency(word)
    
    for prefix in sample_prefixes:
        count = advanced_large_trie.trie.count_words_with_prefix(prefix)
        words_with_prefix = advanced_large_trie.get_words_with_prefix(prefix)
        print(f"  Prefix '{prefix}': {count} words, sample: {words_with_prefix[:3]}")
    
    print(f"\n=== Demo Complete ===") 