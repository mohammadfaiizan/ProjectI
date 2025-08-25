"""
208. Implement Trie (Prefix Tree) - Multiple Approaches
Difficulty: Medium

A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently 
store and search strings in a dataset of strings. There are various applications of this 
data structure, such as autocomplete and spellchecker.

Implement the Trie class:
- Trie() Initializes the trie object.
- void insert(String word) Inserts the string word into the trie.
- boolean search(String word) Returns true if the string word is in the trie, and false otherwise.
- boolean startsWith(String prefix) Returns true if there is a previously inserted string 
  word that has the prefix prefix, and false otherwise.
"""

from typing import Dict, List, Optional
import sys

class TrieBasic:
    """
    Approach 1: Basic Trie Implementation
    
    Standard trie implementation using nested dictionaries.
    
    Time Complexity: 
    - Insert: O(m) where m is the length of the word
    - Search: O(m) where m is the length of the word
    - StartsWith: O(p) where p is the length of the prefix
    
    Space Complexity: O(ALPHABET_SIZE * N * M) where N is number of words, M is average length
    """
    
    def __init__(self):
        self.root = {}
        self.end_symbol = '#'
    
    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node[self.end_symbol] = True
    
    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return self.end_symbol in node
    
    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True

class TrieNode:
    """Helper class for object-oriented trie implementation"""
    
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_word: bool = False
        self.word_count: int = 0  # Count of words passing through this node

class TrieObjectOriented:
    """
    Approach 2: Object-Oriented Trie with TrieNode class
    
    More structured implementation using separate TrieNode class.
    
    Time Complexity: Same as basic implementation
    Space Complexity: Same as basic implementation
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.word_count += 1
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        node = self._find_node(word)
        return node is not None and node.is_end_of_word
    
    def startsWith(self, prefix: str) -> bool:
        return self._find_node(prefix) is not None
    
    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def count_words_with_prefix(self, prefix: str) -> int:
        """Additional method: count words with given prefix"""
        node = self._find_node(prefix)
        return node.word_count if node else 0

class TrieArray:
    """
    Approach 3: Array-based Trie (for lowercase letters only)
    
    Memory-efficient implementation using arrays for fixed alphabet.
    
    Time Complexity: Same as basic implementation
    Space Complexity: O(26 * N * M) - more memory efficient for English letters
    """
    
    class TrieArrayNode:
        def __init__(self):
            self.children = [None] * 26  # For 'a' to 'z'
            self.is_end_of_word = False
    
    def __init__(self):
        self.root = self.TrieArrayNode()
    
    def _char_to_index(self, char: str) -> int:
        return ord(char) - ord('a')
    
    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            index = self._char_to_index(char)
            if node.children[index] is None:
                node.children[index] = self.TrieArrayNode()
            node = node.children[index]
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        node = self._find_node(word)
        return node is not None and node.is_end_of_word
    
    def startsWith(self, prefix: str) -> bool:
        return self._find_node(prefix) is not None
    
    def _find_node(self, prefix: str) -> Optional['TrieArray.TrieArrayNode']:
        node = self.root
        for char in prefix:
            index = self._char_to_index(char)
            if node.children[index] is None:
                return None
            node = node.children[index]
        return node

class TrieCompressed:
    """
    Approach 4: Compressed Trie (Radix Tree)
    
    Space-optimized trie that compresses single-child paths.
    
    Time Complexity: O(m) for operations but with better constants
    Space Complexity: Reduced space usage by path compression
    """
    
    class CompressedNode:
        def __init__(self, compressed_path: str = ""):
            self.compressed_path = compressed_path
            self.children: Dict[str, 'TrieCompressed.CompressedNode'] = {}
            self.is_end_of_word = False
    
    def __init__(self):
        self.root = self.CompressedNode()
    
    def insert(self, word: str) -> None:
        self._insert_recursive(self.root, word, 0)
    
    def _insert_recursive(self, node: 'TrieCompressed.CompressedNode', word: str, index: int) -> None:
        if index == len(word):
            node.is_end_of_word = True
            return
        
        char = word[index]
        
        if char in node.children:
            child = node.children[char]
            # Find common prefix with compressed path
            common_len = 0
            remaining_word = word[index + 1:]
            
            for i in range(min(len(child.compressed_path), len(remaining_word))):
                if child.compressed_path[i] == remaining_word[i]:
                    common_len += 1
                else:
                    break
            
            if common_len == len(child.compressed_path):
                # Full path matches, continue with child
                self._insert_recursive(child, word, index + 1 + common_len)
            else:
                # Need to split the compressed path
                self._split_node(child, common_len)
                self._insert_recursive(child, word, index + 1 + common_len)
        else:
            # Create new compressed node
            remaining_path = word[index + 1:]
            new_node = self.CompressedNode(remaining_path)
            new_node.is_end_of_word = True
            node.children[char] = new_node
    
    def _split_node(self, node: 'TrieCompressed.CompressedNode', split_index: int) -> None:
        if split_index >= len(node.compressed_path):
            return
        
        # Create new intermediate node
        remaining_path = node.compressed_path[split_index:]
        intermediate = self.CompressedNode(remaining_path)
        intermediate.children = node.children
        intermediate.is_end_of_word = node.is_end_of_word
        
        # Update original node
        node.compressed_path = node.compressed_path[:split_index]
        node.children = {remaining_path[0]: intermediate} if remaining_path else {}
        node.is_end_of_word = False
    
    def search(self, word: str) -> bool:
        node = self._find_node(word)
        return node is not None and node.is_end_of_word
    
    def startsWith(self, prefix: str) -> bool:
        return self._find_node(prefix) is not None
    
    def _find_node(self, target: str) -> Optional['TrieCompressed.CompressedNode']:
        return self._find_recursive(self.root, target, 0)
    
    def _find_recursive(self, node: 'TrieCompressed.CompressedNode', target: str, index: int) -> Optional['TrieCompressed.CompressedNode']:
        if index == len(target):
            return node
        
        char = target[index]
        if char not in node.children:
            return None
        
        child = node.children[char]
        remaining_target = target[index + 1:]
        
        # Check if compressed path matches
        if len(remaining_target) >= len(child.compressed_path):
            if remaining_target.startswith(child.compressed_path):
                return self._find_recursive(child, target, index + 1 + len(child.compressed_path))
        
        return None

class TrieWithFeatures:
    """
    Approach 5: Feature-Rich Trie
    
    Enhanced trie with additional features like word frequency, deletion, etc.
    
    Time Complexity: Same as basic operations
    Space Complexity: Additional space for features
    """
    
    class EnhancedNode:
        def __init__(self):
            self.children: Dict[str, 'TrieWithFeatures.EnhancedNode'] = {}
            self.is_end_of_word = False
            self.frequency = 0
            self.words_ending_here = []
    
    def __init__(self):
        self.root = self.EnhancedNode()
        self.word_count = 0
    
    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = self.EnhancedNode()
            node = node.children[char]
        
        if not node.is_end_of_word:
            self.word_count += 1
        
        node.is_end_of_word = True
        node.frequency += 1
        if word not in node.words_ending_here:
            node.words_ending_here.append(word)
    
    def search(self, word: str) -> bool:
        node = self._find_node(word)
        return node is not None and node.is_end_of_word
    
    def startsWith(self, prefix: str) -> bool:
        return self._find_node(prefix) is not None
    
    def delete(self, word: str) -> bool:
        """Delete a word from the trie"""
        if not self.search(word):
            return False
        
        def _delete_recursive(node: 'TrieWithFeatures.EnhancedNode', word: str, index: int) -> bool:
            if index == len(word):
                if not node.is_end_of_word:
                    return False
                
                node.is_end_of_word = False
                node.frequency = 0
                node.words_ending_here.clear()
                
                # Return True if current node has no children
                return len(node.children) == 0
            
            char = word[index]
            if char not in node.children:
                return False
            
            should_delete_child = _delete_recursive(node.children[char], word, index + 1)
            
            if should_delete_child:
                del node.children[char]
            
            # Return True if current node should be deleted
            return (not node.is_end_of_word and 
                    len(node.children) == 0)
        
        _delete_recursive(self.root, word, 0)
        self.word_count -= 1
        return True
    
    def get_all_words_with_prefix(self, prefix: str) -> List[str]:
        """Get all words that start with the given prefix"""
        node = self._find_node(prefix)
        if not node:
            return []
        
        words = []
        self._collect_words(node, prefix, words)
        return words
    
    def _collect_words(self, node: 'TrieWithFeatures.EnhancedNode', current_prefix: str, words: List[str]) -> None:
        if node.is_end_of_word:
            words.extend(node.words_ending_here)
        
        for char, child in node.children.items():
            self._collect_words(child, current_prefix + char, words)
    
    def get_word_frequency(self, word: str) -> int:
        """Get frequency of a word"""
        node = self._find_node(word)
        return node.frequency if node and node.is_end_of_word else 0
    
    def _find_node(self, prefix: str) -> Optional['TrieWithFeatures.EnhancedNode']:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def size(self) -> int:
        """Get total number of words in trie"""
        return self.word_count


def test_trie_basic_operations():
    """Test basic trie operations"""
    print("=== Testing Basic Trie Operations ===")
    
    implementations = [
        ("Basic Dictionary", TrieBasic),
        ("Object-Oriented", TrieObjectOriented),
        ("Array-based", TrieArray),
        ("Feature-Rich", TrieWithFeatures)
    ]
    
    test_words = ["apple", "app", "application", "apply", "banana"]
    
    for name, TrieClass in implementations:
        print(f"\n{name}:")
        
        trie = TrieClass()
        
        # Test insertions
        for word in test_words:
            trie.insert(word)
        
        # Test searches
        search_tests = ["app", "apple", "application", "orange", "ban", "banana"]
        for word in search_tests:
            result = trie.search(word)
            print(f"  search('{word}'): {result}")
        
        # Test prefix searches
        prefix_tests = ["app", "ban", "xyz"]
        for prefix in prefix_tests:
            result = trie.startsWith(prefix)
            print(f"  startsWith('{prefix}'): {result}")

def test_trie_advanced_features():
    """Test advanced trie features"""
    print("\n=== Testing Advanced Trie Features ===")
    
    # Test object-oriented trie with counting
    print("Object-Oriented Trie with Word Counting:")
    trie = TrieObjectOriented()
    
    words = ["cat", "car", "card", "care", "careful", "cars"]
    for word in words:
        trie.insert(word)
    
    prefixes = ["car", "care", "cat"]
    for prefix in prefixes:
        count = trie.count_words_with_prefix(prefix)
        print(f"  Words with prefix '{prefix}': {count}")
    
    # Test feature-rich trie
    print(f"\nFeature-Rich Trie:")
    feature_trie = TrieWithFeatures()
    
    # Insert with frequencies
    words_with_freq = [("hello", 3), ("world", 2), ("help", 1), ("hell", 1)]
    for word, freq in words_with_freq:
        for _ in range(freq):
            feature_trie.insert(word)
    
    print(f"  Total words: {feature_trie.size()}")
    
    for word, expected_freq in words_with_freq:
        actual_freq = feature_trie.get_word_frequency(word)
        print(f"  Frequency of '{word}': {actual_freq}")
    
    # Test getting words with prefix
    prefix = "hel"
    words_with_prefix = feature_trie.get_all_words_with_prefix(prefix)
    print(f"  Words with prefix '{prefix}': {words_with_prefix}")
    
    # Test deletion
    print(f"  Deleting 'help': {feature_trie.delete('help')}")
    print(f"  After deletion, size: {feature_trie.size()}")

def test_trie_performance():
    """Test trie performance"""
    print("\n=== Testing Trie Performance ===")
    
    import time
    import random
    import string
    
    def generate_random_words(count: int, min_length: int = 3, max_length: int = 10) -> List[str]:
        words = []
        for _ in range(count):
            length = random.randint(min_length, max_length)
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.append(word)
        return words
    
    implementations = [
        ("Basic Dictionary", TrieBasic),
        ("Object-Oriented", TrieObjectOriented),
        ("Array-based", TrieArray)
    ]
    
    word_count = 10000
    test_words = generate_random_words(word_count)
    
    for name, TrieClass in implementations:
        trie = TrieClass()
        
        # Test insertion performance
        start_time = time.time()
        for word in test_words:
            trie.insert(word)
        insert_time = (time.time() - start_time) * 1000
        
        # Test search performance
        search_words = random.sample(test_words, 1000) + generate_random_words(1000)
        
        start_time = time.time()
        hits = 0
        for word in search_words:
            if trie.search(word):
                hits += 1
        search_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    Insert {word_count} words: {insert_time:.2f}ms")
        print(f"    Search 2000 words: {search_time:.2f}ms ({hits} hits)")

def test_trie_memory_efficiency():
    """Test trie memory efficiency"""
    print("\n=== Testing Trie Memory Efficiency ===")
    
    # Test with words having common prefixes
    common_prefix_words = [
        f"programming{i}" for i in range(100)
    ] + [
        f"program{i}" for i in range(100)
    ] + [
        f"progress{i}" for i in range(100)
    ]
    
    implementations = [
        ("Basic Dictionary", TrieBasic),
        ("Object-Oriented", TrieObjectOriented),
        ("Feature-Rich", TrieWithFeatures)
    ]
    
    for name, TrieClass in implementations:
        trie = TrieClass()
        
        for word in common_prefix_words:
            trie.insert(word)
        
        # Rough memory estimation
        if hasattr(trie, 'word_count'):
            size_metric = trie.word_count
        else:
            size_metric = len(common_prefix_words)
        
        print(f"  {name}: Stored {len(common_prefix_words)} words")

def demonstrate_trie_applications():
    """Demonstrate real-world trie applications"""
    print("\n=== Demonstrating Trie Applications ===")
    
    # Application 1: Autocomplete system
    print("Application 1: Autocomplete System")
    autocomplete_trie = TrieWithFeatures()
    
    dictionary_words = [
        "python", "programming", "program", "programmer", "progress",
        "project", "product", "production", "productive", "procedure"
    ]
    
    for word in dictionary_words:
        autocomplete_trie.insert(word)
    
    user_input = "pro"
    suggestions = autocomplete_trie.get_all_words_with_prefix(user_input)
    print(f"  User types '{user_input}', suggestions: {suggestions[:5]}")
    
    # Application 2: Spell checker
    print(f"\nApplication 2: Spell Checker")
    spell_checker = TrieObjectOriented()
    
    valid_words = ["hello", "world", "check", "spell", "correct"]
    for word in valid_words:
        spell_checker.insert(word)
    
    test_words = ["hello", "helo", "wrold", "check", "spel"]
    for word in test_words:
        is_valid = spell_checker.search(word)
        print(f"  '{word}': {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    # Application 3: IP routing table
    print(f"\nApplication 3: IP Routing (Concept)")
    routing_trie = TrieBasic()
    
    # Simulate IP prefixes (simplified)
    ip_prefixes = ["192.168", "192.169", "10.0", "172.16"]
    for prefix in ip_prefixes:
        routing_trie.insert(prefix)
    
    test_ips = ["192.168", "192.169", "192.170", "10.0"]
    for ip in test_ips:
        has_route = routing_trie.startsWith(ip)
        print(f"  IP '{ip}': {'Route found' if has_route else 'No route'}")

def benchmark_trie_operations():
    """Benchmark trie operations"""
    print("\n=== Benchmarking Trie Operations ===")
    
    import time
    
    # Create large dataset
    word_count = 50000
    words = [f"word{i:05d}" for i in range(word_count)]
    
    trie = TrieObjectOriented()
    
    # Benchmark bulk insertion
    start_time = time.time()
    for word in words:
        trie.insert(word)
    insert_time = time.time() - start_time
    
    # Benchmark searches
    start_time = time.time()
    for i in range(0, word_count, 100):  # Sample every 100th word
        trie.search(words[i])
    search_time = time.time() - start_time
    
    # Benchmark prefix searches
    start_time = time.time()
    for i in range(0, word_count, 1000):  # Sample every 1000th word
        prefix = words[i][:4]  # Use first 4 characters as prefix
        trie.startsWith(prefix)
    prefix_time = time.time() - start_time
    
    print(f"Results for {word_count} words:")
    print(f"  Bulk insertion: {insert_time:.3f}s ({word_count/insert_time:.0f} words/sec)")
    print(f"  Search operations: {search_time:.3f}s")
    print(f"  Prefix operations: {prefix_time:.3f}s")

if __name__ == "__main__":
    test_trie_basic_operations()
    test_trie_advanced_features()
    test_trie_performance()
    test_trie_memory_efficiency()
    demonstrate_trie_applications()
    benchmark_trie_operations()

"""
Trie (Prefix Tree) Design demonstrates several important concepts:

Core Approaches:
1. Basic Dictionary - Simple and straightforward implementation
2. Object-Oriented - Clean separation with TrieNode class
3. Array-based - Memory-efficient for fixed alphabets
4. Compressed Trie - Space optimization through path compression
5. Feature-Rich - Enhanced with additional functionality

Key Design Principles:
- Efficient prefix-based operations
- Space-time tradeoffs in different implementations
- Extensibility for additional features

Real-world Applications:
- Autocomplete and typeahead systems
- Spell checkers and text suggestions
- IP routing tables
- Phone number directories
- File system path lookups
- DNA sequence analysis

The object-oriented approach with TrieNode is most commonly
preferred for its clarity and extensibility.
"""
