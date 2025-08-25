"""
208. Implement Trie (Prefix Tree) - Multiple Approaches
Difficulty: Medium

Implement a trie (prefix tree) with the following operations:
- Trie() initializes the trie object.
- void insert(String word) inserts the word into the trie.
- boolean search(String word) returns true if word is in the trie.
- boolean startsWith(String prefix) returns true if there is any word that starts with prefix.

LeetCode Problem: https://leetcode.com/problems/implement-trie-prefix-tree/
"""

from typing import List, Dict, Optional
from collections import defaultdict
import array

class TrieNode:
    """Standard Trie Node Implementation"""
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_word: bool = False

class Trie1:
    """
    Approach 1: Dictionary-based Trie
    
    Standard implementation using dictionary for children.
    
    Time Complexity:
    - insert: O(m) where m is the length of the word
    - search: O(m)
    - startsWith: O(m)
    
    Space Complexity: O(ALPHABET_SIZE * N * M) where N is number of words
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        node = self.root
        
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return node.is_end_of_word
    
    def startsWith(self, prefix: str) -> bool:
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return True

class TrieArrayNode:
    """Array-based Trie Node for lowercase letters"""
    def __init__(self):
        self.children: List[Optional['TrieArrayNode']] = [None] * 26
        self.is_end_of_word: bool = False

class Trie2:
    """
    Approach 2: Array-based Trie (lowercase letters only)
    
    More memory efficient for English lowercase letters.
    
    Time Complexity: Same as approach 1
    Space Complexity: Fixed 26 * N * M space
    """
    
    def __init__(self):
        self.root = TrieArrayNode()
    
    def _char_to_index(self, char: str) -> int:
        return ord(char) - ord('a')
    
    def insert(self, word: str) -> None:
        node = self.root
        
        for char in word:
            index = self._char_to_index(char)
            if node.children[index] is None:
                node.children[index] = TrieArrayNode()
            node = node.children[index]
        
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        node = self.root
        
        for char in word:
            index = self._char_to_index(char)
            if node.children[index] is None:
                return False
            node = node.children[index]
        
        return node.is_end_of_word
    
    def startsWith(self, prefix: str) -> bool:
        node = self.root
        
        for char in prefix:
            index = self._char_to_index(char)
            if node.children[index] is None:
                return False
            node = node.children[index]
        
        return True

class Trie3:
    """
    Approach 3: Set-based Implementation
    
    Simple implementation using sets for quick lookup.
    
    Time Complexity:
    - insert: O(m)
    - search: O(m)
    - startsWith: O(n * m) where n is number of words
    
    Space Complexity: O(total length of all words)
    """
    
    def __init__(self):
        self.words = set()
        self.prefixes = set()
    
    def insert(self, word: str) -> None:
        self.words.add(word)
        
        # Add all prefixes
        for i in range(1, len(word) + 1):
            self.prefixes.add(word[:i])
    
    def search(self, word: str) -> bool:
        return word in self.words
    
    def startsWith(self, prefix: str) -> bool:
        return prefix in self.prefixes

class Trie4:
    """
    Approach 4: Compressed Trie (Radix Tree)
    
    Space-optimized version that compresses single-child paths.
    
    Time Complexity: O(m) but with potential for better space efficiency
    Space Complexity: Reduced for strings with common prefixes
    """
    
    def __init__(self):
        self.root = {"path": "", "children": {}, "is_end": False}
    
    def insert(self, word: str) -> None:
        self._insert_recursive(self.root, word, 0)
    
    def _insert_recursive(self, node: dict, word: str, start_index: int) -> None:
        if start_index >= len(word):
            node["is_end"] = True
            return
        
        path = node["path"]
        
        # If this is a leaf with a path, we need to split
        if path and not node["children"]:
            if start_index < len(word) and word[start_index:].startswith(path):
                # Extend the path
                if len(word) - start_index == len(path):
                    node["is_end"] = True
                else:
                    # Create child for remaining part
                    remaining = word[start_index + len(path):]
                    char = remaining[0]
                    node["children"][char] = {
                        "path": remaining[1:],
                        "children": {},
                        "is_end": True
                    }
            else:
                # Split the path
                common_len = 0
                remaining_word = word[start_index:]
                while (common_len < len(path) and 
                       common_len < len(remaining_word) and
                       path[common_len] == remaining_word[common_len]):
                    common_len += 1
                
                if common_len > 0:
                    # Split the node
                    old_path = path[common_len:]
                    node["path"] = path[:common_len]
                    
                    # Create child for old path
                    if old_path:
                        first_char = old_path[0]
                        node["children"][first_char] = {
                            "path": old_path[1:],
                            "children": {},
                            "is_end": node["is_end"]
                        }
                    
                    node["is_end"] = False
                    
                    # Insert new word
                    if common_len < len(remaining_word):
                        self._insert_recursive(node, word, start_index + common_len)
                    else:
                        node["is_end"] = True
        else:
            # Navigate through existing path
            if path:
                remaining_word = word[start_index:]
                if remaining_word.startswith(path):
                    start_index += len(path)
                else:
                    # Need to split
                    self._split_and_insert(node, word, start_index)
                    return
            
            # Navigate to children
            if start_index < len(word):
                char = word[start_index]
                if char not in node["children"]:
                    node["children"][char] = {
                        "path": word[start_index + 1:],
                        "children": {},
                        "is_end": True
                    }
                else:
                    self._insert_recursive(node["children"][char], word, start_index + 1)
            else:
                node["is_end"] = True
    
    def _split_and_insert(self, node: dict, word: str, start_index: int) -> None:
        """Split a compressed path and insert new word"""
        path = node["path"]
        remaining_word = word[start_index:]
        
        # Find common prefix
        common_len = 0
        while (common_len < len(path) and 
               common_len < len(remaining_word) and
               path[common_len] == remaining_word[common_len]):
            common_len += 1
        
        if common_len > 0:
            # Split the path
            old_path = path[common_len:]
            node["path"] = path[:common_len]
            
            # Preserve old children
            old_children = node["children"]
            old_is_end = node["is_end"]
            
            # Reset current node
            node["children"] = {}
            node["is_end"] = False
            
            # Create child for old continuation
            if old_path:
                first_char = old_path[0]
                node["children"][first_char] = {
                    "path": old_path[1:],
                    "children": old_children,
                    "is_end": old_is_end
                }
            
            # Insert new word continuation
            if common_len < len(remaining_word):
                new_path = remaining_word[common_len:]
                first_char = new_path[0]
                node["children"][first_char] = {
                    "path": new_path[1:],
                    "children": {},
                    "is_end": True
                }
            else:
                node["is_end"] = True
    
    def search(self, word: str) -> bool:
        return self._search_recursive(self.root, word, 0)
    
    def _search_recursive(self, node: dict, word: str, start_index: int) -> bool:
        if start_index >= len(word):
            return node["is_end"]
        
        path = node["path"]
        remaining_word = word[start_index:]
        
        # Check if path matches
        if path:
            if remaining_word.startswith(path):
                return self._search_recursive(node, word, start_index + len(path))
            else:
                return False
        
        # Navigate to children
        char = word[start_index]
        if char in node["children"]:
            return self._search_recursive(node["children"][char], word, start_index + 1)
        
        return False
    
    def startsWith(self, prefix: str) -> bool:
        return self._starts_with_recursive(self.root, prefix, 0)
    
    def _starts_with_recursive(self, node: dict, prefix: str, start_index: int) -> bool:
        if start_index >= len(prefix):
            return True
        
        path = node["path"]
        remaining_prefix = prefix[start_index:]
        
        # Check if path matches
        if path:
            if len(remaining_prefix) <= len(path):
                return path.startswith(remaining_prefix)
            elif remaining_prefix.startswith(path):
                return self._starts_with_recursive(node, prefix, start_index + len(path))
            else:
                return False
        
        # Navigate to children
        char = prefix[start_index]
        if char in node["children"]:
            return self._starts_with_recursive(node["children"][char], prefix, start_index + 1)
        
        return False

class Trie5:
    """
    Approach 5: Bit-optimized Trie
    
    Uses bit manipulation for memory optimization.
    
    Time Complexity: O(m) with optimized constants
    Space Complexity: Optimized memory layout
    """
    
    def __init__(self):
        self.root = 0  # Start with root index 0
        self.nodes = [{"children": 0, "is_end": False}]  # Node storage
        self.char_maps = [{}]  # Character to child index mapping
    
    def _get_or_create_child(self, node_idx: int, char: str) -> int:
        """Get or create child node for character"""
        if char not in self.char_maps[node_idx]:
            # Create new node
            new_idx = len(self.nodes)
            self.nodes.append({"children": 0, "is_end": False})
            self.char_maps.append({})
            self.char_maps[node_idx][char] = new_idx
        
        return self.char_maps[node_idx][char]
    
    def insert(self, word: str) -> None:
        node_idx = self.root
        
        for char in word:
            node_idx = self._get_or_create_child(node_idx, char)
        
        self.nodes[node_idx]["is_end"] = True
    
    def search(self, word: str) -> bool:
        node_idx = self.root
        
        for char in word:
            if char not in self.char_maps[node_idx]:
                return False
            node_idx = self.char_maps[node_idx][char]
        
        return self.nodes[node_idx]["is_end"]
    
    def startsWith(self, prefix: str) -> bool:
        node_idx = self.root
        
        for char in prefix:
            if char not in self.char_maps[node_idx]:
                return False
            node_idx = self.char_maps[node_idx][char]
        
        return True

class Trie6:
    """
    Approach 6: Trie with Additional Operations
    
    Extended trie with useful utility methods.
    
    Additional Features:
    - Count words with prefix
    - Get all words with prefix
    - Delete operations
    - Longest common prefix
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0
    
    def insert(self, word: str) -> None:
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end_of_word:
            self.word_count += 1
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        node = self.root
        
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return node.is_end_of_word
    
    def startsWith(self, prefix: str) -> bool:
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return True
    
    def delete(self, word: str) -> bool:
        """Delete a word from the trie"""
        def _delete_helper(node: TrieNode, word: str, index: int) -> bool:
            if index == len(word):
                if not node.is_end_of_word:
                    return False
                
                node.is_end_of_word = False
                self.word_count -= 1
                return len(node.children) == 0
            
            char = word[index]
            if char not in node.children:
                return False
            
            should_delete = _delete_helper(node.children[char], word, index + 1)
            
            if should_delete:
                del node.children[char]
                return not node.is_end_of_word and len(node.children) == 0
            
            return False
        
        return _delete_helper(self.root, word, 0)
    
    def count_words_with_prefix(self, prefix: str) -> int:
        """Count words that start with prefix"""
        node = self.root
        
        # Navigate to prefix
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        
        # Count words in subtree
        def count_words(node: TrieNode) -> int:
            count = 1 if node.is_end_of_word else 0
            for child in node.children.values():
                count += count_words(child)
            return count
        
        return count_words(node)
    
    def get_words_with_prefix(self, prefix: str) -> List[str]:
        """Get all words that start with prefix"""
        node = self.root
        
        # Navigate to prefix
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Collect words
        words = []
        
        def collect_words(node: TrieNode, current_word: str):
            if node.is_end_of_word:
                words.append(current_word)
            
            for char, child in node.children.items():
                collect_words(child, current_word + char)
        
        collect_words(node, prefix)
        return words
    
    def longest_common_prefix(self, words: List[str]) -> str:
        """Find longest common prefix of all words in trie"""
        if not words:
            return ""
        
        # Insert all words
        for word in words:
            self.insert(word)
        
        # Find LCP by traversing trie
        lcp = ""
        node = self.root
        
        while len(node.children) == 1 and not node.is_end_of_word:
            char = next(iter(node.children.keys()))
            lcp += char
            node = node.children[char]
        
        return lcp


def test_basic_operations():
    """Test basic trie operations with different implementations"""
    print("=== Testing Trie Implementations ===")
    
    implementations = [
        ("Dictionary-based", Trie1),
        ("Array-based", Trie2),
        ("Set-based", Trie3),
        ("Compressed", Trie4),
        ("Bit-optimized", Trie5),
        ("Extended", Trie6),
    ]
    
    test_words = ["apple", "app", "apricot", "banana", "band", "bandana"]
    
    for name, TrieClass in implementations:
        print(f"\n--- Testing {name} ---")
        
        trie = TrieClass()
        
        # Insert words
        for word in test_words:
            trie.insert(word)
        
        # Test search
        search_tests = ["apple", "app", "appl", "banana", "xyz"]
        print("Search results:")
        for word in search_tests:
            result = trie.search(word)
            print(f"  {word}: {result}")
        
        # Test startsWith
        prefix_tests = ["app", "ban", "xyz"]
        print("Prefix results:")
        for prefix in prefix_tests:
            result = trie.startsWith(prefix)
            print(f"  {prefix}: {result}")


def test_extended_operations():
    """Test extended trie operations"""
    print("\n=== Testing Extended Operations ===")
    
    trie = Trie6()
    
    words = ["cat", "cats", "dog", "dogs", "doggy", "door", "doors"]
    for word in words:
        trie.insert(word)
    
    print(f"Total words: {trie.word_count}")
    
    # Test prefix counting
    test_prefixes = ["cat", "dog", "do", "x"]
    for prefix in test_prefixes:
        count = trie.count_words_with_prefix(prefix)
        words_list = trie.get_words_with_prefix(prefix)
        print(f"Prefix '{prefix}': {count} words -> {words_list}")
    
    # Test deletion
    print(f"\nBefore deletion: {trie.get_words_with_prefix('')}")
    trie.delete("cats")
    print(f"After deleting 'cats': {trie.get_words_with_prefix('')}")
    
    # Test LCP
    lcp_words = ["flower", "flow", "flight"]
    lcp = trie.longest_common_prefix(lcp_words)
    print(f"LCP of {lcp_words}: '{lcp}'")


def test_leetcode_examples():
    """Test with LeetCode examples"""
    print("\n=== Testing LeetCode Examples ===")
    
    trie = Trie1()
    
    # Example operations
    operations = [
        ("insert", "apple"),
        ("search", "apple"),    # True
        ("search", "app"),      # False
        ("startsWith", "app"),  # True
        ("insert", "app"),
        ("search", "app"),      # True
    ]
    
    for operation, word in operations:
        if operation == "insert":
            trie.insert(word)
            print(f"insert('{word}')")
        elif operation == "search":
            result = trie.search(word)
            print(f"search('{word}') -> {result}")
        elif operation == "startsWith":
            result = trie.startsWith(word)
            print(f"startsWith('{word}') -> {result}")


def benchmark_approaches():
    """Benchmark different trie implementations"""
    print("\n=== Benchmarking Trie Implementations ===")
    
    import time
    import random
    import string
    
    # Generate test data
    words = []
    for _ in range(1000):
        length = random.randint(3, 10)
        word = ''.join(random.choices(string.ascii_lowercase, k=length))
        words.append(word)
    
    implementations = [
        ("Dictionary", Trie1),
        ("Array", Trie2),
        ("Set", Trie3),
        ("Extended", Trie6),
    ]
    
    for name, TrieClass in implementations:
        start_time = time.time()
        
        trie = TrieClass()
        
        # Insert all words
        for word in words:
            trie.insert(word)
        
        # Search operations
        for word in words[:100]:
            trie.search(word)
        
        # Prefix operations
        for word in words[:50]:
            trie.startsWith(word[:3])
        
        end_time = time.time()
        
        print(f"{name:12}: {(end_time - start_time)*1000:.2f} ms")


def demonstrate_real_world_usage():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Trie Applications ===")
    
    # Application 1: Autocomplete
    print("1. Autocomplete System:")
    autocomplete = Trie6()
    
    # Add programming terms
    terms = [
        "python", "programming", "program", "project", "print",
        "java", "javascript", "json", "jupyter",
        "algorithm", "array", "api", "application"
    ]
    
    for term in terms:
        autocomplete.insert(term)
    
    def get_autocomplete_suggestions(prefix: str, max_suggestions: int = 5) -> List[str]:
        return autocomplete.get_words_with_prefix(prefix)[:max_suggestions]
    
    test_prefixes = ["pro", "java", "a"]
    for prefix in test_prefixes:
        suggestions = get_autocomplete_suggestions(prefix)
        print(f"   '{prefix}' -> {suggestions}")
    
    # Application 2: Dictionary/Spell Checker
    print("\n2. Dictionary Lookup:")
    dictionary = Trie1()
    
    # Add common English words
    common_words = [
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "her", "was", "one", "our", "out", "day", "get", "has", "him", "his"
    ]
    
    for word in common_words:
        dictionary.insert(word)
    
    def is_valid_word(word: str) -> bool:
        return dictionary.search(word.lower())
    
    test_words = ["the", "programming", "xyz", "and"]
    for word in test_words:
        valid = is_valid_word(word)
        print(f"   '{word}' is valid: {valid}")
    
    # Application 3: IP Address/URL Routing
    print("\n3. URL Routing:")
    router = Trie1()
    
    # Add URL patterns
    routes = [
        "/api/users",
        "/api/users/profile",
        "/api/posts",
        "/api/posts/comments",
        "/admin/dashboard",
        "/static/css",
        "/static/js"
    ]
    
    for route in routes:
        router.insert(route)
    
    def route_exists(url: str) -> bool:
        return router.search(url)
    
    def get_route_suggestions(prefix: str) -> List[str]:
        if hasattr(router, 'get_words_with_prefix'):
            return router.get_words_with_prefix(prefix)
        return []
    
    test_urls = ["/api/users", "/api/xyz", "/admin"]
    for url in test_urls:
        exists = route_exists(url)
        print(f"   Route '{url}' exists: {exists}")


if __name__ == "__main__":
    test_basic_operations()
    test_extended_operations()
    test_leetcode_examples()
    benchmark_approaches()
    demonstrate_real_world_usage()

"""
208. Implement Trie (Prefix Tree) showcases multiple approaches to trie implementation:
- Dictionary-based for flexibility and ease of implementation
- Array-based for memory efficiency with fixed alphabet
- Set-based for simple implementation with different trade-offs
- Compressed for space optimization with complex strings
- Bit-optimized for memory-constrained environments
- Extended with additional utility operations for real-world usage

Each approach demonstrates different optimization strategies and use cases.
"""
