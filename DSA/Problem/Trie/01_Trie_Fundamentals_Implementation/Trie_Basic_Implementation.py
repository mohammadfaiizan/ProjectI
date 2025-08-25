"""
Trie Basic Implementation - Multiple Approaches
Difficulty: Easy

Implement a basic Trie (Prefix Tree) data structure with fundamental operations:
insert, search, starts_with, and various traversal methods.
"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
import json

class TrieNode:
    """Basic Trie Node with character and end marker"""
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_word: bool = False
        self.word_count: int = 0  # For counting word frequencies

class TrieArrayNode:
    """Array-based Trie Node for lowercase letters only"""
    def __init__(self):
        self.children: List[Optional['TrieArrayNode']] = [None] * 26
        self.is_end_of_word: bool = False
        self.word_count: int = 0

class TrieBasic:
    """
    Approach 1: Dictionary-based Trie
    
    Uses dictionary to store children, flexible for any character set.
    
    Time Complexity:
    - Insert: O(m) where m is word length
    - Search: O(m)
    - StartsWith: O(m)
    
    Space Complexity: O(ALPHABET_SIZE * N * M) where N is number of words
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0
    
    def insert(self, word: str) -> None:
        """Insert a word into the trie"""
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end_of_word:
            self.word_count += 1
        
        node.is_end_of_word = True
        node.word_count += 1
    
    def search(self, word: str) -> bool:
        """Search for a complete word in the trie"""
        node = self.root
        
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return node.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with the given prefix"""
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
                node.word_count = 0
                self.word_count -= 1
                
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
        
        return _delete_helper(self.root, word, 0)
    
    def get_all_words(self) -> List[str]:
        """Get all words in the trie"""
        words = []
        
        def dfs(node: TrieNode, prefix: str):
            if node.is_end_of_word:
                words.append(prefix)
            
            for char, child in node.children.items():
                dfs(child, prefix + char)
        
        dfs(self.root, "")
        return words
    
    def get_words_with_prefix(self, prefix: str) -> List[str]:
        """Get all words that start with the given prefix"""
        # Navigate to prefix node
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Collect all words from this node
        words = []
        
        def dfs(node: TrieNode, current_prefix: str):
            if node.is_end_of_word:
                words.append(current_prefix)
            
            for char, child in node.children.items():
                dfs(child, current_prefix + char)
        
        dfs(node, prefix)
        return words

class TrieArray:
    """
    Approach 2: Array-based Trie (for lowercase letters only)
    
    Uses array of size 26 for children, memory efficient for English words.
    
    Time Complexity: Same as dictionary-based
    Space Complexity: More memory efficient for lowercase letters
    """
    
    def __init__(self):
        self.root = TrieArrayNode()
        self.word_count = 0
    
    def _char_to_index(self, char: str) -> int:
        """Convert character to array index"""
        return ord(char) - ord('a')
    
    def insert(self, word: str) -> None:
        """Insert a word into the trie"""
        node = self.root
        
        for char in word.lower():
            if not char.isalpha():
                continue
                
            index = self._char_to_index(char)
            if node.children[index] is None:
                node.children[index] = TrieArrayNode()
            node = node.children[index]
        
        if not node.is_end_of_word:
            self.word_count += 1
        
        node.is_end_of_word = True
        node.word_count += 1
    
    def search(self, word: str) -> bool:
        """Search for a complete word in the trie"""
        node = self.root
        
        for char in word.lower():
            if not char.isalpha():
                continue
                
            index = self._char_to_index(char)
            if node.children[index] is None:
                return False
            node = node.children[index]
        
        return node.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with the given prefix"""
        node = self.root
        
        for char in prefix.lower():
            if not char.isalpha():
                continue
                
            index = self._char_to_index(char)
            if node.children[index] is None:
                return False
            node = node.children[index]
        
        return True
    
    def get_all_words(self) -> List[str]:
        """Get all words in the trie"""
        words = []
        
        def dfs(node: TrieArrayNode, prefix: str):
            if node.is_end_of_word:
                words.append(prefix)
            
            for i in range(26):
                if node.children[i] is not None:
                    char = chr(ord('a') + i)
                    dfs(node.children[i], prefix + char)
        
        dfs(self.root, "")
        return words

class TrieCompressed:
    """
    Approach 3: Compressed Trie (Radix Tree)
    
    Compresses chains of single-child nodes to reduce memory usage.
    
    Time Complexity: O(m) for operations, but with better space efficiency
    Space Complexity: Reduced by compressing single-child chains
    """
    
    def __init__(self):
        self.root = {"children": {}, "is_end": False, "compressed_path": ""}
        self.word_count = 0
    
    def insert(self, word: str) -> None:
        """Insert a word with path compression"""
        node = self.root
        i = 0
        
        while i < len(word):
            # Skip compressed path if it matches
            if node["compressed_path"]:
                path = node["compressed_path"]
                if i + len(path) <= len(word) and word[i:i+len(path)] == path:
                    i += len(path)
                else:
                    # Split the compressed path
                    self._split_compressed_path(node, word, i)
                    continue
            
            if i >= len(word):
                break
            
            char = word[i]
            if char not in node["children"]:
                # Create new node with potential compression
                remaining = word[i:]
                new_node = {"children": {}, "is_end": True, "compressed_path": remaining}
                node["children"][char] = new_node
                self.word_count += 1
                return
            
            node = node["children"][char]
            i += 1
        
        if not node["is_end"]:
            self.word_count += 1
        node["is_end"] = True
    
    def _split_compressed_path(self, node: dict, word: str, word_index: int) -> None:
        """Split a compressed path when inserting a diverging word"""
        path = node["compressed_path"]
        
        # Find common prefix
        common_len = 0
        while (common_len < len(path) and 
               word_index + common_len < len(word) and
               path[common_len] == word[word_index + common_len]):
            common_len += 1
        
        if common_len > 0:
            # Create intermediate node
            old_children = node["children"]
            old_is_end = node["is_end"]
            
            # Update current node
            node["compressed_path"] = path[:common_len]
            node["children"] = {}
            node["is_end"] = False
            
            # Create child for old path continuation
            if common_len < len(path):
                remaining_path = path[common_len:]
                old_node = {
                    "children": old_children,
                    "is_end": old_is_end,
                    "compressed_path": remaining_path[1:] if len(remaining_path) > 1 else ""
                }
                node["children"][remaining_path[0]] = old_node
    
    def search(self, word: str) -> bool:
        """Search for a complete word"""
        node = self.root
        i = 0
        
        while i < len(word):
            if node["compressed_path"]:
                path = node["compressed_path"]
                if i + len(path) <= len(word) and word[i:i+len(path)] == path:
                    i += len(path)
                else:
                    return False
            
            if i >= len(word):
                break
            
            char = word[i]
            if char not in node["children"]:
                return False
            
            node = node["children"][char]
            i += 1
        
        return node["is_end"]
    
    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with the given prefix"""
        node = self.root
        i = 0
        
        while i < len(prefix):
            if node["compressed_path"]:
                path = node["compressed_path"]
                remaining_prefix = prefix[i:]
                
                if len(remaining_prefix) <= len(path):
                    return path.startswith(remaining_prefix)
                elif path == remaining_prefix[:len(path)]:
                    i += len(path)
                else:
                    return False
            
            if i >= len(prefix):
                break
            
            char = prefix[i]
            if char not in node["children"]:
                return False
            
            node = node["children"][char]
            i += 1
        
        return True

class TrieWithFrequency:
    """
    Approach 4: Trie with Frequency Tracking
    
    Tracks word frequencies and provides sorted results.
    
    Additional Features:
    - Word frequency counting
    - Most frequent words retrieval
    - Frequency-based autocomplete
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.word_frequencies: Dict[str, int] = defaultdict(int)
        self.total_words = 0
    
    def insert(self, word: str, frequency: int = 1) -> None:
        """Insert a word with frequency"""
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end_of_word:
            node.is_end_of_word = True
        
        self.word_frequencies[word] += frequency
        node.word_count = self.word_frequencies[word]
        self.total_words += frequency
    
    def get_frequency(self, word: str) -> int:
        """Get frequency of a word"""
        return self.word_frequencies.get(word, 0)
    
    def get_most_frequent_words(self, n: int) -> List[Tuple[str, int]]:
        """Get n most frequent words"""
        return sorted(self.word_frequencies.items(), 
                     key=lambda x: x[1], reverse=True)[:n]
    
    def get_suggestions_by_frequency(self, prefix: str, n: int = 5) -> List[Tuple[str, int]]:
        """Get autocomplete suggestions sorted by frequency"""
        words_with_prefix = []
        
        # Navigate to prefix node
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Collect all words from this node with frequencies
        def dfs(node: TrieNode, current_prefix: str):
            if node.is_end_of_word:
                words_with_prefix.append((current_prefix, self.word_frequencies[current_prefix]))
            
            for char, child in node.children.items():
                dfs(child, current_prefix + char)
        
        dfs(node, prefix)
        
        # Sort by frequency and return top n
        return sorted(words_with_prefix, key=lambda x: x[1], reverse=True)[:n]

class TriePersistent:
    """
    Approach 5: Persistent Trie with Serialization
    
    Supports saving and loading trie to/from file.
    
    Features:
    - JSON serialization
    - File persistence
    - Memory snapshot
    """
    
    def __init__(self):
        self.trie = TrieBasic()
    
    def insert(self, word: str) -> None:
        """Insert word"""
        self.trie.insert(word)
    
    def search(self, word: str) -> bool:
        """Search word"""
        return self.trie.search(word)
    
    def starts_with(self, prefix: str) -> bool:
        """Check prefix"""
        return self.trie.starts_with(prefix)
    
    def save_to_file(self, filename: str) -> None:
        """Save trie to JSON file"""
        data = {
            "words": self.trie.get_all_words(),
            "word_count": self.trie.word_count
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filename: str) -> None:
        """Load trie from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.trie = TrieBasic()
            for word in data["words"]:
                self.trie.insert(word)
                
        except FileNotFoundError:
            print(f"File {filename} not found")
        except json.JSONDecodeError:
            print(f"Invalid JSON in file {filename}")
    
    def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        def count_nodes(node: TrieNode) -> int:
            count = 1
            for child in node.children.values():
                count += count_nodes(child)
            return count
        
        total_nodes = count_nodes(self.trie.root)
        
        return {
            "total_nodes": total_nodes,
            "total_words": self.trie.word_count,
            "estimated_memory_kb": total_nodes * 0.1,  # Rough estimate
        }


def test_basic_operations():
    """Test basic trie operations"""
    print("=== Testing Basic Trie Operations ===")
    
    # Test dictionary-based trie
    trie = TrieBasic()
    
    # Insert words
    words = ["apple", "app", "application", "apply", "banana", "band", "bandana"]
    for word in words:
        trie.insert(word)
    
    print(f"Inserted words: {words}")
    print(f"Total words in trie: {trie.word_count}")
    
    # Test search
    test_words = ["apple", "app", "appl", "banana", "ban", "xyz"]
    print("\n--- Search Tests ---")
    for word in test_words:
        result = trie.search(word)
        print(f"Search '{word}': {result}")
    
    # Test prefix
    test_prefixes = ["app", "ban", "xyz", "a"]
    print("\n--- Prefix Tests ---")
    for prefix in test_prefixes:
        result = trie.starts_with(prefix)
        words_with_prefix = trie.get_words_with_prefix(prefix)
        print(f"Prefix '{prefix}': {result} -> {words_with_prefix}")
    
    # Test delete
    print("\n--- Delete Tests ---")
    print(f"Before delete: {trie.get_all_words()}")
    trie.delete("app")
    print(f"After deleting 'app': {trie.get_all_words()}")


def test_array_based_trie():
    """Test array-based trie"""
    print("\n=== Testing Array-Based Trie ===")
    
    trie = TrieArray()
    
    words = ["hello", "world", "help", "hero", "her"]
    for word in words:
        trie.insert(word)
    
    print(f"Inserted words: {words}")
    
    test_words = ["hello", "help", "he", "world", "word"]
    for word in test_words:
        search_result = trie.search(word)
        prefix_result = trie.starts_with(word)
        print(f"'{word}': search={search_result}, prefix={prefix_result}")
    
    print(f"All words: {trie.get_all_words()}")


def test_compressed_trie():
    """Test compressed trie"""
    print("\n=== Testing Compressed Trie ===")
    
    trie = TrieCompressed()
    
    words = ["testing", "test", "tea", "ted", "ten", "i", "in", "inn"]
    for word in words:
        trie.insert(word)
    
    print(f"Inserted words: {words}")
    
    test_words = ["test", "testing", "tea", "te", "xyz"]
    for word in test_words:
        search_result = trie.search(word)
        prefix_result = trie.starts_with(word)
        print(f"'{word}': search={search_result}, prefix={prefix_result}")


def test_frequency_trie():
    """Test frequency-tracking trie"""
    print("\n=== Testing Frequency Trie ===")
    
    trie = TrieWithFrequency()
    
    # Insert words with frequencies
    word_freq = [("apple", 10), ("app", 15), ("application", 5), ("apply", 8), ("banana", 12)]
    for word, freq in word_freq:
        trie.insert(word, freq)
    
    print("Inserted words with frequencies:")
    for word, freq in word_freq:
        print(f"  {word}: {freq}")
    
    print(f"\nMost frequent words: {trie.get_most_frequent_words(3)}")
    
    print(f"\nSuggestions for 'app': {trie.get_suggestions_by_frequency('app', 3)}")


def test_persistent_trie():
    """Test persistent trie with file operations"""
    print("\n=== Testing Persistent Trie ===")
    
    trie = TriePersistent()
    
    words = ["save", "load", "persist", "memory", "file"]
    for word in words:
        trie.insert(word)
    
    print(f"Created trie with words: {words}")
    
    # Test memory usage
    memory_stats = trie.get_memory_usage()
    print(f"Memory usage: {memory_stats}")
    
    # Note: File operations would require actual file I/O
    # In a real implementation, you would test:
    # trie.save_to_file("test_trie.json")
    # new_trie = TriePersistent()
    # new_trie.load_from_file("test_trie.json")


def benchmark_implementations():
    """Benchmark different trie implementations"""
    print("\n=== Benchmarking Trie Implementations ===")
    
    import time
    
    # Test data
    words = [f"word{i}" for i in range(1000)]
    
    implementations = [
        ("Dictionary-based", TrieBasic),
        ("Array-based", TrieArray),
        ("Compressed", TrieCompressed),
    ]
    
    for name, TrieClass in implementations:
        start_time = time.time()
        
        trie = TrieClass()
        
        # Insert
        for word in words:
            trie.insert(word)
        
        # Search
        for word in words[:100]:  # Test subset
            trie.search(word)
        
        # Prefix
        for word in words[:50]:  # Test subset
            trie.starts_with(word[:3])
        
        end_time = time.time()
        
        print(f"{name:15}: {(end_time - start_time)*1000:.2f} ms")


def demonstrate_real_world_usage():
    """Demonstrate real-world trie applications"""
    print("\n=== Real-World Trie Applications ===")
    
    # Application 1: Autocomplete System
    print("1. Autocomplete System:")
    autocomplete = TrieWithFrequency()
    
    # Simulate search queries with frequencies
    queries = [
        ("python", 100), ("programming", 80), ("program", 90), 
        ("project", 70), ("print", 60), ("process", 50)
    ]
    
    for query, freq in queries:
        autocomplete.insert(query, freq)
    
    prefix = "pro"
    suggestions = autocomplete.get_suggestions_by_frequency(prefix, 3)
    print(f"   Autocomplete for '{prefix}': {suggestions}")
    
    # Application 2: Spell Checker
    print("\n2. Basic Spell Checker:")
    dictionary = TrieBasic()
    
    # Add dictionary words
    dict_words = ["hello", "world", "help", "held", "hell", "shell"]
    for word in dict_words:
        dictionary.insert(word)
    
    def simple_spell_check(word: str) -> List[str]:
        if dictionary.search(word):
            return [word]
        
        # Find words with similar prefixes
        suggestions = []
        for i in range(1, len(word) + 1):
            prefix = word[:i]
            if dictionary.starts_with(prefix):
                words_with_prefix = dictionary.get_words_with_prefix(prefix)
                suggestions.extend(words_with_prefix[:3])
        
        return list(set(suggestions))
    
    test_word = "helo"
    suggestions = simple_spell_check(test_word)
    print(f"   Spell check for '{test_word}': {suggestions}")
    
    # Application 3: URL/Path Processing
    print("\n3. URL Path Processing:")
    url_trie = TrieBasic()
    
    urls = [
        "/api/users",
        "/api/users/profile",
        "/api/posts",
        "/api/posts/comments",
        "/admin/dashboard",
        "/admin/users"
    ]
    
    for url in urls:
        url_trie.insert(url)
    
    def find_matching_routes(path: str) -> List[str]:
        return url_trie.get_words_with_prefix(path)
    
    test_path = "/api"
    matching_routes = find_matching_routes(test_path)
    print(f"   Routes starting with '{test_path}': {matching_routes}")


if __name__ == "__main__":
    test_basic_operations()
    test_array_based_trie()
    test_compressed_trie()
    test_frequency_trie()
    test_persistent_trie()
    benchmark_implementations()
    demonstrate_real_world_usage()

"""
Trie Basic Implementation demonstrates comprehensive trie functionality
with multiple approaches optimized for different use cases:
- Dictionary-based for flexibility
- Array-based for memory efficiency
- Compressed for space optimization
- Frequency-tracking for autocomplete systems
- Persistent for data storage applications
"""
