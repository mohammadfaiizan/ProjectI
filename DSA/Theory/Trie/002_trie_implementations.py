"""
Trie Implementations - Various Implementation Strategies
=======================================================

Topics: Array-based, HashMap-based, optimized implementations
Companies: Google, Amazon, Microsoft, Facebook, Apple
Difficulty: Medium to Hard
Time Complexity: O(m) for basic operations where m is string length
Space Complexity: Varies by implementation strategy
"""

from typing import List, Optional, Dict, Any, Set, Union
from collections import defaultdict
import array

class TrieImplementations:
    
    def __init__(self):
        """Initialize with implementation tracking"""
        self.implementation_count = 0
        self.performance_data = {}
    
    # ==========================================
    # 1. ARRAY-BASED TRIE IMPLEMENTATION
    # ==========================================
    
    def demonstrate_array_based_trie(self) -> None:
        """
        Demonstrate array-based trie implementation
        
        Best for: Fixed, small alphabets (e.g., lowercase letters)
        Memory: More predictable, better cache performance
        """
        print("=== ARRAY-BASED TRIE IMPLEMENTATION ===")
        print("Best for: Fixed alphabets (e.g., 26 lowercase letters)")
        print("Advantages: Better cache performance, predictable memory layout")
        print("Disadvantages: Wastes space for sparse alphabets")
        print()
        
        trie = ArrayBasedTrie()
        
        # Test words
        words = ["apple", "app", "application", "apply", "apt"]
        
        print("Inserting words:")
        for word in words:
            print(f"Inserting '{word}':")
            trie.insert(word)
            print()
        
        print("Trie structure visualization:")
        trie.display_structure()
        
        print("\nSearch operations:")
        test_words = ["app", "apple", "apply", "ap", "missing"]
        for word in test_words:
            result = trie.search(word)
            print(f"  '{word}': {'Found' if result else 'Not found'}")
        
        print("\nPrefix operations:")
        prefixes = ["ap", "app", "application", "xyz"]
        for prefix in prefixes:
            result = trie.starts_with(prefix)
            print(f"  Starts with '{prefix}': {'Yes' if result else 'No'}")


class ArrayBasedTrieNode:
    """
    Array-based trie node for fixed alphabet (26 lowercase letters)
    
    Uses array for O(1) child access
    More memory efficient for dense alphabets
    """
    
    def __init__(self):
        self.ALPHABET_SIZE = 26
        self.children = [None] * self.ALPHABET_SIZE  # Array of child nodes
        self.is_end_of_word = False
        self.word = None  # Optional: store the complete word
    
    def _char_to_index(self, char: str) -> int:
        """Convert character to array index (a=0, b=1, ..., z=25)"""
        return ord(char.lower()) - ord('a')
    
    def _index_to_char(self, index: int) -> str:
        """Convert array index to character"""
        return chr(ord('a') + index)
    
    def has_child(self, char: str) -> bool:
        """Check if child exists for given character"""
        index = self._char_to_index(char)
        return 0 <= index < self.ALPHABET_SIZE and self.children[index] is not None
    
    def get_child(self, char: str) -> Optional['ArrayBasedTrieNode']:
        """Get child node for given character"""
        if not self.has_child(char):
            return None
        index = self._char_to_index(char)
        return self.children[index]
    
    def set_child(self, char: str, node: 'ArrayBasedTrieNode') -> None:
        """Set child node for given character"""
        index = self._char_to_index(char)
        if 0 <= index < self.ALPHABET_SIZE:
            self.children[index] = node
    
    def get_children_chars(self) -> List[str]:
        """Get list of characters that have children"""
        chars = []
        for i in range(self.ALPHABET_SIZE):
            if self.children[i] is not None:
                chars.append(self._index_to_char(i))
        return chars


class ArrayBasedTrie:
    """
    Array-based trie implementation
    
    Optimized for lowercase English letters (a-z)
    Better cache performance due to array-based storage
    """
    
    def __init__(self):
        self.root = ArrayBasedTrieNode()
        self.word_count = 0
    
    def insert(self, word: str) -> None:
        """
        Insert word into array-based trie
        
        Time: O(m), Space: O(m) worst case
        """
        if not word or not word.isalpha():
            print(f"    Invalid word '{word}' - only alphabetic characters allowed")
            return
        
        word = word.lower()
        current = self.root
        path = []
        
        print(f"    Tracing path for '{word}':")
        
        for i, char in enumerate(word):
            path.append(char)
            
            if not current.has_child(char):
                current.set_child(char, ArrayBasedTrieNode())
                print(f"      Step {i+1}: Created new node for '{char}' at '{(''.join(path))}'")
            else:
                print(f"      Step {i+1}: Found existing node for '{char}' at '{(''.join(path))}'")
            
            current = current.get_child(char)
        
        if not current.is_end_of_word:
            current.is_end_of_word = True
            current.word = word
            self.word_count += 1
            print(f"    Marked end of word: '{word}'")
        else:
            print(f"    Word '{word}' already exists")
    
    def search(self, word: str) -> bool:
        """Search for word in trie"""
        if not word or not word.isalpha():
            return False
        
        word = word.lower()
        current = self.root
        
        for char in word:
            if not current.has_child(char):
                return False
            current = current.get_child(char)
        
        return current.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with prefix"""
        if not prefix or not prefix.isalpha():
            return False
        
        prefix = prefix.lower()
        current = self.root
        
        for char in prefix:
            if not current.has_child(char):
                return False
            current = current.get_child(char)
        
        return True
    
    def display_structure(self) -> None:
        """Display the array-based trie structure"""
        print("  Array-based Trie Structure:")
        
        def _display_helper(node: ArrayBasedTrieNode, prefix: str, depth: int):
            indent = "    " + "  " * depth
            
            if node.is_end_of_word:
                print(f"{indent}'{prefix}' [WORD]")
            
            children_chars = node.get_children_chars()
            for char in children_chars:
                child = node.get_child(char)
                print(f"{indent}├── '{char}'")
                _display_helper(child, prefix + char, depth + 1)
        
        if not any(self.root.children):
            print("    (empty)")
        else:
            _display_helper(self.root, "", 0)
        
        print(f"    Total words: {self.word_count}")


# ==========================================
# 2. HASHMAP-BASED TRIE IMPLEMENTATION
# ==========================================

class HashMapBasedTrieNode:
    """
    HashMap-based trie node for variable alphabets
    
    Uses dictionary for flexible character sets
    More memory efficient for sparse alphabets
    """
    
    def __init__(self):
        self.children = {}  # Dict[str, HashMapBasedTrieNode]
        self.is_end_of_word = False
        self.frequency = 0  # Word frequency for suggestions
        self.last_accessed = 0  # For LRU operations


class HashMapBasedTrie:
    """
    HashMap-based trie implementation
    
    Supports any character set (Unicode, special characters, etc.)
    More flexible but potentially higher memory overhead per node
    """
    
    def __init__(self, case_sensitive: bool = False):
        self.root = HashMapBasedTrieNode()
        self.word_count = 0
        self.case_sensitive = case_sensitive
        self.access_counter = 0
    
    def _normalize_word(self, word: str) -> str:
        """Normalize word based on case sensitivity setting"""
        return word if self.case_sensitive else word.lower()
    
    def insert(self, word: str) -> None:
        """
        Insert word into HashMap-based trie
        
        Supports any character set including Unicode
        """
        if not word:
            return
        
        word = self._normalize_word(word)
        current = self.root
        
        print(f"    Inserting '{word}' (case_sensitive={self.case_sensitive}):")
        
        for i, char in enumerate(word):
            if char not in current.children:
                current.children[char] = HashMapBasedTrieNode()
                print(f"      Created node for '{char}' at position {i}")
            else:
                print(f"      Found existing node for '{char}' at position {i}")
            
            current = current.children[char]
        
        if not current.is_end_of_word:
            current.is_end_of_word = True
            self.word_count += 1
            print(f"      Marked end of word")
        
        current.frequency += 1
        current.last_accessed = self.access_counter
        self.access_counter += 1
        print(f"      Updated frequency to {current.frequency}")
    
    def search(self, word: str) -> bool:
        """Search with frequency tracking"""
        if not word:
            return False
        
        word = self._normalize_word(word)
        current = self.root
        
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]
        
        if current.is_end_of_word:
            current.last_accessed = self.access_counter
            self.access_counter += 1
        
        return current.is_end_of_word
    
    def get_word_frequency(self, word: str) -> int:
        """Get frequency of a word"""
        if not word:
            return 0
        
        word = self._normalize_word(word)
        current = self.root
        
        for char in word:
            if char not in current.children:
                return 0
            current = current.children[char]
        
        return current.frequency if current.is_end_of_word else 0
    
    def get_words_with_prefix(self, prefix: str, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get words with given prefix, sorted by frequency
        
        Returns list of (word, frequency) tuples
        """
        if not prefix:
            return []
        
        prefix = self._normalize_word(prefix)
        current = self.root
        
        # Navigate to prefix node
        for char in prefix:
            if char not in current.children:
                return []
            current = current.children[char]
        
        # Collect all words in subtree
        words = []
        
        def _collect_words(node: HashMapBasedTrieNode, current_word: str):
            if node.is_end_of_word:
                words.append((current_word, node.frequency))
            
            for char, child in node.children.items():
                _collect_words(child, current_word + char)
        
        _collect_words(current, prefix)
        
        # Sort by frequency (descending) and limit results
        words.sort(key=lambda x: x[1], reverse=True)
        return words[:limit]
    
    def auto_complete(self, prefix: str, limit: int = 5) -> List[str]:
        """
        Auto-complete functionality
        
        Returns suggestions based on frequency and recency
        """
        print(f"    Auto-complete for '{prefix}':")
        
        suggestions = self.get_words_with_prefix(prefix, limit)
        
        if not suggestions:
            print(f"      No suggestions found")
            return []
        
        result = [word for word, freq in suggestions]
        
        print(f"      Suggestions: {result}")
        for word, freq in suggestions:
            print(f"        '{word}' (frequency: {freq})")
        
        return result


# ==========================================
# 3. MEMORY-OPTIMIZED IMPLEMENTATIONS
# ==========================================

class CompactTrie:
    """
    Memory-optimized trie using bit manipulation and compact storage
    
    Suitable for applications where memory is critical
    Uses bitsets for small alphabets
    """
    
    class CompactTrieNode:
        def __init__(self):
            self.children_bitmap = 0  # Bitmap indicating which children exist
            self.children = []  # Compact array of only existing children
            self.is_end_of_word = False
        
        def _char_to_bit(self, char: str) -> int:
            """Convert character to bit position"""
            return ord(char.lower()) - ord('a')
        
        def has_child(self, char: str) -> bool:
            """Check if child exists using bitmap"""
            bit_pos = self._char_to_bit(char)
            return (self.children_bitmap & (1 << bit_pos)) != 0
        
        def get_child_index(self, char: str) -> int:
            """Get index in children array for character"""
            bit_pos = self._char_to_bit(char)
            if not self.has_child(char):
                return -1
            
            # Count number of set bits before this position
            mask = (1 << bit_pos) - 1
            return bin(self.children_bitmap & mask).count('1')
        
        def add_child(self, char: str, node: 'CompactTrie.CompactTrieNode') -> None:
            """Add child node with bitmap update"""
            bit_pos = self._char_to_bit(char)
            if self.has_child(char):
                # Replace existing child
                index = self.get_child_index(char)
                self.children[index] = node
            else:
                # Add new child
                self.children_bitmap |= (1 << bit_pos)
                index = self.get_child_index(char)
                self.children.insert(index, node)
    
    def __init__(self):
        self.root = self.CompactTrieNode()
        self.word_count = 0
    
    def insert(self, word: str) -> None:
        """Insert word into compact trie"""
        if not word or not word.isalpha():
            return
        
        word = word.lower()
        current = self.root
        
        for char in word:
            if not current.has_child(char):
                current.add_child(char, self.CompactTrieNode())
            
            index = current.get_child_index(char)
            current = current.children[index]
        
        if not current.is_end_of_word:
            current.is_end_of_word = True
            self.word_count += 1
    
    def search(self, word: str) -> bool:
        """Search in compact trie"""
        if not word or not word.isalpha():
            return False
        
        word = word.lower()
        current = self.root
        
        for char in word:
            if not current.has_child(char):
                return False
            index = current.get_child_index(char)
            current = current.children[index]
        
        return current.is_end_of_word
    
    def get_memory_stats(self) -> Dict[str, int]:
        """Get memory usage statistics"""
        def _count_nodes(node: CompactTrie.CompactTrieNode) -> int:
            count = 1
            for child in node.children:
                count += _count_nodes(child)
            return count
        
        total_nodes = _count_nodes(self.root)
        
        return {
            'total_nodes': total_nodes,
            'words': self.word_count,
            'estimated_memory_bytes': total_nodes * 32,  # Rough estimate
            'memory_per_word': (total_nodes * 32) // max(1, self.word_count)
        }


# ==========================================
# 4. SPECIALIZED TRIE IMPLEMENTATIONS
# ==========================================

class VersionedTrie:
    """
    Versioned Trie supporting time-travel queries
    
    Maintains history of all insertions and deletions
    Useful for undo/redo functionality and historical queries
    """
    
    class VersionedTrieNode:
        def __init__(self):
            self.children = {}
            self.word_versions = []  # List of (version, is_active) tuples
        
        def is_word_at_version(self, version: int) -> bool:
            """Check if this is end of word at specific version"""
            active = False
            for v, is_active in self.word_versions:
                if v <= version:
                    active = is_active
                else:
                    break
            return active
    
    def __init__(self):
        self.root = self.VersionedTrieNode()
        self.current_version = 0
        self.version_history = []  # List of (version, operation, word)
    
    def insert(self, word: str) -> None:
        """Insert word with version tracking"""
        if not word:
            return
        
        self.current_version += 1
        current = self.root
        
        # Navigate/create path
        for char in word:
            if char not in current.children:
                current.children[char] = self.VersionedTrieNode()
            current = current.children[char]
        
        # Add version entry
        current.word_versions.append((self.current_version, True))
        self.version_history.append((self.current_version, "INSERT", word))
        
        print(f"    Inserted '{word}' at version {self.current_version}")
    
    def delete(self, word: str) -> None:
        """Delete word with version tracking"""
        if not word:
            return
        
        current = self.root
        
        # Navigate to word
        for char in word:
            if char not in current.children:
                print(f"    Word '{word}' not found for deletion")
                return
            current = current.children[char]
        
        # Check if word exists at current version
        if not current.is_word_at_version(self.current_version):
            print(f"    Word '{word}' not active at current version")
            return
        
        self.current_version += 1
        current.word_versions.append((self.current_version, False))
        self.version_history.append((self.current_version, "DELETE", word))
        
        print(f"    Deleted '{word}' at version {self.current_version}")
    
    def search_at_version(self, word: str, version: int) -> bool:
        """Search for word at specific version"""
        if not word:
            return False
        
        current = self.root
        
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]
        
        return current.is_word_at_version(version)
    
    def get_words_at_version(self, version: int) -> List[str]:
        """Get all active words at specific version"""
        words = []
        
        def _collect_words(node: VersionedTrie.VersionedTrieNode, prefix: str):
            if node.is_word_at_version(version):
                words.append(prefix)
            
            for char, child in node.children.items():
                _collect_words(child, prefix + char)
        
        _collect_words(self.root, "")
        return sorted(words)
    
    def show_version_history(self) -> None:
        """Display version history"""
        print(f"    Version History (current version: {self.current_version}):")
        for version, operation, word in self.version_history:
            print(f"      v{version}: {operation} '{word}'")


# ==========================================
# 5. PERFORMANCE COMPARISON
# ==========================================

def compare_trie_implementations():
    """
    Compare different trie implementations
    """
    print("=== TRIE IMPLEMENTATIONS COMPARISON ===")
    print()
    
    # Test data
    test_words = ["apple", "app", "application", "apply", "banana", "band", "bandana", "can", "cannot", "cat"]
    
    # Test implementations
    implementations = {
        "Array-based": ArrayBasedTrie(),
        "HashMap-based": HashMapBasedTrie(),
        "Compact": CompactTrie()
    }
    
    print("Test words:", test_words)
    print()
    
    # Insert words into all implementations
    for name, trie in implementations.items():
        print(f"{name} Trie - Insertion:")
        for word in test_words:
            trie.insert(word)
        print()
    
    # Test search operations
    search_words = ["app", "apple", "ban", "banana", "missing"]
    
    print("Search Results Comparison:")
    print(f"{'Word':<10} {'Array':<8} {'HashMap':<8} {'Compact':<8}")
    print("-" * 40)
    
    for word in search_words:
        results = []
        for impl_name, trie in implementations.items():
            result = trie.search(word)
            results.append("✓" if result else "✗")
        
        print(f"{word:<10} {results[0]:<8} {results[1]:<8} {results[2]:<8}")
    
    print()
    
    # Memory analysis (for compact trie)
    if hasattr(implementations["Compact"], 'get_memory_stats'):
        print("Memory Statistics for Compact Trie:")
        stats = implementations["Compact"].get_memory_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    print()
    
    print("Implementation Characteristics:")
    print()
    print("Array-based Trie:")
    print("  ✓ Best cache performance")
    print("  ✓ Predictable memory layout")
    print("  ✗ Wastes space for sparse alphabets")
    print("  ✗ Limited to fixed alphabet size")
    print()
    
    print("HashMap-based Trie:")
    print("  ✓ Supports any character set")
    print("  ✓ Memory efficient for sparse data")
    print("  ✓ Flexible and extensible")
    print("  ✗ Higher memory overhead per node")
    print("  ✗ Potential hash collision overhead")
    print()
    
    print("Compact Trie:")
    print("  ✓ Minimal memory usage")
    print("  ✓ Efficient bitmap operations")
    print("  ✗ More complex implementation")
    print("  ✗ Fixed alphabet only")


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_trie_implementations():
    """Demonstrate all trie implementations"""
    print("=== TRIE IMPLEMENTATIONS DEMONSTRATION ===\n")
    
    implementations = TrieImplementations()
    
    # 1. Array-based trie
    implementations.demonstrate_array_based_trie()
    print("\n" + "="*60 + "\n")
    
    # 2. HashMap-based trie with advanced features
    print("=== HASHMAP-BASED TRIE WITH FEATURES ===")
    
    hashmap_trie = HashMapBasedTrie(case_sensitive=False)
    
    words = ["apple", "app", "application", "Apple", "APP", "banana", "band"]
    print("Inserting words with frequency tracking:")
    for word in words:
        hashmap_trie.insert(word)
    
    # Test auto-complete
    print("\nAuto-complete examples:")
    prefixes = ["app", "ap", "ban"]
    for prefix in prefixes:
        hashmap_trie.auto_complete(prefix)
    
    print("\n" + "="*60 + "\n")
    
    # 3. Versioned trie
    print("=== VERSIONED TRIE ===")
    
    versioned_trie = VersionedTrie()
    
    print("Building versioned trie with timeline:")
    timeline_operations = [
        ("insert", "cat"),
        ("insert", "car"),
        ("insert", "card"),
        ("delete", "car"),
        ("insert", "care"),
        ("delete", "cat")
    ]
    
    for operation, word in timeline_operations:
        if operation == "insert":
            versioned_trie.insert(word)
        elif operation == "delete":
            versioned_trie.delete(word)
    
    versioned_trie.show_version_history()
    
    print("\nWords at different versions:")
    for version in [1, 3, 5, 6]:
        words_at_version = versioned_trie.get_words_at_version(version)
        print(f"    Version {version}: {words_at_version}")
    
    print("\n" + "="*60 + "\n")
    
    # 4. Performance comparison
    compare_trie_implementations()


if __name__ == "__main__":
    demonstrate_trie_implementations()
    
    print("\n=== TRIE IMPLEMENTATIONS MASTERY GUIDE ===")
    
    print("\n🎯 IMPLEMENTATION SELECTION GUIDE:")
    print("• Array-based: Fixed small alphabets, cache performance critical")
    print("• HashMap-based: Variable alphabets, Unicode support needed")
    print("• Compact: Memory-constrained environments")
    print("• Versioned: Undo/redo functionality required")
    
    print("\n📊 PERFORMANCE CHARACTERISTICS:")
    print("• Array-based: Best cache locality, O(1) child access")
    print("• HashMap-based: Flexible but higher overhead")
    print("• Compact: Minimal memory, bitmap operations")
    print("• Versioned: Historical queries, higher space complexity")
    
    print("\n⚡ OPTIMIZATION STRATEGIES:")
    print("• Choose implementation based on alphabet size")
    print("• Use compact tries for memory-critical applications")
    print("• Add frequency tracking for auto-complete")
    print("• Implement lazy deletion for better performance")
    print("• Consider path compression for unique suffixes")
    
    print("\n🔧 ADVANCED FEATURES:")
    print("• Case sensitivity handling")
    print("• Unicode and special character support")
    print("• Frequency and recency tracking")
    print("• Version control and time-travel queries")
    print("• Memory optimization techniques")
    
    print("\n🎓 IMPLEMENTATION MASTERY:")
    print("• Understand memory layout implications")
    print("• Master bit manipulation for compact implementations")
    print("• Learn when to use arrays vs hash maps")
    print("• Practice implementing from scratch")
    print("• Study real-world optimization techniques")
