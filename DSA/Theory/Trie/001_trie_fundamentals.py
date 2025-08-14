"""
Trie Fundamentals - Core Concepts and Basic Implementation
=========================================================

Topics: Trie definition, properties, basic operations, complexity analysis
Companies: All tech companies test trie understanding in string problems
Difficulty: Medium to Hard
Time Complexity: O(m) for operations where m is string length
Space Complexity: O(ALPHABET_SIZE * N * M) in worst case
"""

from typing import List, Optional, Dict, Any, Set
from collections import defaultdict, deque

class TrieFundamentals:
    
    def __init__(self):
        """Initialize with demonstration tracking"""
        self.operation_count = 0
        self.demo_steps = []
    
    # ==========================================
    # 1. WHAT IS A TRIE?
    # ==========================================
    
    def explain_trie_concept(self) -> None:
        """
        Explain the fundamental concept of trie data structure
        
        Trie (pronounced "try") is a tree-like data structure for storing strings
        """
        print("=== WHAT IS A TRIE? ===")
        print("A Trie (Prefix Tree) is a tree-like data structure with these characteristics:")
        print()
        print("KEY PROPERTIES:")
        print("• Each node represents a character in a string")
        print("• Root node represents empty string")
        print("• Path from root to any node represents a prefix")
        print("• Complete path from root to leaf represents a word")
        print("• Common prefixes share the same path")
        print()
        print("STRUCTURAL PROPERTIES:")
        print("• Tree-based hierarchical structure")
        print("• Each node has at most ALPHABET_SIZE children")
        print("• Each edge represents a character")
        print("• Nodes may be marked as 'end of word'")
        print("• Height of tree = length of longest word")
        print()
        print("ALPHABET CONSIDERATIONS:")
        print("• Lowercase letters: 26 children per node")
        print("• All ASCII: 128 children per node")
        print("• Unicode: Variable, often use HashMap")
        print("• Custom alphabet: Define your own character set")
        print()
        print("SPACE vs TIME TRADEOFF:")
        print("• Space: Can be large due to sparse nodes")
        print("• Time: Very fast prefix operations")
        print("• Best for: Applications with many prefix queries")
        print()
        print("Real-world Analogies:")
        print("• Dictionary: Words sharing prefixes grouped together")
        print("• File system: Directories and subdirectories")
        print("• Phone book: Names sorted alphabetically")
        print("• Auto-complete: Suggestions based on typed prefix")
    
    def trie_vs_other_structures(self) -> None:
        """Compare trie with other data structures for string operations"""
        print("=== TRIE VS OTHER DATA STRUCTURES ===")
        print()
        print("Trie vs Hash Table:")
        print("  Trie: O(m) search, excellent prefix operations")
        print("  Hash: O(1) average search, no prefix support")
        print("  Use Trie when: Prefix operations are important")
        print()
        print("Trie vs Binary Search Tree:")
        print("  Trie: O(m) operations, space can be large")
        print("  BST: O(log n) operations, more space efficient")
        print("  Use Trie when: String-specific operations needed")
        print()
        print("Trie vs Sorted Array:")
        print("  Trie: O(m) search, O(m) insertion")
        print("  Array: O(log n) search, O(n) insertion")
        print("  Use Trie when: Many insertions with prefix queries")
        print()
        print("When to Use Trie:")
        print("• Auto-complete and suggestion systems")
        print("• Spell checkers and dictionaries")
        print("• IP routing tables")
        print("• String matching with wildcards")
        print("• Prefix-based search operations")
        print("• Word games (Scrabble, Boggle)")
    
    # ==========================================
    # 2. BASIC TRIE IMPLEMENTATION
    # ==========================================
    
    def demonstrate_basic_trie(self) -> None:
        """
        Demonstrate basic trie operations with detailed explanation
        """
        print("=== BASIC TRIE IMPLEMENTATION DEMONSTRATION ===")
        
        trie = BasicTrie()
        
        # Words to insert
        words = ["cat", "car", "card", "care", "careful", "cats", "dog", "dodge"]
        
        print(f"Inserting words: {words}")
        print()
        
        # Insert words with detailed tracing
        for word in words:
            print(f"Inserting '{word}':")
            trie.insert(word)
            print()
        
        print("="*50)
        
        # Search operations
        search_words = ["cat", "car", "card", "care", "careful", "cats", "dog", "dodge", "ca", "do", "missing"]
        
        print("\nSearch operations:")
        for word in search_words:
            result = trie.search(word)
            print(f"Search '{word}': {'Found' if result else 'Not found'}")
        
        print("\n" + "="*50)
        
        # Prefix operations
        prefixes = ["ca", "car", "d", "do", "missing"]
        
        print("\nPrefix search operations:")
        for prefix in prefixes:
            result = trie.starts_with(prefix)
            print(f"Starts with '{prefix}': {'Yes' if result else 'No'}")
        
        print("\n" + "="*50)
        
        # Display trie structure
        print("\nTrie structure visualization:")
        trie.display_structure()


class TrieNode:
    """
    Basic Trie Node implementation
    
    Each node contains:
    - children: Dictionary mapping characters to child nodes
    - is_end_of_word: Boolean flag indicating complete word
    - Optional: additional data like frequency, word, etc.
    """
    
    def __init__(self):
        self.children = {}  # Character -> TrieNode mapping
        self.is_end_of_word = False
        self.word_count = 0  # Optional: count of words ending here
    
    def __repr__(self):
        return f"TrieNode(children={list(self.children.keys())}, is_end={self.is_end_of_word})"


class BasicTrie:
    """
    Basic Trie implementation with fundamental operations
    
    Supports:
    - Insert: Add a word to the trie
    - Search: Check if word exists
    - StartsWith: Check if prefix exists
    - Delete: Remove a word (advanced)
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0
        self.operation_count = 0
    
    def insert(self, word: str) -> None:
        """
        Insert a word into the trie
        
        Time: O(m) where m is length of word
        Space: O(m) in worst case (all unique characters)
        """
        self.operation_count += 1
        
        print(f"  Inserting '{word}' - Step by step:")
        
        current = self.root
        path = []
        
        for i, char in enumerate(word):
            path.append(char)
            
            if char not in current.children:
                current.children[char] = TrieNode()
                print(f"    Step {i+1}: Created new node for '{char}' at path '{(''.join(path))}'")
            else:
                print(f"    Step {i+1}: Using existing node for '{char}' at path '{(''.join(path))}'")
            
            current = current.children[char]
        
        if not current.is_end_of_word:
            current.is_end_of_word = True
            current.word_count += 1
            self.word_count += 1
            print(f"    Final: Marked end of word at '{word}'")
        else:
            print(f"    Final: Word '{word}' already exists, incrementing count")
            current.word_count += 1
    
    def search(self, word: str) -> bool:
        """
        Search for a complete word in the trie
        
        Time: O(m) where m is length of word
        Space: O(1)
        """
        print(f"  Searching for '{word}':")
        
        current = self.root
        path = []
        
        for i, char in enumerate(word):
            path.append(char)
            
            if char not in current.children:
                print(f"    Step {i+1}: Character '{char}' not found at path '{(''.join(path))}'")
                print(f"    Result: '{word}' NOT FOUND")
                return False
            
            print(f"    Step {i+1}: Found '{char}' at path '{(''.join(path))}'")
            current = current.children[char]
        
        result = current.is_end_of_word
        print(f"    Final: Reached end, is_end_of_word = {result}")
        print(f"    Result: '{word}' {'FOUND' if result else 'NOT FOUND (prefix only)'}")
        
        return result
    
    def starts_with(self, prefix: str) -> bool:
        """
        Check if any word in trie starts with given prefix
        
        Time: O(m) where m is length of prefix
        Space: O(1)
        """
        print(f"  Checking prefix '{prefix}':")
        
        current = self.root
        path = []
        
        for i, char in enumerate(prefix):
            path.append(char)
            
            if char not in current.children:
                print(f"    Step {i+1}: Character '{char}' not found at path '{(''.join(path))}'")
                print(f"    Result: Prefix '{prefix}' NOT FOUND")
                return False
            
            print(f"    Step {i+1}: Found '{char}' at path '{(''.join(path))}'")
            current = current.children[char]
        
        print(f"    Result: Prefix '{prefix}' EXISTS")
        return True
    
    def delete(self, word: str) -> bool:
        """
        Delete a word from the trie
        
        Time: O(m) where m is length of word
        Space: O(m) for recursion stack
        
        Cases to handle:
        1. Word doesn't exist
        2. Word is prefix of another word
        3. Word has other words as prefix
        4. Word is standalone
        """
        print(f"  Deleting '{word}':")
        
        def _delete_helper(node: TrieNode, word: str, index: int) -> bool:
            """
            Recursive helper for deletion
            Returns True if current node should be deleted
            """
            if index == len(word):
                # Reached end of word
                if not node.is_end_of_word:
                    print(f"    Word '{word}' doesn't exist")
                    return False
                
                node.is_end_of_word = False
                node.word_count = 0
                print(f"    Unmarked end of word for '{word}'")
                
                # Delete node if it has no children
                return len(node.children) == 0
            
            char = word[index]
            child_node = node.children.get(char)
            
            if child_node is None:
                print(f"    Character '{char}' not found at index {index}")
                return False
            
            should_delete_child = _delete_helper(child_node, word, index + 1)
            
            if should_delete_child:
                del node.children[char]
                print(f"    Deleted node for character '{char}'")
                
                # Delete current node if:
                # 1. It's not end of another word
                # 2. It has no other children
                return not node.is_end_of_word and len(node.children) == 0
            
            return False
        
        if _delete_helper(self.root, word, 0):
            self.word_count -= 1
            print(f"    Successfully deleted '{word}'")
            return True
        else:
            print(f"    Failed to delete '{word}' or word doesn't exist")
            return False
    
    def display_structure(self) -> None:
        """
        Display the trie structure in a readable format
        """
        print("Trie Structure:")
        
        def _display_helper(node: TrieNode, prefix: str, depth: int):
            """Recursive helper to display trie structure"""
            indent = "  " * depth
            
            if node.is_end_of_word:
                print(f"{indent}'{prefix}' [END] (count: {node.word_count})")
            
            for char, child in sorted(node.children.items()):
                print(f"{indent}├── '{char}'")
                _display_helper(child, prefix + char, depth + 1)
        
        if not self.root.children:
            print("  (empty)")
        else:
            print("  Root")
            _display_helper(self.root, "", 1)
        
        print(f"\nStatistics:")
        print(f"  Total words: {self.word_count}")
        print(f"  Total operations: {self.operation_count}")
    
    def get_all_words(self) -> List[str]:
        """
        Get all words stored in the trie
        
        Time: O(N) where N is total number of characters in all words
        Space: O(N) for storing all words
        """
        words = []
        
        def _collect_words(node: TrieNode, current_word: str):
            """Recursive helper to collect all words"""
            if node.is_end_of_word:
                words.append(current_word)
            
            for char, child in node.children.items():
                _collect_words(child, current_word + char)
        
        _collect_words(self.root, "")
        return sorted(words)
    
    def count_words_with_prefix(self, prefix: str) -> int:
        """
        Count how many words start with given prefix
        
        Time: O(m + k) where m is prefix length, k is number of words with prefix
        Space: O(h) where h is height of trie (for recursion)
        """
        print(f"  Counting words with prefix '{prefix}':")
        
        # Navigate to prefix node
        current = self.root
        for char in prefix:
            if char not in current.children:
                print(f"    Prefix '{prefix}' not found")
                return 0
            current = current.children[char]
        
        # Count words in subtree
        def _count_words(node: TrieNode) -> int:
            count = node.word_count if node.is_end_of_word else 0
            for child in node.children.values():
                count += _count_words(child)
            return count
        
        count = _count_words(current)
        print(f"    Found {count} words with prefix '{prefix}'")
        return count


# ==========================================
# 3. TRIE VARIANTS AND ENHANCEMENTS
# ==========================================

class CompressedTrie:
    """
    Compressed Trie (Radix Tree) - Space-optimized version
    
    Compresses chains of single-child nodes into single edges
    More space-efficient but more complex implementation
    """
    
    class CompressedTrieNode:
        def __init__(self, substring: str = ""):
            self.substring = substring  # Compressed edge label
            self.children = {}
            self.is_end_of_word = False
        
        def __repr__(self):
            return f"CompressedNode('{self.substring}', end={self.is_end_of_word})"
    
    def __init__(self):
        self.root = self.CompressedTrieNode()
        self.word_count = 0
    
    def insert(self, word: str) -> None:
        """
        Insert word into compressed trie
        
        More complex than basic trie due to substring handling
        """
        print(f"Inserting '{word}' into compressed trie:")
        
        current = self.root
        i = 0
        
        while i < len(word):
            char = word[i]
            
            if char not in current.children:
                # Create new node with remaining substring
                remaining = word[i:]
                current.children[char] = self.CompressedTrieNode(remaining)
                current.children[char].is_end_of_word = True
                print(f"  Created new compressed node: '{remaining}'")
                self.word_count += 1
                return
            
            child = current.children[char]
            common_length = 0
            
            # Find common prefix length
            remaining_word = word[i:]
            for j in range(min(len(remaining_word), len(child.substring))):
                if remaining_word[j] == child.substring[j]:
                    common_length += 1
                else:
                    break
            
            if common_length == len(child.substring):
                # Full substring matches, continue to next level
                print(f"  Full match with '{child.substring}', continuing")
                current = child
                i += common_length
            elif common_length == len(remaining_word):
                # Word is prefix of existing substring, need to split
                print(f"  Word is prefix, splitting node")
                # Implementation details for splitting...
                # This is complex and requires careful node restructuring
                self.word_count += 1
                return
            else:
                # Partial match, need to split the node
                print(f"  Partial match, splitting at position {common_length}")
                # Implementation details for splitting...
                # This is complex and requires careful node restructuring
                self.word_count += 1
                return
    
    def search(self, word: str) -> bool:
        """Search in compressed trie - simplified version"""
        current = self.root
        i = 0
        
        while i < len(word):
            char = word[i]
            if char not in current.children:
                return False
            
            child = current.children[char]
            remaining = word[i:]
            
            if not remaining.startswith(child.substring):
                return False
            
            if len(remaining) == len(child.substring):
                return child.is_end_of_word
            
            current = child
            i += len(child.substring)
        
        return current.is_end_of_word


class SuffixTrie:
    """
    Suffix Trie - Stores all suffixes of a given string
    
    Useful for pattern matching and string analysis
    More space-intensive but enables powerful string operations
    """
    
    def __init__(self, text: str):
        self.text = text
        self.trie = BasicTrie()
        self._build_suffix_trie()
    
    def _build_suffix_trie(self):
        """Build suffix trie from all suffixes of text"""
        print(f"Building suffix trie for text: '{self.text}'")
        
        for i in range(len(self.text)):
            suffix = self.text[i:] + "$"  # Add terminator
            print(f"  Adding suffix {i}: '{suffix}'")
            self.trie.insert(suffix)
    
    def contains_pattern(self, pattern: str) -> bool:
        """Check if pattern exists as substring in original text"""
        return self.trie.starts_with(pattern)
    
    def find_all_occurrences(self, pattern: str) -> List[int]:
        """Find all starting positions of pattern in text"""
        positions = []
        
        for i in range(len(self.text) - len(pattern) + 1):
            if self.text[i:].startswith(pattern):
                positions.append(i)
        
        return positions


# ==========================================
# 4. PERFORMANCE ANALYSIS
# ==========================================

def analyze_trie_performance():
    """
    Analyze performance characteristics of trie operations
    """
    print("=== TRIE PERFORMANCE ANALYSIS ===")
    print()
    
    print("TIME COMPLEXITY:")
    print("• Insert:           O(m) - where m is length of word")
    print("• Search:           O(m) - where m is length of word")
    print("• Delete:           O(m) - where m is length of word")
    print("• Prefix Search:    O(p) - where p is length of prefix")
    print("• All words:        O(N) - where N is total characters")
    print()
    
    print("SPACE COMPLEXITY:")
    print("• Worst case:       O(ALPHABET_SIZE × N × M)")
    print("  - ALPHABET_SIZE: Number of possible characters")
    print("  - N: Number of words")
    print("  - M: Average length of words")
    print("• Best case:        O(N × M) - when words share many prefixes")
    print("• Average case:     Between best and worst, depends on data")
    print()
    
    print("COMPARISON WITH OTHER STRUCTURES:")
    print("┌─────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐")
    print("│ Operation       │ Trie        │ Hash Table  │ BST         │ Sorted Array│")
    print("├─────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤")
    print("│ Insert          │ O(m)        │ O(m)        │ O(m log n)  │ O(n)        │")
    print("│ Search          │ O(m)        │ O(m)        │ O(m log n)  │ O(log n)    │")
    print("│ Delete          │ O(m)        │ O(m)        │ O(m log n)  │ O(n)        │")
    print("│ Prefix Search   │ O(p)        │ O(n)        │ O(p log n)  │ O(log n)    │")
    print("│ All with prefix │ O(p + k)    │ O(n)        │ O(p + k)    │ O(log n + k)│")
    print("│ Space           │ O(A×N×M)    │ O(N×M)      │ O(N×M)      │ O(N×M)      │")
    print("└─────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘")
    print()
    print("Where: m=word length, n=number of words, p=prefix length, k=results, A=alphabet size")
    print()
    
    print("ADVANTAGES OF TRIE:")
    print("• Very fast prefix operations")
    print("• No hash collisions")
    print("• Alphabetically ordered traversal")
    print("• Memory-efficient for many shared prefixes")
    print("• Support for partial matches and wildcards")
    print()
    
    print("DISADVANTAGES OF TRIE:")
    print("• High memory usage for sparse data")
    print("• More complex implementation")
    print("• Not suitable for non-string data")
    print("• Poor cache performance due to pointer chasing")
    print("• Memory fragmentation with many small nodes")
    print()
    
    print("WHEN TO USE TRIE:")
    print("• Auto-complete and spell-check systems")
    print("• Dictionary and word games")
    print("• IP routing tables")
    print("• Pattern matching applications")
    print("• When prefix operations are frequent")
    print("• Data with many shared prefixes")


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_trie_fundamentals():
    """Demonstrate all trie fundamental concepts"""
    print("=== TRIE FUNDAMENTALS DEMONSTRATION ===\n")
    
    fundamentals = TrieFundamentals()
    
    # 1. Concept explanation
    fundamentals.explain_trie_concept()
    print("\n" + "="*60 + "\n")
    
    # 2. Comparison with other structures
    fundamentals.trie_vs_other_structures()
    print("\n" + "="*60 + "\n")
    
    # 3. Basic trie demonstration
    fundamentals.demonstrate_basic_trie()
    print("\n" + "="*60 + "\n")
    
    # 4. Additional operations
    print("=== ADDITIONAL TRIE OPERATIONS ===")
    trie = BasicTrie()
    words = ["apple", "app", "application", "apply", "banana", "band", "bandana"]
    
    print("Building trie with words:", words)
    for word in words:
        trie.insert(word)
    
    print("\nAll words in trie:")
    all_words = trie.get_all_words()
    print(f"  {all_words}")
    
    print("\nPrefix counting:")
    prefixes = ["app", "ban", "b", "xyz"]
    for prefix in prefixes:
        count = trie.count_words_with_prefix(prefix)
    
    print("\nDeletion examples:")
    delete_words = ["app", "banana", "missing"]
    for word in delete_words:
        trie.delete(word)
    
    print("\n" + "="*60 + "\n")
    
    # 5. Compressed trie example
    print("=== COMPRESSED TRIE EXAMPLE ===")
    compressed = CompressedTrie()
    compressed_words = ["romane", "romanus", "romulus"]
    
    print("Inserting into compressed trie:")
    for word in compressed_words:
        compressed.insert(word)
    
    print("\n" + "="*60 + "\n")
    
    # 6. Suffix trie example
    print("=== SUFFIX TRIE EXAMPLE ===")
    text = "banana"
    suffix_trie = SuffixTrie(text)
    
    patterns = ["ana", "ban", "xyz"]
    print(f"\nPattern search in '{text}':")
    for pattern in patterns:
        found = suffix_trie.contains_pattern(pattern)
        print(f"  '{pattern}': {'Found' if found else 'Not found'}")
    
    print("\n" + "="*60 + "\n")
    
    # 7. Performance analysis
    analyze_trie_performance()


if __name__ == "__main__":
    demonstrate_trie_fundamentals()
    
    print("\n" + "="*60)
    print("=== TRIE MASTERY GUIDE ===")
    print("="*60)
    
    print("\n🎯 WHEN TO USE TRIE:")
    print("✅ Auto-complete and suggestion systems")
    print("✅ Spell checkers and dictionaries")
    print("✅ Pattern matching and string search")
    print("✅ Prefix-based operations")
    print("✅ Word games and puzzles")
    print("✅ IP routing and network applications")
    
    print("\n📋 TRIE IMPLEMENTATION CHECKLIST:")
    print("1. Choose appropriate node structure (array vs map)")
    print("2. Handle case sensitivity requirements")
    print("3. Implement proper deletion with cleanup")
    print("4. Consider memory optimization techniques")
    print("5. Add word frequency tracking if needed")
    print("6. Implement iterators for word traversal")
    
    print("\n⚡ OPTIMIZATION STRATEGIES:")
    print("• Use arrays for small, fixed alphabets (e.g., lowercase letters)")
    print("• Use hash maps for large or variable alphabets")
    print("• Implement compressed trie for space efficiency")
    print("• Add path compression for long unique paths")
    print("• Consider suffix compression for better space usage")
    
    print("\n🚨 COMMON PITFALLS:")
    print("• Forgetting to mark end-of-word nodes")
    print("• Not handling case sensitivity properly")
    print("• Inefficient deletion leaving orphaned nodes")
    print("• Memory leaks in dynamic trie implementations")
    print("• Not considering unicode and special characters")
    
    print("\n🎓 LEARNING PROGRESSION:")
    print("1. Master basic trie operations (insert, search, delete)")
    print("2. Understand space-time tradeoffs")
    print("3. Learn trie variants (compressed, suffix tries)")
    print("4. Practice with real-world applications")
    print("5. Study advanced optimizations and techniques")
    
    print("\n📚 PROBLEM CATEGORIES TO PRACTICE:")
    print("• String search and pattern matching")
    print("• Auto-complete and suggestion systems")
    print("• Word games and dictionary operations")
    print("• Longest common prefix problems")
    print("• String validation and spell checking")
    print("• IP address and URL routing")
    
    print("\n💡 SUCCESS TIPS:")
    print("• Always consider the alphabet size and data characteristics")
    print("• Think about memory vs speed tradeoffs")
    print("• Practice implementing from scratch")
    print("• Understand when trie is better than hash table")
    print("• Master prefix-based operations and optimizations")
