"""
Trie Delete Operations - Multiple Approaches
Difficulty: Medium

Implement comprehensive delete operations for Trie data structure.
Cover various deletion strategies and edge cases.
"""

from typing import List, Dict, Optional, Set
from collections import defaultdict

class TrieNode:
    """Enhanced Trie Node with additional metadata"""
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_word: bool = False
        self.word_count: int = 0  # For frequency tracking
        self.words_ending_here: List[str] = []  # Store actual words

class TrieWithDelete1:
    """
    Approach 1: Standard Recursive Delete
    
    Classic recursive deletion with node cleanup.
    
    Time: O(m) where m is word length
    Space: O(m) for recursion stack
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.total_words = 0
    
    def insert(self, word: str) -> None:
        """Insert word into trie"""
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end_of_word:
            self.total_words += 1
        
        node.is_end_of_word = True
        node.words_ending_here.append(word)
    
    def search(self, word: str) -> bool:
        """Search for word in trie"""
        node = self.root
        
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return node.is_end_of_word
    
    def delete(self, word: str) -> bool:
        """Delete word from trie"""
        def _delete_helper(node: TrieNode, word: str, index: int) -> bool:
            # Base case: reached end of word
            if index == len(word):
                if not node.is_end_of_word:
                    return False  # Word doesn't exist
                
                node.is_end_of_word = False
                if word in node.words_ending_here:
                    node.words_ending_here.remove(word)
                self.total_words -= 1
                
                # Return True if node has no children (can be deleted)
                return len(node.children) == 0
            
            char = word[index]
            if char not in node.children:
                return False  # Word doesn't exist
            
            # Recursively delete
            should_delete_child = _delete_helper(node.children[char], word, index + 1)
            
            if should_delete_child:
                del node.children[char]
                
                # Return True if current node can be deleted
                return not node.is_end_of_word and len(node.children) == 0
            
            return False
        
        return _delete_helper(self.root, word, 0)
    
    def get_all_words(self) -> List[str]:
        """Get all words in trie"""
        words = []
        
        def dfs(node: TrieNode, prefix: str):
            if node.is_end_of_word:
                words.extend(node.words_ending_here)
            
            for char, child in node.children.items():
                dfs(child, prefix + char)
        
        dfs(self.root, "")
        return words

class TrieWithDelete2:
    """
    Approach 2: Iterative Delete with Stack
    
    Non-recursive deletion using explicit stack.
    
    Time: O(m)
    Space: O(m) for stack
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.total_words = 0
    
    def insert(self, word: str) -> None:
        """Insert word"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end_of_word:
            self.total_words += 1
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """Search for word"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def delete(self, word: str) -> bool:
        """Iterative delete using stack"""
        # First check if word exists
        if not self.search(word):
            return False
        
        # Build path to word
        path = []
        node = self.root
        
        for char in word:
            path.append((node, char))
            node = node.children[char]
        
        # Mark end node as not end of word
        node.is_end_of_word = False
        self.total_words -= 1
        
        # Clean up nodes from bottom to top
        # Start from the last node (end of word)
        for i in range(len(path) - 1, -1, -1):
            parent, char = path[i]
            current = parent.children[char]
            
            # Can delete if: not end of word AND no children
            if not current.is_end_of_word and len(current.children) == 0:
                del parent.children[char]
            else:
                # Stop cleanup if node is still needed
                break
        
        return True

class TrieWithDelete3:
    """
    Approach 3: Reference Counting Delete
    
    Track references to optimize deletion decisions.
    
    Time: O(m)
    Space: O(alphabet_size * total_chars)
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.node_ref_count: Dict[TrieNode, int] = defaultdict(int)
        self.total_words = 0
    
    def insert(self, word: str) -> None:
        """Insert with reference counting"""
        node = self.root
        self.node_ref_count[node] += 1
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            self.node_ref_count[node] += 1
        
        if not node.is_end_of_word:
            self.total_words += 1
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """Search for word"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def delete(self, word: str) -> bool:
        """Delete with reference counting"""
        if not self.search(word):
            return False
        
        # Decrease reference counts
        node = self.root
        path = [node]
        
        for char in word:
            node = node.children[char]
            path.append(node)
        
        # Mark as not end of word
        node.is_end_of_word = False
        self.total_words -= 1
        
        # Decrease reference counts and clean up
        for i in range(len(path) - 1, 0, -1):
            current = path[i]
            parent = path[i - 1]
            
            self.node_ref_count[current] -= 1
            
            # If no references and not end of word, delete
            if (self.node_ref_count[current] == 0 and 
                not current.is_end_of_word):
                
                # Find the character that leads to this node
                char_to_delete = None
                for char, child in parent.children.items():
                    if child == current:
                        char_to_delete = char
                        break
                
                if char_to_delete:
                    del parent.children[char_to_delete]
                    del self.node_ref_count[current]
        
        return True

class TrieWithDelete4:
    """
    Approach 4: Lazy Deletion
    
    Mark nodes as deleted without immediate cleanup.
    
    Time: O(m) for delete, periodic cleanup
    Space: O(deleted_nodes)
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.deleted_words: Set[str] = set()
        self.total_words = 0
        self.cleanup_threshold = 100  # Cleanup after 100 deletions
    
    def insert(self, word: str) -> None:
        """Insert word"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if word in self.deleted_words:
            self.deleted_words.remove(word)
        elif not node.is_end_of_word:
            self.total_words += 1
        
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """Search excluding deleted words"""
        if word in self.deleted_words:
            return False
        
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return node.is_end_of_word
    
    def delete(self, word: str) -> bool:
        """Lazy delete - just mark as deleted"""
        if not self.search(word):
            return False
        
        self.deleted_words.add(word)
        self.total_words -= 1
        
        # Trigger cleanup if threshold reached
        if len(self.deleted_words) >= self.cleanup_threshold:
            self._cleanup()
        
        return True
    
    def _cleanup(self) -> None:
        """Periodic cleanup of deleted nodes"""
        # Rebuild trie without deleted words
        all_words = self.get_all_words()
        valid_words = [w for w in all_words if w not in self.deleted_words]
        
        # Rebuild
        self.__init__()
        for word in valid_words:
            self.insert(word)
    
    def get_all_words(self) -> List[str]:
        """Get all non-deleted words"""
        words = []
        
        def dfs(node: TrieNode, prefix: str):
            if node.is_end_of_word and prefix not in self.deleted_words:
                words.append(prefix)
            
            for char, child in node.children.items():
                dfs(child, prefix + char)
        
        dfs(self.root, "")
        return words

class TrieWithDelete5:
    """
    Approach 5: Batch Delete Operations
    
    Optimize for multiple deletions at once.
    
    Time: O(total_chars_in_all_words)
    Space: O(words_to_delete)
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.total_words = 0
    
    def insert(self, word: str) -> None:
        """Insert word"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end_of_word:
            self.total_words += 1
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """Search for word"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def batch_delete(self, words: List[str]) -> int:
        """Delete multiple words efficiently"""
        deleted_count = 0
        words_to_delete = set(words)
        
        # Mark all words for deletion
        for word in words_to_delete:
            if self.search(word):
                deleted_count += 1
        
        # Rebuild trie with remaining words
        remaining_words = []
        self._collect_words_except(self.root, "", words_to_delete, remaining_words)
        
        # Rebuild
        self.__init__()
        for word in remaining_words:
            self.insert(word)
        
        return deleted_count
    
    def _collect_words_except(self, node: TrieNode, prefix: str, 
                             exclude: Set[str], result: List[str]) -> None:
        """Collect all words except those in exclude set"""
        if node.is_end_of_word and prefix not in exclude:
            result.append(prefix)
        
        for char, child in node.children.items():
            self._collect_words_except(child, prefix + char, exclude, result)

class TrieWithDelete6:
    """
    Approach 6: Undo-able Delete Operations
    
    Support undo functionality for deletions.
    
    Time: O(m) for delete/undo
    Space: O(deleted_operations)
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.total_words = 0
        self.delete_history: List[str] = []  # Stack of deleted words
        self.max_history = 1000
    
    def insert(self, word: str) -> None:
        """Insert word"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end_of_word:
            self.total_words += 1
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """Search for word"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def delete(self, word: str) -> bool:
        """Delete with undo support"""
        if not self.search(word):
            return False
        
        # Perform deletion
        self._delete_recursive(self.root, word, 0)
        
        # Add to history
        self.delete_history.append(word)
        if len(self.delete_history) > self.max_history:
            self.delete_history.pop(0)  # Remove oldest
        
        self.total_words -= 1
        return True
    
    def _delete_recursive(self, node: TrieNode, word: str, index: int) -> bool:
        """Recursive delete helper"""
        if index == len(word):
            node.is_end_of_word = False
            return len(node.children) == 0
        
        char = word[index]
        if char not in node.children:
            return False
        
        should_delete = self._delete_recursive(node.children[char], word, index + 1)
        
        if should_delete:
            del node.children[char]
            return not node.is_end_of_word and len(node.children) == 0
        
        return False
    
    def undo_delete(self) -> Optional[str]:
        """Undo last delete operation"""
        if not self.delete_history:
            return None
        
        word = self.delete_history.pop()
        self.insert(word)  # Re-insert the word
        return word
    
    def get_delete_history(self) -> List[str]:
        """Get history of deleted words"""
        return self.delete_history.copy()


def test_basic_delete_operations():
    """Test basic delete functionality"""
    print("=== Testing Basic Delete Operations ===")
    
    implementations = [
        ("Recursive", TrieWithDelete1),
        ("Iterative", TrieWithDelete2),
        ("Ref Counting", TrieWithDelete3),
        ("Lazy Delete", TrieWithDelete4),
        ("Undo-able", TrieWithDelete6),
    ]
    
    test_words = ["cat", "cats", "car", "card", "care", "careful"]
    
    for name, TrieClass in implementations:
        print(f"\n--- Testing {name} ---")
        
        trie = TrieClass()
        
        # Insert words
        for word in test_words:
            trie.insert(word)
        
        print(f"Inserted: {test_words}")
        print(f"All words: {trie.get_all_words()}")
        
        # Test deletions
        delete_tests = ["cats", "car", "xyz", "careful"]
        
        for word in delete_tests:
            exists_before = trie.search(word)
            deleted = trie.delete(word)
            exists_after = trie.search(word)
            
            print(f"Delete '{word}': before={exists_before}, deleted={deleted}, after={exists_after}")
        
        print(f"Remaining words: {trie.get_all_words()}")


def test_edge_cases():
    """Test edge cases for delete operations"""
    print("\n=== Testing Edge Cases ===")
    
    trie = TrieWithDelete1()
    
    edge_cases = [
        # Delete from empty trie
        ("Empty trie", [], "word"),
        
        # Delete non-existent word
        ("Non-existent", ["hello"], "world"),
        
        # Delete prefix of existing word
        ("Prefix delete", ["hello"], "hel"),
        
        # Delete word that is prefix of another
        ("Word is prefix", ["hello", "helloworld"], "hello"),
        
        # Delete all words
        ("Delete all", ["a", "b", "c"], ["a", "b", "c"]),
        
        # Delete same word multiple times
        ("Multiple delete", ["test"], ["test", "test"]),
    ]
    
    for description, insert_words, delete_words in edge_cases:
        print(f"\n{description}:")
        
        # Reset trie
        trie = TrieWithDelete1()
        
        # Insert words
        for word in insert_words:
            trie.insert(word)
        
        print(f"  Initial: {trie.get_all_words()}")
        
        # Delete words
        if isinstance(delete_words, str):
            delete_words = [delete_words]
        
        for word in delete_words:
            result = trie.delete(word)
            print(f"  Delete '{word}': {result}")
        
        print(f"  Final: {trie.get_all_words()}")


def test_batch_operations():
    """Test batch delete operations"""
    print("\n=== Testing Batch Operations ===")
    
    trie = TrieWithDelete5()
    
    # Insert many words
    words = ["apple", "app", "application", "apply", "banana", "band", "bandana", "can", "cat", "dog"]
    for word in words:
        trie.insert(word)
    
    print(f"Initial words: {trie.get_all_words()}")
    
    # Batch delete
    to_delete = ["app", "banana", "cat", "xyz"]  # xyz doesn't exist
    deleted_count = trie.batch_delete(to_delete)
    
    print(f"Batch deleted {to_delete}: {deleted_count} words removed")
    print(f"Remaining words: {trie.get_all_words()}")


def test_undo_functionality():
    """Test undo delete functionality"""
    print("\n=== Testing Undo Functionality ===")
    
    trie = TrieWithDelete6()
    
    # Insert words
    words = ["test", "testing", "tester", "team"]
    for word in words:
        trie.insert(word)
    
    print(f"Initial: {trie.get_all_words()}")
    
    # Delete some words
    deletions = ["testing", "team"]
    for word in deletions:
        trie.delete(word)
        print(f"Deleted '{word}', remaining: {trie.get_all_words()}")
    
    # Show delete history
    print(f"Delete history: {trie.get_delete_history()}")
    
    # Undo deletions
    print(f"\nUndoing deletions:")
    while True:
        undone = trie.undo_delete()
        if undone is None:
            break
        print(f"Undid deletion of '{undone}', current: {trie.get_all_words()}")


def benchmark_delete_approaches():
    """Benchmark different delete approaches"""
    print("\n=== Benchmarking Delete Approaches ===")
    
    import time
    import random
    import string
    
    # Generate test data
    words = []
    for _ in range(1000):
        length = random.randint(3, 10)
        word = ''.join(random.choices(string.ascii_lowercase, k=length))
        words.append(word)
    
    words = list(set(words))  # Remove duplicates
    delete_words = random.sample(words, min(100, len(words) // 2))
    
    implementations = [
        ("Recursive", TrieWithDelete1),
        ("Iterative", TrieWithDelete2),
        ("Lazy Delete", TrieWithDelete4),
    ]
    
    print(f"Testing with {len(words)} words, deleting {len(delete_words)}")
    
    for name, TrieClass in implementations:
        start_time = time.time()
        
        trie = TrieClass()
        
        # Insert all words
        for word in words:
            trie.insert(word)
        
        # Delete selected words
        for word in delete_words:
            trie.delete(word)
        
        end_time = time.time()
        
        remaining = len(trie.get_all_words())
        print(f"{name:15}: {(end_time - start_time)*1000:.2f}ms, {remaining} words remaining")


def demonstrate_real_world_usage():
    """Demonstrate real-world delete scenarios"""
    print("\n=== Real-World Delete Scenarios ===")
    
    # Scenario 1: Dictionary maintenance
    print("1. Dictionary Word Management:")
    dictionary = TrieWithDelete6()  # Use undo-able version
    
    # Add words
    words = ["hello", "world", "help", "held", "hell"]
    for word in words:
        dictionary.insert(word)
    
    print(f"   Dictionary: {dictionary.get_all_words()}")
    
    # Remove outdated words
    dictionary.delete("hell")  # Remove inappropriate word
    print(f"   After removing 'hell': {dictionary.get_all_words()}")
    
    # Undo if needed
    dictionary.undo_delete()
    print(f"   After undo: {dictionary.get_all_words()}")
    
    # Scenario 2: Cache management
    print("\n2. Cache Management:")
    cache = TrieWithDelete4()  # Use lazy delete for cache
    
    # Add cache entries
    cache_entries = ["user:123", "user:456", "session:abc", "session:def"]
    for entry in cache_entries:
        cache.insert(entry)
    
    print(f"   Cache entries: {cache.get_all_words()}")
    
    # Expire old sessions
    cache.delete("session:abc")
    print(f"   After expiring session:abc: {cache.get_all_words()}")
    
    # Scenario 3: Configuration management
    print("\n3. Configuration Key Management:")
    config = TrieWithDelete5()  # Use batch delete
    
    # Add config keys
    config_keys = ["app.debug", "app.version", "db.host", "db.port", "cache.enabled"]
    for key in config_keys:
        config.insert(key)
    
    print(f"   Config keys: {config.get_all_words()}")
    
    # Remove deprecated config section
    deprecated_keys = ["app.debug", "cache.enabled"]
    removed = config.batch_delete(deprecated_keys)
    print(f"   Removed {removed} deprecated keys: {config.get_all_words()}")


if __name__ == "__main__":
    test_basic_delete_operations()
    test_edge_cases()
    test_batch_operations()
    test_undo_functionality()
    benchmark_delete_approaches()
    demonstrate_real_world_usage()

"""
Trie Delete Operations demonstrates comprehensive deletion strategies:

1. Recursive Delete - Classic approach with proper node cleanup
2. Iterative Delete - Stack-based non-recursive implementation
3. Reference Counting - Optimized deletion using reference tracking
4. Lazy Deletion - Deferred cleanup for better performance
5. Batch Delete - Efficient multiple word deletion
6. Undo-able Delete - Deletion with undo functionality

Each approach addresses different requirements from simple deletion
to advanced features like batch operations and undo capabilities.
"""
