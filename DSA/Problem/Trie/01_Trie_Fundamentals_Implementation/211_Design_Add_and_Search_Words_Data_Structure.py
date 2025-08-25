"""
211. Design Add and Search Words Data Structure - Multiple Approaches
Difficulty: Medium

Design a data structure that supports adding new words and finding if a string 
matches any previously added string. The search function can search for a literal 
word or a regular expression string containing only letters a-z or dots '.'.
A '.' means it can represent any one letter.

LeetCode Problem: https://leetcode.com/problems/design-add-and-search-words-data-structure/

Example:
addWord("bad")
addWord("dad") 
addWord("mad")
search("pad") -> false
search("bad") -> true
search(".ad") -> true
search("b..") -> true
"""

from typing import List, Dict, Optional, Set
from collections import defaultdict

class TrieNode:
    """Standard Trie Node for wildcard search"""
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_word: bool = False

class WordDictionary1:
    """
    Approach 1: Trie with DFS for Wildcard Search
    
    Standard trie with recursive DFS to handle '.' wildcards.
    
    Time Complexity:
    - addWord: O(m) where m is word length
    - search: O(n * 26^k) where n is number of nodes, k is number of dots
    
    Space Complexity: O(ALPHABET_SIZE * N * M)
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def addWord(self, word: str) -> None:
        """Add a word to the data structure"""
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """Search for a word (with '.' wildcards allowed)"""
        return self._dfs_search(word, 0, self.root)
    
    def _dfs_search(self, word: str, index: int, node: TrieNode) -> bool:
        """DFS helper for wildcard search"""
        if index == len(word):
            return node.is_end_of_word
        
        char = word[index]
        
        if char == '.':
            # Try all possible children
            for child in node.children.values():
                if self._dfs_search(word, index + 1, child):
                    return True
            return False
        else:
            # Exact character match
            if char not in node.children:
                return False
            return self._dfs_search(word, index + 1, node.children[char])

class WordDictionary2:
    """
    Approach 2: Length-based Grouping with Set Matching
    
    Group words by length and use pattern matching for efficiency.
    
    Time Complexity:
    - addWord: O(m)
    - search: O(k) where k is number of words of same length
    
    Space Complexity: O(N * M)
    """
    
    def __init__(self):
        self.words_by_length: Dict[int, Set[str]] = defaultdict(set)
    
    def addWord(self, word: str) -> None:
        """Add word grouped by length"""
        self.words_by_length[len(word)].add(word)
    
    def search(self, word: str) -> bool:
        """Search using pattern matching"""
        length = len(word)
        if length not in self.words_by_length:
            return False
        
        # Check each word of same length
        for candidate in self.words_by_length[length]:
            if self._matches_pattern(candidate, word):
                return True
        
        return False
    
    def _matches_pattern(self, candidate: str, pattern: str) -> bool:
        """Check if candidate matches pattern with '.' wildcards"""
        if len(candidate) != len(pattern):
            return False
        
        for i in range(len(pattern)):
            if pattern[i] != '.' and pattern[i] != candidate[i]:
                return False
        
        return True

class WordDictionary3:
    """
    Approach 3: Trie with Optimized Wildcard Handling
    
    Optimized trie that pre-computes some wildcard patterns.
    
    Time Complexity:
    - addWord: O(m)
    - search: O(n * 26^k) but with optimizations
    
    Space Complexity: O(ALPHABET_SIZE * N * M)
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.words_by_length: Dict[int, List[str]] = defaultdict(list)
    
    def addWord(self, word: str) -> None:
        """Add word to both trie and length index"""
        # Add to trie
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        
        # Add to length index for optimization
        self.words_by_length[len(word)].append(word)
    
    def search(self, word: str) -> bool:
        """Optimized search with early termination"""
        # Quick check: if no words of this length exist
        if len(word) not in self.words_by_length:
            return False
        
        # If no wildcards, use simple trie search
        if '.' not in word:
            return self._simple_search(word)
        
        # Use DFS for wildcard search
        return self._dfs_search(word, 0, self.root)
    
    def _simple_search(self, word: str) -> bool:
        """Simple trie search without wildcards"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def _dfs_search(self, word: str, index: int, node: TrieNode) -> bool:
        """DFS with optimizations"""
        if index == len(word):
            return node.is_end_of_word
        
        char = word[index]
        
        if char == '.':
            # Optimization: if remaining pattern has no more wildcards,
            # we can be more selective
            remaining = word[index + 1:]
            if '.' not in remaining:
                # Check if any child can lead to the exact remaining pattern
                for child_char, child_node in node.children.items():
                    if self._exact_match_from_node(remaining, child_node):
                        return True
                return False
            else:
                # Standard wildcard search
                for child in node.children.values():
                    if self._dfs_search(word, index + 1, child):
                        return True
                return False
        else:
            if char not in node.children:
                return False
            return self._dfs_search(word, index + 1, node.children[char])
    
    def _exact_match_from_node(self, pattern: str, node: TrieNode) -> bool:
        """Check exact match from given node"""
        current = node
        for char in pattern:
            if char not in current.children:
                return False
            current = current.children[char]
        return current.is_end_of_word

class WordDictionary4:
    """
    Approach 4: Multiple Tries for Different Patterns
    
    Separate tries for different wildcard patterns for optimization.
    
    Time Complexity:
    - addWord: O(m * p) where p is number of pattern tries
    - search: O(m) for exact patterns, O(n * 26^k) for wildcards
    
    Space Complexity: O(p * ALPHABET_SIZE * N * M)
    """
    
    def __init__(self):
        self.exact_trie = TrieNode()  # For words without wildcards
        self.pattern_words: Dict[str, Set[str]] = defaultdict(set)  # Pattern -> words
        self.all_words: Set[str] = set()
    
    def addWord(self, word: str) -> None:
        """Add word to appropriate data structures"""
        self.all_words.add(word)
        
        # Add to exact trie
        node = self.exact_trie
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        
        # Generate and store patterns
        patterns = self._generate_patterns(word)
        for pattern in patterns:
            self.pattern_words[pattern].add(word)
    
    def _generate_patterns(self, word: str) -> List[str]:
        """Generate common wildcard patterns for the word"""
        patterns = []
        
        # Single character wildcard patterns
        for i in range(len(word)):
            pattern = word[:i] + '.' + word[i+1:]
            patterns.append(pattern)
        
        # Two character wildcard patterns (for short words)
        if len(word) <= 5:
            for i in range(len(word)):
                for j in range(i + 1, len(word)):
                    pattern = list(word)
                    pattern[i] = '.'
                    pattern[j] = '.'
                    patterns.append(''.join(pattern))
        
        return patterns
    
    def search(self, word: str) -> bool:
        """Search using pre-computed patterns or fallback to DFS"""
        # Check if exact pattern exists
        if word in self.pattern_words:
            return len(self.pattern_words[word]) > 0
        
        # Check for exact word (no wildcards)
        if '.' not in word:
            return self._exact_search(word)
        
        # Fallback to pattern matching
        for candidate in self.all_words:
            if len(candidate) == len(word) and self._matches_pattern(candidate, word):
                return True
        
        return False
    
    def _exact_search(self, word: str) -> bool:
        """Exact search in trie"""
        node = self.exact_trie
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def _matches_pattern(self, candidate: str, pattern: str) -> bool:
        """Pattern matching helper"""
        if len(candidate) != len(pattern):
            return False
        
        for i in range(len(pattern)):
            if pattern[i] != '.' and pattern[i] != candidate[i]:
                return False
        
        return True

class WordDictionary5:
    """
    Approach 5: Iterative Implementation with Stack
    
    Non-recursive implementation using explicit stack.
    
    Time Complexity: Same as recursive but avoids recursion overhead
    Space Complexity: O(stack size) instead of recursion stack
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def addWord(self, word: str) -> None:
        """Add word to trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """Iterative search using stack"""
        if not word:
            return self.root.is_end_of_word
        
        # Stack contains (node, word_index) pairs
        stack = [(self.root, 0)]
        
        while stack:
            node, index = stack.pop()
            
            if index == len(word):
                if node.is_end_of_word:
                    return True
                continue
            
            char = word[index]
            
            if char == '.':
                # Add all children to stack
                for child in node.children.values():
                    stack.append((child, index + 1))
            else:
                # Add specific child if exists
                if char in node.children:
                    stack.append((node.children[char], index + 1))
        
        return False

class WordDictionary6:
    """
    Approach 6: Enhanced with Additional Features
    
    Extended implementation with useful additional methods.
    
    Features:
    - Count matches
    - Get all matching words
    - Pattern statistics
    - Memory optimization
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0
        self.total_searches = 0
        self.wildcard_searches = 0
    
    def addWord(self, word: str) -> None:
        """Add word with statistics tracking"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end_of_word:
            self.word_count += 1
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """Search with statistics tracking"""
        self.total_searches += 1
        if '.' in word:
            self.wildcard_searches += 1
        
        return self._dfs_search(word, 0, self.root)
    
    def searchAndCount(self, word: str) -> int:
        """Count how many words match the pattern"""
        matches = []
        self._dfs_collect(word, 0, self.root, "", matches)
        return len(matches)
    
    def getAllMatches(self, word: str) -> List[str]:
        """Get all words that match the pattern"""
        matches = []
        self._dfs_collect(word, 0, self.root, "", matches)
        return matches
    
    def _dfs_search(self, word: str, index: int, node: TrieNode) -> bool:
        """Standard DFS search"""
        if index == len(word):
            return node.is_end_of_word
        
        char = word[index]
        
        if char == '.':
            for child in node.children.values():
                if self._dfs_search(word, index + 1, child):
                    return True
            return False
        else:
            if char not in node.children:
                return False
            return self._dfs_search(word, index + 1, node.children[char])
    
    def _dfs_collect(self, pattern: str, index: int, node: TrieNode, 
                     current_word: str, matches: List[str]) -> None:
        """Collect all matching words"""
        if index == len(pattern):
            if node.is_end_of_word:
                matches.append(current_word)
            return
        
        char = pattern[index]
        
        if char == '.':
            for child_char, child_node in node.children.items():
                self._dfs_collect(pattern, index + 1, child_node, 
                                current_word + child_char, matches)
        else:
            if char in node.children:
                self._dfs_collect(pattern, index + 1, node.children[char], 
                                current_word + char, matches)
    
    def getStats(self) -> Dict[str, int]:
        """Get usage statistics"""
        return {
            "total_words": self.word_count,
            "total_searches": self.total_searches,
            "wildcard_searches": self.wildcard_searches,
            "wildcard_percentage": (self.wildcard_searches / max(1, self.total_searches)) * 100
        }


def test_basic_operations():
    """Test basic WordDictionary operations"""
    print("=== Testing Basic Operations ===")
    
    implementations = [
        ("Trie DFS", WordDictionary1),
        ("Length Grouping", WordDictionary2),
        ("Optimized Trie", WordDictionary3),
        ("Pattern Tries", WordDictionary4),
        ("Iterative", WordDictionary5),
        ("Enhanced", WordDictionary6),
    ]
    
    # Test operations
    test_operations = [
        ("addWord", "bad"),
        ("addWord", "dad"),
        ("addWord", "mad"),
        ("search", "pad"),  # False
        ("search", "bad"),  # True
        ("search", ".ad"),  # True
        ("search", "b.."),  # True
        ("search", "..."),  # True
        ("search", "...."), # False
    ]
    
    for name, DictClass in implementations:
        print(f"\n--- Testing {name} ---")
        
        wd = DictClass()
        
        for operation, arg in test_operations:
            if operation == "addWord":
                wd.addWord(arg)
                print(f"  addWord('{arg}')")
            elif operation == "search":
                result = wd.search(arg)
                print(f"  search('{arg}') -> {result}")


def test_complex_patterns():
    """Test complex wildcard patterns"""
    print("\n=== Testing Complex Patterns ===")
    
    wd = WordDictionary1()
    
    # Add various words
    words = [
        "word", "world", "work", "walk", "wall", "ball", "call", "tall",
        "cat", "bat", "rat", "hat", "mat", "fat", "sat", "pat"
    ]
    
    for word in words:
        wd.addWord(word)
    
    print(f"Added words: {words}")
    
    # Test complex patterns
    patterns = [
        "w...",   # Should match word, work, walk, wall
        "...l",   # Should match wall, ball, call, tall
        ".at",    # Should match cat, bat, rat, hat, mat, fat, sat, pat
        "w.r.",   # Should match word, work
        "..ll",   # Should match wall, ball, call, tall
        "....",   # Should match all 4-letter words
        ".....",  # Should match all 5-letter words
        "c..",    # Should match cat
        "...t",   # Should not match anything (no 4-letter words ending in 't')
    ]
    
    for pattern in patterns:
        result = wd.search(pattern)
        print(f"  Pattern '{pattern}': {result}")


def test_enhanced_features():
    """Test enhanced WordDictionary features"""
    print("\n=== Testing Enhanced Features ===")
    
    wd = WordDictionary6()
    
    # Add words
    words = ["cat", "car", "card", "care", "careful", "call", "bat", "bar"]
    for word in words:
        wd.addWord(word)
    
    print(f"Added words: {words}")
    
    # Test enhanced features
    patterns = ["ca.", "c.r", "..r", "...."]
    
    for pattern in patterns:
        exists = wd.search(pattern)
        count = wd.searchAndCount(pattern)
        matches = wd.getAllMatches(pattern)
        
        print(f"  Pattern '{pattern}': exists={exists}, count={count}, matches={matches}")
    
    # Show statistics
    stats = wd.getStats()
    print(f"\nStatistics: {stats}")


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import time
    import random
    import string
    
    # Generate test data
    def generate_words(n: int, length: int) -> List[str]:
        words = []
        for _ in range(n):
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.append(word)
        return list(set(words))  # Remove duplicates
    
    def generate_patterns(words: List[str], n_patterns: int) -> List[str]:
        patterns = []
        for _ in range(n_patterns):
            word = random.choice(words)
            pattern = list(word)
            # Replace some characters with wildcards
            for i in range(len(pattern)):
                if random.random() < 0.3:  # 30% chance to become wildcard
                    pattern[i] = '.'
            patterns.append(''.join(pattern))
        return patterns
    
    test_words = generate_words(500, 4)
    test_patterns = generate_patterns(test_words, 100)
    
    implementations = [
        ("Trie DFS", WordDictionary1),
        ("Length Group", WordDictionary2),
        ("Optimized", WordDictionary3),
        ("Iterative", WordDictionary5),
    ]
    
    print(f"Testing with {len(test_words)} words, {len(test_patterns)} patterns")
    
    for name, DictClass in implementations:
        start_time = time.time()
        
        wd = DictClass()
        
        # Add words
        for word in test_words:
            wd.addWord(word)
        
        # Search patterns
        results = []
        for pattern in test_patterns:
            results.append(wd.search(pattern))
        
        end_time = time.time()
        
        true_count = sum(results)
        print(f"{name:12}: {(end_time - start_time)*1000:.2f}ms, {true_count}/{len(test_patterns)} matches")


def demonstrate_real_world_usage():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Crossword Puzzle Solver
    print("1. Crossword Puzzle Helper:")
    crossword = WordDictionary1()
    
    # Dictionary words
    dictionary = ["cat", "car", "card", "care", "arc", "are", "ear", "era", "red", "read"]
    for word in dictionary:
        crossword.addWord(word)
    
    # Crossword clues (known letters with blanks)
    clues = ["c.r", ".ar", "r.d", "..e"]
    print(f"   Dictionary: {dictionary}")
    print(f"   Crossword clues:")
    for clue in clues:
        matches = crossword.search(clue)
        print(f"     '{clue}' -> possible: {matches}")
    
    # Application 2: Fuzzy String Matching
    print("\n2. Fuzzy String Matching:")
    fuzzy = WordDictionary6()
    
    # Product names
    products = ["iPhone", "iPad", "iMac", "MacBook", "iPod", "iWatch"]
    for product in products:
        fuzzy.addWord(product.lower())
    
    # User searches with typos (simulated with wildcards)
    searches = ["i....", ".pad", "mac..", "ipho.e"]
    print(f"   Products: {products}")
    print(f"   Fuzzy searches:")
    for search in searches:
        matches = fuzzy.getAllMatches(search)
        print(f"     '{search}' -> {matches}")
    
    # Application 3: Pattern Validation
    print("\n3. Pattern Validation:")
    validator = WordDictionary1()
    
    # Valid patterns for some domain
    patterns = ["abc123", "def456", "ghi789", "xyz000"]
    for pattern in patterns:
        validator.addWord(pattern)
    
    # Test inputs
    test_inputs = ["abc123", "ab.123", "...456", "xyz..."]
    print(f"   Valid patterns: {patterns}")
    print(f"   Validation tests:")
    for test_input in test_inputs:
        valid = validator.search(test_input)
        print(f"     '{test_input}' -> valid: {valid}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    wd = WordDictionary1()
    
    # Edge cases
    edge_cases = [
        ("", "empty string"),
        (".", "single wildcard"),
        ("a", "single character"),
        ("...", "all wildcards"),
        ("a.b.c", "alternating pattern"),
    ]
    
    # Add some words first
    wd.addWord("a")
    wd.addWord("ab")
    wd.addWord("abc")
    wd.addWord("abcd")
    
    print("Testing edge cases:")
    for pattern, description in edge_cases:
        try:
            result = wd.search(pattern)
            print(f"  '{pattern}' ({description}): {result}")
        except Exception as e:
            print(f"  '{pattern}' ({description}): Error - {e}")


if __name__ == "__main__":
    test_basic_operations()
    test_complex_patterns()
    test_enhanced_features()
    benchmark_approaches()
    demonstrate_real_world_usage()
    test_edge_cases()

"""
211. Design Add and Search Words Data Structure showcases multiple approaches:

1. Trie with DFS - Standard recursive approach for wildcard matching
2. Length Grouping - Optimization by grouping words of same length
3. Optimized Trie - Enhanced trie with pattern optimizations
4. Pattern Tries - Multiple tries for different pattern types
5. Iterative - Stack-based non-recursive implementation
6. Enhanced - Extended features with statistics and pattern analysis

Each approach demonstrates different optimization strategies for handling
wildcard search patterns efficiently in various scenarios.
"""
