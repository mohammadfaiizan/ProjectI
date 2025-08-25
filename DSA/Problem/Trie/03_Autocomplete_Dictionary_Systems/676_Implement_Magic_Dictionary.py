"""
676. Implement Magic Dictionary - Multiple Approaches
Difficulty: Medium

Design a data structure that is initialized with a list of different words. 
Provided a string, you should determine if you can change exactly one character 
in this string to match any word in the data structure.

Implement the MagicDictionary class:
- MagicDictionary() Initializes the object.
- void buildDict(String[] dictionary) Sets the data structure with an array of distinct strings dictionary.
- bool search(String searchWord) Returns true if you can change exactly one character in searchWord to match any string in the data structure, otherwise returns false.

LeetCode Problem: https://leetcode.com/problems/implement-magic-dictionary/

Example:
Input: ["MagicDictionary", "buildDict", "search", "search", "search", "search"]
[[], [["hello", "leetcode"]], ["hello"], ["hhllo"], ["hell"], ["leetcoded"]]
Output: [null, null, false, true, false, false]
"""

from typing import List, Set, Dict, Tuple
from collections import defaultdict

class TrieNode:
    """Trie node for magic dictionary"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.words = []  # Store words ending at this node

class MagicDictionary1:
    """
    Approach 1: Brute Force with Character Replacement
    
    For each search, try replacing each character and check if result exists.
    """
    
    def __init__(self):
        """Initialize the magic dictionary"""
        self.words = set()
    
    def buildDict(self, dictionary: List[str]) -> None:
        """
        Build dictionary from word list.
        
        Time: O(sum of word lengths)
        Space: O(sum of word lengths)
        """
        self.words = set(dictionary)
    
    def search(self, searchWord: str) -> bool:
        """
        Search for word with exactly one character difference.
        
        Time: O(|searchWord| * 26 * average_word_length)
        Space: O(1)
        """
        for i in range(len(searchWord)):
            # Try replacing character at position i
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != searchWord[i]:  # Must be different character
                    candidate = searchWord[:i] + c + searchWord[i+1:]
                    if candidate in self.words:
                        return True
        
        return False


class MagicDictionary2:
    """
    Approach 2: Trie with Wildcard Search
    
    Build trie and search allowing exactly one mismatch.
    """
    
    def __init__(self):
        """Initialize the magic dictionary"""
        self.root = TrieNode()
    
    def buildDict(self, dictionary: List[str]) -> None:
        """
        Build trie from dictionary.
        
        Time: O(sum of word lengths)
        Space: O(sum of word lengths)
        """
        for word in dictionary:
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
            node.words.append(word)
    
    def search(self, searchWord: str) -> bool:
        """
        Search trie allowing exactly one mismatch.
        
        Time: O(26 * |searchWord|) in worst case
        Space: O(|searchWord|) for recursion
        """
        def dfs(node: TrieNode, word_idx: int, mismatches: int) -> bool:
            """DFS with mismatch counting"""
            # Base case: processed entire word
            if word_idx == len(searchWord):
                return node.is_end and mismatches == 1
            
            # Pruning: too many mismatches
            if mismatches > 1:
                return False
            
            char = searchWord[word_idx]
            
            # Try exact match
            if char in node.children:
                if dfs(node.children[char], word_idx + 1, mismatches):
                    return True
            
            # Try mismatch (if we haven't used our one mismatch yet)
            if mismatches == 0:
                for child_char, child_node in node.children.items():
                    if child_char != char:
                        if dfs(child_node, word_idx + 1, 1):
                            return True
            
            return False
        
        return dfs(self.root, 0, 0)


class MagicDictionary3:
    """
    Approach 3: Length-based Grouping with Pattern Matching
    
    Group words by length and use pattern matching.
    """
    
    def __init__(self):
        """Initialize the magic dictionary"""
        self.words_by_length = defaultdict(list)
    
    def buildDict(self, dictionary: List[str]) -> None:
        """
        Group words by length.
        
        Time: O(sum of word lengths)
        Space: O(sum of word lengths)
        """
        for word in dictionary:
            self.words_by_length[len(word)].append(word)
    
    def search(self, searchWord: str) -> bool:
        """
        Search among words of same length.
        
        Time: O(words_of_same_length * |searchWord|)
        Space: O(1)
        """
        target_length = len(searchWord)
        
        for word in self.words_by_length[target_length]:
            if self._has_one_diff(searchWord, word):
                return True
        
        return False
    
    def _has_one_diff(self, word1: str, word2: str) -> bool:
        """Check if two words differ by exactly one character"""
        if len(word1) != len(word2):
            return False
        
        diff_count = 0
        for i in range(len(word1)):
            if word1[i] != word2[i]:
                diff_count += 1
                if diff_count > 1:
                    return False
        
        return diff_count == 1


class MagicDictionary4:
    """
    Approach 4: Pattern-based Hashing
    
    Generate patterns with wildcards and use hash map.
    """
    
    def __init__(self):
        """Initialize the magic dictionary"""
        self.patterns = defaultdict(list)
    
    def buildDict(self, dictionary: List[str]) -> None:
        """
        Generate patterns for each word.
        
        Time: O(sum of word lengths squared)
        Space: O(sum of word lengths squared)
        """
        for word in dictionary:
            # Generate all patterns with one wildcard
            for i in range(len(word)):
                pattern = word[:i] + '*' + word[i+1:]
                self.patterns[pattern].append(word)
    
    def search(self, searchWord: str) -> bool:
        """
        Check if any pattern matches search word.
        
        Time: O(|searchWord|)
        Space: O(1)
        """
        for i in range(len(searchWord)):
            pattern = searchWord[:i] + '*' + searchWord[i+1:]
            
            # Check if any word with this pattern exists and is different
            for word in self.patterns[pattern]:
                if word != searchWord and len(word) == len(searchWord):
                    return True
        
        return False


class MagicDictionary5:
    """
    Approach 5: Edit Distance with Constraint
    
    Use modified edit distance allowing exactly one substitution.
    """
    
    def __init__(self):
        """Initialize the magic dictionary"""
        self.words = []
    
    def buildDict(self, dictionary: List[str]) -> None:
        """
        Store words for edit distance computation.
        
        Time: O(1)
        Space: O(sum of word lengths)
        """
        self.words = dictionary
    
    def search(self, searchWord: str) -> bool:
        """
        Check edit distance with each word.
        
        Time: O(number_of_words * |searchWord| * max_word_length)
        Space: O(|searchWord| * max_word_length)
        """
        for word in self.words:
            if self._is_one_edit_substitution(searchWord, word):
                return True
        
        return False
    
    def _is_one_edit_substitution(self, word1: str, word2: str) -> bool:
        """Check if words differ by exactly one substitution"""
        if len(word1) != len(word2):
            return False
        
        m = len(word1)
        
        # DP table: dp[i][j] = min edits to transform word1[:i] to word2[:j]
        dp = [[float('inf')] * (m + 1) for _ in range(m + 1)]
        
        # Base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, m + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # Only allow substitution (not insertion/deletion)
                    dp[i][j] = dp[i-1][j-1] + 1
        
        return dp[m][m] == 1


class MagicDictionary6:
    """
    Approach 6: Neighborhood Graph
    
    Build graph of words that are one edit apart.
    """
    
    def __init__(self):
        """Initialize the magic dictionary"""
        self.words = set()
        self.neighbors = defaultdict(set)
    
    def buildDict(self, dictionary: List[str]) -> None:
        """
        Build neighborhood graph.
        
        Time: O(number_of_words^2 * max_word_length)
        Space: O(number_of_words^2)
        """
        self.words = set(dictionary)
        
        # Build neighborhood relationships
        word_list = list(dictionary)
        for i in range(len(word_list)):
            for j in range(i + 1, len(word_list)):
                word1, word2 = word_list[i], word_list[j]
                if self._are_neighbors(word1, word2):
                    self.neighbors[word1].add(word2)
                    self.neighbors[word2].add(word1)
    
    def _are_neighbors(self, word1: str, word2: str) -> bool:
        """Check if two words are exactly one edit apart"""
        if len(word1) != len(word2):
            return False
        
        diff_count = 0
        for i in range(len(word1)):
            if word1[i] != word2[i]:
                diff_count += 1
                if diff_count > 1:
                    return False
        
        return diff_count == 1
    
    def search(self, searchWord: str) -> bool:
        """
        Check if search word has any neighbors in dictionary.
        
        Time: O(|searchWord| * 26)
        Space: O(1)
        """
        # Generate all possible one-character modifications
        for i in range(len(searchWord)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != searchWord[i]:
                    candidate = searchWord[:i] + c + searchWord[i+1:]
                    if candidate in self.words:
                        return True
        
        return False


class MagicDictionary7:
    """
    Approach 7: Optimized Trie with Early Termination
    
    Enhanced trie with pruning and optimizations.
    """
    
    def __init__(self):
        """Initialize the magic dictionary"""
        self.root = TrieNode()
    
    def buildDict(self, dictionary: List[str]) -> None:
        """
        Build optimized trie.
        
        Time: O(sum of word lengths)
        Space: O(sum of word lengths)
        """
        for word in dictionary:
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
    
    def search(self, searchWord: str) -> bool:
        """
        Optimized search with early termination.
        
        Time: O(26 * |searchWord|) with pruning
        Space: O(|searchWord|)
        """
        def dfs(node: TrieNode, index: int, used_mismatch: bool) -> bool:
            """DFS with mismatch tracking and pruning"""
            if index == len(searchWord):
                return node.is_end and used_mismatch
            
            char = searchWord[index]
            
            # Exact match path
            if char in node.children:
                if dfs(node.children[char], index + 1, used_mismatch):
                    return True
            
            # Mismatch path (only if we haven't used our mismatch yet)
            if not used_mismatch:
                for child_char, child_node in node.children.items():
                    if child_char != char:
                        if dfs(child_node, index + 1, True):
                            return True
            
            return False
        
        return dfs(self.root, 0, False)


def test_magic_dictionary():
    """Test all magic dictionary implementations"""
    print("=== Testing Magic Dictionary Implementations ===")
    
    # Test data
    dictionary = ["hello", "leetcode", "judge", "dad", "mad"]
    test_cases = [
        ("hello", False),    # Same word
        ("hhllo", True),     # One character different
        ("hell", False),     # Different length
        ("leetcoded", False), # Different length
        ("judge", False),    # Same word
        ("jbdge", True),     # One character different
        ("dad", False),      # Same word
        ("mad", False),      # Same word (exists in dictionary)
        ("bad", True),       # One character different from "dad"
    ]
    
    implementations = [
        ("Brute Force", MagicDictionary1),
        ("Trie Wildcard", MagicDictionary2),
        ("Length Grouping", MagicDictionary3),
        ("Pattern Hashing", MagicDictionary4),
        ("Edit Distance", MagicDictionary5),
        ("Neighborhood Graph", MagicDictionary6),
        ("Optimized Trie", MagicDictionary7),
    ]
    
    for name, DictClass in implementations:
        print(f"\n{name}:")
        
        # Initialize and build dictionary
        magic_dict = DictClass()
        magic_dict.buildDict(dictionary)
        
        # Test search operations
        for word, expected in test_cases:
            result = magic_dict.search(word)
            status = "✓" if result == expected else "✗"
            print(f"  search('{word}'): {result} {status}")


def demonstrate_usage_patterns():
    """Demonstrate different usage patterns"""
    print("\n=== Usage Patterns Demo ===")
    
    # Pattern 1: Spell checker
    print("1. Spell Checker Application:")
    
    correct_words = ["python", "programming", "algorithm", "data", "structure"]
    spell_checker = MagicDictionary2()
    spell_checker.buildDict(correct_words)
    
    user_inputs = ["pytho", "programing", "algoritm", "dta", "structur"]
    
    print(f"   Dictionary: {correct_words}")
    print(f"   User inputs with typos:")
    
    for user_input in user_inputs:
        can_fix = spell_checker.search(user_input)
        print(f"     '{user_input}': {'Can be corrected' if can_fix else 'Cannot be corrected'}")
    
    # Pattern 2: Word game
    print(f"\n2. Word Game - One Letter Change:")
    
    valid_words = ["cat", "bat", "hat", "car", "bar", "far"]
    game_dict = MagicDictionary3()
    game_dict.buildDict(valid_words)
    
    print(f"   Valid words: {valid_words}")
    print(f"   Player attempts:")
    
    attempts = ["cot", "bet", "hit", "cap", "bag"]
    
    for attempt in attempts:
        is_valid = game_dict.search(attempt)
        print(f"     '{attempt}': {'Valid move' if is_valid else 'Invalid move'}")
    
    # Pattern 3: Fuzzy matching
    print(f"\n3. Fuzzy String Matching:")
    
    database_entries = ["user123", "admin", "guest", "developer", "tester"]
    fuzzy_matcher = MagicDictionary4()
    fuzzy_matcher.buildDict(database_entries)
    
    print(f"   Database entries: {database_entries}")
    print(f"   Fuzzy queries:")
    
    queries = ["user124", "admen", "geast", "developr"]
    
    for query in queries:
        has_match = fuzzy_matcher.search(query)
        print(f"     '{query}': {'Close match found' if has_match else 'No close match'}")


def benchmark_implementations():
    """Benchmark different implementations"""
    print("\n=== Benchmarking Implementations ===")
    
    import time
    import random
    import string
    
    # Generate test data
    def generate_dictionary(size: int, avg_length: int) -> List[str]:
        words = []
        for _ in range(size):
            length = max(1, avg_length + random.randint(-2, 2))
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.append(word)
        return list(set(words))  # Remove duplicates
    
    def generate_test_words(dictionary: List[str], count: int) -> List[str]:
        test_words = []
        for _ in range(count):
            # 50% chance to modify existing word, 50% random word
            if random.random() < 0.5 and dictionary:
                # Modify existing word
                word = random.choice(dictionary)
                if word:
                    pos = random.randint(0, len(word) - 1)
                    new_char = random.choice(string.ascii_lowercase)
                    modified = word[:pos] + new_char + word[pos+1:]
                    test_words.append(modified)
            else:
                # Random word
                length = random.randint(3, 8)
                word = ''.join(random.choices(string.ascii_lowercase, k=length))
                test_words.append(word)
        
        return test_words
    
    test_scenarios = [
        ("Small", generate_dictionary(50, 5)),
        ("Medium", generate_dictionary(200, 7)),
        ("Large", generate_dictionary(1000, 8)),
    ]
    
    implementations = [
        ("Brute Force", MagicDictionary1),
        ("Trie Wildcard", MagicDictionary2),
        ("Length Grouping", MagicDictionary3),
        ("Pattern Hashing", MagicDictionary4),
    ]
    
    for scenario_name, dictionary in test_scenarios:
        print(f"\n--- {scenario_name} Dataset ---")
        print(f"Dictionary size: {len(dictionary)}, Avg word length: {sum(len(w) for w in dictionary)/len(dictionary):.1f}")
        
        test_words = generate_test_words(dictionary, 20)
        
        for impl_name, DictClass in implementations:
            # Measure build time
            start_time = time.time()
            magic_dict = DictClass()
            magic_dict.buildDict(dictionary)
            build_time = time.time() - start_time
            
            # Measure search time
            start_time = time.time()
            for word in test_words:
                magic_dict.search(word)
            search_time = (time.time() - start_time) / len(test_words)
            
            print(f"  {impl_name:15}: Build {build_time*1000:.2f}ms, Search {search_time*1000:.3f}ms/query")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    magic_dict = MagicDictionary2()
    
    edge_cases = [
        # Empty dictionary
        ([], ["hello"], "Empty dictionary"),
        
        # Single character words
        (["a", "b", "c"], ["x", "a", "b"], "Single character words"),
        
        # Same length words only
        (["abc", "def", "ghi"], ["abd", "abc", "xyz"], "Same length words"),
        
        # Very long words
        (["abcdefghijk"], ["abcdefghijl", "abcdefghijk"], "Very long words"),
        
        # All same words
        (["test", "test", "test"], ["tast", "test"], "Duplicate words"),
        
        # Mixed lengths
        (["a", "ab", "abc", "abcd"], ["b", "ac", "abd", "abce"], "Mixed lengths"),
    ]
    
    for dictionary, test_words, description in edge_cases:
        print(f"\n{description}:")
        print(f"  Dictionary: {dictionary}")
        print(f"  Test words: {test_words}")
        
        try:
            magic_dict.buildDict(dictionary)
            for word in test_words:
                result = magic_dict.search(word)
                print(f"    search('{word}'): {result}")
        except Exception as e:
            print(f"  Error: {e}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Brute Force Character Replacement",
         "Build: O(sum of word lengths)",
         "Search: O(|searchWord| * 26 * average_word_length)",
         "Space: O(sum of word lengths)"),
        
        ("Trie with Wildcard Search",
         "Build: O(sum of word lengths)",
         "Search: O(26 * |searchWord|) worst case",
         "Space: O(sum of word lengths)"),
        
        ("Length-based Grouping",
         "Build: O(sum of word lengths)",
         "Search: O(words_of_same_length * |searchWord|)",
         "Space: O(sum of word lengths)"),
        
        ("Pattern-based Hashing",
         "Build: O(sum of word lengths squared)",
         "Search: O(|searchWord|)",
         "Space: O(sum of word lengths squared)"),
        
        ("Edit Distance",
         "Build: O(1)",
         "Search: O(number_of_words * |searchWord| * max_word_length)",
         "Space: O(|searchWord| * max_word_length)"),
        
        ("Neighborhood Graph",
         "Build: O(number_of_words^2 * max_word_length)",
         "Search: O(|searchWord| * 26)",
         "Space: O(number_of_words^2)"),
    ]
    
    print("Implementation Analysis:")
    for impl, build_complexity, search_complexity, space_complexity in complexity_analysis:
        print(f"\n{impl}:")
        print(f"  Build: {build_complexity}")
        print(f"  Search: {search_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nRecommendations:")
    print(f"  • Use Trie with Wildcard for balanced performance")
    print(f"  • Use Pattern Hashing for optimal search time (high space cost)")
    print(f"  • Use Length Grouping for simple implementation")
    print(f"  • Use Brute Force for small dictionaries")


if __name__ == "__main__":
    test_magic_dictionary()
    demonstrate_usage_patterns()
    benchmark_implementations()
    test_edge_cases()
    analyze_complexity()

"""
676. Implement Magic Dictionary demonstrates multiple approaches for one-edit matching:

1. Brute Force Character Replacement - Try all possible single character changes
2. Trie with Wildcard Search - Build trie and search allowing exactly one mismatch
3. Length-based Grouping - Group by length and compare among same-length words
4. Pattern-based Hashing - Generate wildcard patterns and use hash map lookup
5. Edit Distance with Constraint - Use DP to verify exactly one substitution
6. Neighborhood Graph - Pre-compute graph of words that are one edit apart
7. Optimized Trie - Enhanced trie with pruning and early termination

Each approach offers different trade-offs between preprocessing time,
search efficiency, and memory usage for single-character-difference matching.
"""

