"""
Dynamic Programming - Trie Combined Patterns
This module implements DP problems combined with Trie data structures including
word break, concatenated words, palindrome pairs, and advanced string matching.
"""

from typing import List, Dict, Tuple, Optional, Set
import time
from collections import defaultdict

# ==================== TRIE DATA STRUCTURE ====================

class TrieNode:
    """Trie node for efficient string operations"""
    def __init__(self):
        self.children = {}
        self.is_end_word = False
        self.word_index = -1  # For storing original word index
        self.palindrome_suffixes = []  # For palindrome pairs

class Trie:
    """
    Trie data structure with DP optimization features
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str, index: int = -1):
        """Insert word into trie with optional index"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_word = True
        node.word_index = index
    
    def search(self, word: str) -> bool:
        """Search for complete word in trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return node.is_end_word
    
    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with given prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return True

# ==================== WORD BREAK PROBLEMS ====================

class WordBreakProblems:
    """
    Word Break Problems using Trie + DP
    
    Various word breaking problems with efficient trie-based solutions.
    """
    
    def word_break(self, s: str, word_dict: List[str]) -> bool:
        """
        Word Break I - Check if string can be segmented
        
        LeetCode 139 - Word Break
        
        Time Complexity: O(n² + m*k) where n=len(s), m=len(word_dict), k=avg word length
        Space Complexity: O(m*k + n)
        
        Args:
            s: String to segment
            word_dict: List of valid words
        
        Returns:
            True if string can be segmented into dictionary words
        """
        # Build trie
        trie = Trie()
        for word in word_dict:
            trie.insert(word)
        
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True  # Empty string can always be segmented
        
        for i in range(1, n + 1):
            node = trie.root
            
            # Check all possible word endings at position i
            for j in range(i - 1, -1, -1):
                char = s[j]
                
                if char not in node.children:
                    break
                
                node = node.children[char]
                
                if node.is_end_word and dp[j]:
                    dp[i] = True
                    break
        
        return dp[n]
    
    def word_break_ii(self, s: str, word_dict: List[str]) -> List[str]:
        """
        Word Break II - Return all possible sentences
        
        LeetCode 140 - Word Break II
        
        Args:
            s: String to segment
            word_dict: List of valid words
        
        Returns:
            List of all possible sentences
        """
        # Build trie
        trie = Trie()
        for word in word_dict:
            trie.insert(word)
        
        n = len(s)
        
        # First check if segmentation is possible
        dp_possible = [False] * (n + 1)
        dp_possible[0] = True
        
        for i in range(1, n + 1):
            node = trie.root
            for j in range(i - 1, -1, -1):
                char = s[j]
                if char not in node.children:
                    break
                node = node.children[char]
                if node.is_end_word and dp_possible[j]:
                    dp_possible[i] = True
                    break
        
        if not dp_possible[n]:
            return []
        
        # DP to store all possible segmentations
        dp = [[] for _ in range(n + 1)]
        dp[0] = [""]
        
        for i in range(1, n + 1):
            if not dp_possible[i]:
                continue
            
            node = trie.root
            for j in range(i - 1, -1, -1):
                char = s[j]
                if char not in node.children:
                    break
                node = node.children[char]
                
                if node.is_end_word and dp[j]:
                    word = s[j:i]
                    for sentence in dp[j]:
                        if sentence:
                            dp[i].append(sentence + " " + word)
                        else:
                            dp[i].append(word)
        
        return dp[n]
    
    def word_break_with_concatenation(self, s: str, word_dict: List[str], 
                                    min_words: int) -> bool:
        """
        Check if string can be segmented using at least min_words
        
        Args:
            s: String to segment
            word_dict: List of valid words
            min_words: Minimum number of words required
        
        Returns:
            True if string can be segmented with enough words
        """
        trie = Trie()
        for word in word_dict:
            trie.insert(word)
        
        n = len(s)
        # dp[i] = minimum words needed to segment s[:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            node = trie.root
            for j in range(i - 1, -1, -1):
                char = s[j]
                if char not in node.children:
                    break
                node = node.children[char]
                
                if node.is_end_word and dp[j] != float('inf'):
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n] >= min_words
    
    def word_break_with_costs(self, s: str, word_costs: Dict[str, int]) -> int:
        """
        Minimum cost to segment string using weighted words
        
        Args:
            s: String to segment
            word_costs: Dictionary mapping words to their costs
        
        Returns:
            Minimum cost to segment string, -1 if impossible
        """
        trie = Trie()
        for word in word_costs:
            trie.insert(word)
        
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            node = trie.root
            for j in range(i - 1, -1, -1):
                char = s[j]
                if char not in node.children:
                    break
                node = node.children[char]
                
                if node.is_end_word:
                    word = s[j:i]
                    if word in word_costs and dp[j] != float('inf'):
                        dp[i] = min(dp[i], dp[j] + word_costs[word])
        
        return dp[n] if dp[n] != float('inf') else -1

# ==================== CONCATENATED WORDS ====================

class ConcatenatedWordsProblems:
    """
    Concatenated Words Problems using Trie + DP
    
    Find words that can be formed by concatenating other words.
    """
    
    def find_all_concatenated_words(self, words: List[str]) -> List[str]:
        """
        Find all concatenated words in the list
        
        LeetCode 472 - Concatenated Words
        
        Time Complexity: O(n * L² + total_length) where n=number of words, L=max length
        Space Complexity: O(total_length)
        
        Args:
            words: List of words
        
        Returns:
            List of words that are concatenated from other words
        """
        # Build trie with all words
        trie = Trie()
        word_set = set(words)
        
        # Sort by length to process shorter words first
        words.sort(key=len)
        
        for word in words:
            trie.insert(word)
        
        result = []
        
        for word in words:
            if self._can_form_from_others(word, word_set, trie):
                result.append(word)
        
        return result
    
    def _can_form_from_others(self, word: str, word_set: Set[str], trie: Trie) -> bool:
        """Check if word can be formed by concatenating other words"""
        n = len(word)
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(1, n + 1):
            node = trie.root
            for j in range(i - 1, -1, -1):
                char = word[j]
                if char not in node.children:
                    break
                node = node.children[char]
                
                if node.is_end_word:
                    sub_word = word[j:i]
                    # Word cannot be formed using itself only
                    if sub_word == word:
                        continue
                    if sub_word in word_set and dp[j]:
                        dp[i] = True
                        break
        
        return dp[n]
    
    def count_concatenated_words(self, words: List[str], min_parts: int) -> int:
        """
        Count words that can be formed by concatenating at least min_parts words
        
        Args:
            words: List of words
            min_parts: Minimum number of parts required
        
        Returns:
            Count of valid concatenated words
        """
        trie = Trie()
        word_set = set(words)
        
        for word in words:
            trie.insert(word)
        
        count = 0
        
        for word in words:
            if self._count_min_parts(word, word_set, trie) >= min_parts:
                count += 1
        
        return count
    
    def _count_min_parts(self, word: str, word_set: Set[str], trie: Trie) -> int:
        """Count minimum parts needed to form word"""
        n = len(word)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            node = trie.root
            for j in range(i - 1, -1, -1):
                char = word[j]
                if char not in node.children:
                    break
                node = node.children[char]
                
                if node.is_end_word:
                    sub_word = word[j:i]
                    if sub_word == word:
                        continue
                    if sub_word in word_set and dp[j] != float('inf'):
                        dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n] if dp[n] != float('inf') else 0
    
    def longest_concatenated_word(self, words: List[str]) -> str:
        """
        Find the longest word that can be formed by concatenating other words
        
        Args:
            words: List of words
        
        Returns:
            Longest concatenated word
        """
        trie = Trie()
        word_set = set(words)
        
        # Sort by length (descending) to find longest first
        words.sort(key=len, reverse=True)
        
        for word in words:
            trie.insert(word)
        
        for word in words:
            if self._can_form_from_others(word, word_set, trie):
                return word
        
        return ""
    
    def concatenated_words_with_limit(self, words: List[str], max_parts: int) -> List[str]:
        """
        Find concatenated words using at most max_parts
        
        Args:
            words: List of words
            max_parts: Maximum number of parts allowed
        
        Returns:
            List of valid concatenated words
        """
        trie = Trie()
        word_set = set(words)
        
        for word in words:
            trie.insert(word)
        
        result = []
        
        for word in words:
            min_parts = self._count_min_parts(word, word_set, trie)
            if 2 <= min_parts <= max_parts:
                result.append(word)
        
        return result

# ==================== PALINDROME PAIRS ====================

class PalindromePairsProblems:
    """
    Palindrome Pairs Problems using Trie + DP
    
    Find pairs of words that form palindromes when concatenated.
    """
    
    def palindrome_pairs(self, words: List[str]) -> List[List[int]]:
        """
        Find all pairs of words that form palindromes when concatenated
        
        LeetCode 336 - Palindrome Pairs
        
        Time Complexity: O(n * k²) where n=number of words, k=average length
        Space Complexity: O(n * k)
        
        Args:
            words: List of words
        
        Returns:
            List of [i, j] pairs where words[i] + words[j] is palindrome
        """
        def is_palindrome(s: str) -> bool:
            return s == s[::-1]
        
        def is_palindrome_range(s: str, start: int, end: int) -> bool:
            while start < end:
                if s[start] != s[end]:
                    return False
                start += 1
                end -= 1
            return True
        
        # Build trie with reversed words
        trie = Trie()
        
        for i, word in enumerate(words):
            # Insert reversed word with palindrome suffix information
            reversed_word = word[::-1]
            node = trie.root
            
            for j, char in enumerate(reversed_word):
                # Check if remaining suffix is palindrome
                if is_palindrome_range(reversed_word, j, len(reversed_word) - 1):
                    node.palindrome_suffixes.append(i)
                
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            
            node.is_end_word = True
            node.word_index = i
            node.palindrome_suffixes.append(i)
        
        result = []
        
        for i, word in enumerate(words):
            node = trie.root
            
            # Case 1: word is longer, check if prefix matches reversed word
            for j, char in enumerate(word):
                if node.is_end_word and node.word_index != i:
                    if is_palindrome_range(word, j, len(word) - 1):
                        result.append([i, node.word_index])
                
                if char not in node.children:
                    break
                node = node.children[char]
            else:
                # Case 2: word completely matched, check palindrome suffixes
                if node.is_end_word and node.word_index != i:
                    result.append([i, node.word_index])
                
                for word_idx in node.palindrome_suffixes:
                    if word_idx != i:
                        result.append([i, word_idx])
        
        return result
    
    def count_palindrome_pairs(self, words: List[str]) -> int:
        """
        Count total number of palindrome pairs
        
        Args:
            words: List of words
        
        Returns:
            Number of palindrome pairs
        """
        return len(self.palindrome_pairs(words))
    
    def longest_palindrome_pair(self, words: List[str]) -> Tuple[int, int, str]:
        """
        Find the longest palindrome formed by concatenating two words
        
        Args:
            words: List of words
        
        Returns:
            Tuple of (index1, index2, palindrome_string)
        """
        pairs = self.palindrome_pairs(words)
        
        max_length = 0
        best_pair = (-1, -1)
        best_palindrome = ""
        
        for i, j in pairs:
            palindrome = words[i] + words[j]
            if len(palindrome) > max_length:
                max_length = len(palindrome)
                best_pair = (i, j)
                best_palindrome = palindrome
        
        return best_pair[0], best_pair[1], best_palindrome
    
    def palindrome_pairs_with_constraints(self, words: List[str], 
                                        min_length: int, max_length: int) -> List[List[int]]:
        """
        Find palindrome pairs with length constraints
        
        Args:
            words: List of words
            min_length: Minimum palindrome length
            max_length: Maximum palindrome length
        
        Returns:
            List of valid palindrome pairs
        """
        pairs = self.palindrome_pairs(words)
        
        result = []
        for i, j in pairs:
            palindrome_length = len(words[i]) + len(words[j])
            if min_length <= palindrome_length <= max_length:
                result.append([i, j])
        
        return result

# ==================== PERFORMANCE ANALYSIS ====================

def performance_comparison():
    """Compare performance of different Trie + DP approaches"""
    print("=== Trie + DP Performance Analysis ===\n")
    
    # Test word break
    word_break = WordBreakProblems()
    
    test_cases = [
        ("leetcode", ["leet", "code"]),
        ("applepenapple", ["apple", "pen"]),
        ("catsandog", ["cats", "dog", "sand", "and", "cat"])
    ]
    
    print("Word Break Performance:")
    for s, word_dict in test_cases:
        start_time = time.time()
        result = word_break.word_break(s, word_dict)
        time_taken = time.time() - start_time
        print(f"  '{s}': {result} ({time_taken:.6f}s)")
    
    # Test concatenated words
    concat = ConcatenatedWordsProblems()
    words = ["cat", "cats", "catsdogcats", "dog", "dogcatsdog", "hippopotamuses", "rat", "ratcatdogcat"]
    
    start_time = time.time()
    concatenated = concat.find_all_concatenated_words(words)
    time_taken = time.time() - start_time
    print(f"\nConcatenated Words: {concatenated} ({time_taken:.6f}s)")
    
    # Test palindrome pairs
    palindrome = PalindromePairsProblems()
    words_pal = ["abcd", "dcba", "lls", "s", "sssll"]
    
    start_time = time.time()
    pairs = palindrome.palindrome_pairs(words_pal)
    time_taken = time.time() - start_time
    print(f"Palindrome Pairs: {pairs} ({time_taken:.6f}s)")

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Trie + DP Demo ===\n")
    
    # Word Break Problems
    print("1. Word Break Problems:")
    word_break = WordBreakProblems()
    
    s = "leetcode"
    word_dict = ["leet", "code"]
    can_break = word_break.word_break(s, word_dict)
    print(f"  Can break '{s}' with {word_dict}: {can_break}")
    
    all_sentences = word_break.word_break_ii(s, word_dict)
    print(f"  All possible sentences: {all_sentences}")
    
    min_words_check = word_break.word_break_with_concatenation(s, word_dict, 2)
    print(f"  Can break with at least 2 words: {min_words_check}")
    
    word_costs = {"leet": 5, "code": 3}
    min_cost = word_break.word_break_with_costs(s, word_costs)
    print(f"  Minimum cost to break: {min_cost}")
    print()
    
    # Concatenated Words
    print("2. Concatenated Words Problems:")
    concat = ConcatenatedWordsProblems()
    
    words = ["cat", "cats", "catsdogcats", "dog", "dogcatsdog", "hippopotamuses", "rat", "ratcatdogcat"]
    concatenated = concat.find_all_concatenated_words(words)
    print(f"  Words: {words}")
    print(f"  Concatenated words: {concatenated}")
    
    count = concat.count_concatenated_words(words, 2)
    print(f"  Count with at least 2 parts: {count}")
    
    longest = concat.longest_concatenated_word(words)
    print(f"  Longest concatenated word: '{longest}'")
    
    limited = concat.concatenated_words_with_limit(words, 3)
    print(f"  With at most 3 parts: {limited}")
    print()
    
    # Palindrome Pairs
    print("3. Palindrome Pairs Problems:")
    palindrome = PalindromePairsProblems()
    
    words_pal = ["abcd", "dcba", "lls", "s", "sssll"]
    pairs = palindrome.palindrome_pairs(words_pal)
    print(f"  Words: {words_pal}")
    print(f"  Palindrome pairs: {pairs}")
    
    for i, j in pairs:
        palindrome_str = words_pal[i] + words_pal[j]
        print(f"    {i},{j}: '{words_pal[i]}' + '{words_pal[j]}' = '{palindrome_str}'")
    
    count_pairs = palindrome.count_palindrome_pairs(words_pal)
    print(f"  Total palindrome pairs: {count_pairs}")
    
    i, j, longest_pal = palindrome.longest_palindrome_pair(words_pal)
    if i >= 0:
        print(f"  Longest palindrome: indices ({i},{j}) = '{longest_pal}'")
    
    constrained = palindrome.palindrome_pairs_with_constraints(words_pal, 4, 8)
    print(f"  Pairs with length 4-8: {constrained}")
    print()
    
    # Performance comparison
    performance_comparison()
    
    # Pattern Recognition Guide
    print("\n=== Trie + DP Pattern Recognition ===")
    print("When to Use Trie + DP:")
    print("  1. Multiple string matching against dictionary")
    print("  2. Prefix-based search with DP optimization")
    print("  3. Word segmentation problems")
    print("  4. String concatenation/composition problems")
    
    print("\nCommon Patterns:")
    print("  1. Build trie from dictionary/word list")
    print("  2. Use DP array to track segmentation possibilities")
    print("  3. Traverse trie while iterating through string")
    print("  4. Mark valid positions in DP array")
    
    print("\nTrie + DP Benefits:")
    print("  1. Efficient prefix matching O(k) vs O(n*k)")
    print("  2. Shared prefix optimization")
    print("  3. Early termination when no prefix matches")
    print("  4. Memory efficient for large dictionaries")
    
    print("\nOptimization Techniques:")
    print("  1. Reverse trie for suffix matching")
    print("  2. Palindrome suffix precomputation")
    print("  3. Length-based early pruning")
    print("  4. Memoization for repeated subproblems")
    
    print("\nReal-world Applications:")
    print("  1. Text segmentation and tokenization")
    print("  2. Spell checkers and autocomplete")
    print("  3. Code completion and syntax analysis")
    print("  4. Natural language processing")
    print("  5. String matching in bioinformatics")
    
    print("\n=== Demo Complete ===") 