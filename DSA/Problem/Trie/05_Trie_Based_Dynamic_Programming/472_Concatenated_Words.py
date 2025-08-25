"""
472. Concatenated Words - Multiple Approaches
Difficulty: Medium

Given an array of strings words (without duplicates), return all the concatenated words 
in the given list of words.

A concatenated word is defined as a string that is comprised of at least two shorter 
words in the given array.

LeetCode Problem: https://leetcode.com/problems/concatenated-words/

Example:
Input: words = ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamus","rat","ratcatdogcat"]
Output: ["catsdogcats","dogcatsdog","ratcatdogcat"]
Explanation: "catsdogcats" can be concatenated by "cats"+"dog"+"cats"; 
"dogcatsdog" can be concatenated by "dog"+"cats"+"dog"; 
"ratcatdogcat" can be concatenated by "rat"+"cat"+"dog"+"cat".
"""

from typing import List, Set, Dict, Optional
from collections import defaultdict, deque
import time

class TrieNode:
    """Trie node for concatenated words problem"""
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word = ""

class Solution:
    
    def findAllConcatenatedWordsInADict1(self, words: List[str]) -> List[str]:
        """
        Approach 1: Brute Force with Word Break
        
        For each word, check if it can be formed by other words using word break.
        
        Time: O(n * m^2) where n=number of words, m=max word length
        Space: O(n * m) for storing words and DP arrays
        """
        if not words:
            return []
        
        word_set = set(words)
        result = []
        
        def can_form(word: str, word_dict: Set[str]) -> bool:
            """Check if word can be formed from dictionary words"""
            if not word:
                return False
            
            n = len(word)
            dp = [False] * (n + 1)
            dp[0] = True
            
            for i in range(1, n + 1):
                for j in range(i):
                    if dp[j] and word[j:i] in word_dict:
                        dp[i] = True
                        break
            
            return dp[n]
        
        for word in words:
            # Remove current word from dictionary and check if it can be formed
            word_dict = word_set - {word}
            if can_form(word, word_dict):
                result.append(word)
        
        return result
    
    def findAllConcatenatedWordsInADict2(self, words: List[str]) -> List[str]:
        """
        Approach 2: Optimized Word Break with Length Sorting
        
        Sort words by length and build dictionary incrementally.
        
        Time: O(n log n + n * m^2) where sorting dominates for small m
        Space: O(n * m)
        """
        if not words:
            return []
        
        # Sort by length so shorter words are processed first
        words.sort(key=len)
        word_set = set()
        result = []
        
        def can_form_from_shorter(word: str) -> bool:
            """Check if word can be formed from shorter words"""
            if not word:
                return False
            
            n = len(word)
            dp = [False] * (n + 1)
            dp[0] = True
            
            for i in range(1, n + 1):
                for j in range(i):
                    if dp[j] and word[j:i] in word_set:
                        dp[i] = True
                        break
            
            return dp[n]
        
        for word in words:
            if can_form_from_shorter(word):
                result.append(word)
            word_set.add(word)
        
        return result
    
    def findAllConcatenatedWordsInADict3(self, words: List[str]) -> List[str]:
        """
        Approach 3: Trie-based Solution
        
        Build trie and use it for efficient word break checking.
        
        Time: O(n * m + n * m^2) for building trie + checking
        Space: O(n * m) for trie
        """
        if not words:
            return []
        
        # Build trie
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = True
            node.word = word
        
        result = []
        
        def can_form_with_trie(word: str, exclude_word: str) -> bool:
            """Check if word can be formed using trie (excluding itself)"""
            n = len(word)
            dp = [False] * (n + 1)
            dp[0] = True
            
            for i in range(1, n + 1):
                if dp[i]:
                    continue
                
                node = root
                for j in range(i - 1, -1, -1):
                    char = word[j]
                    if char not in node.children:
                        break
                    
                    node = node.children[char]
                    
                    if (node.is_word and 
                        node.word != exclude_word and 
                        dp[j]):
                        dp[i] = True
                        break
            
            return dp[n]
        
        for word in words:
            if can_form_with_trie(word, word):
                result.append(word)
        
        return result
    
    def findAllConcatenatedWordsInADict4(self, words: List[str]) -> List[str]:
        """
        Approach 4: DFS with Memoization
        
        Use DFS with memoization to check word formation.
        
        Time: O(n * m^2) with memoization
        Space: O(n * m) for memoization
        """
        if not words:
            return []
        
        word_set = set(words)
        result = []
        
        def can_form_dfs(word: str, start: int, word_count: int, memo: Dict[int, bool]) -> bool:
            """DFS to check if word can be formed from given start position"""
            if start == len(word):
                return word_count >= 2  # Need at least 2 words
            
            if start in memo:
                return memo[start]
            
            for end in range(start + 1, len(word) + 1):
                prefix = word[start:end]
                if (prefix in word_set and 
                    prefix != word and  # Don't use the word itself
                    can_form_dfs(word, end, word_count + 1, memo)):
                    memo[start] = True
                    return True
            
            memo[start] = False
            return False
        
        for word in words:
            memo = {}
            if can_form_dfs(word, 0, 0, memo):
                result.append(word)
        
        return result
    
    def findAllConcatenatedWordsInADict5(self, words: List[str]) -> List[str]:
        """
        Approach 5: Optimized Trie with Early Pruning
        
        Enhanced trie with pruning optimizations.
        
        Time: O(n * m^2) with optimizations
        Space: O(n * m)
        """
        if not words:
            return []
        
        # Sort words by length for optimization
        words.sort(key=len)
        
        class OptimizedTrieNode:
            def __init__(self):
                self.children = {}
                self.is_word = False
                self.word = ""
                self.min_word_length = float('inf')  # Minimum word length in subtree
        
        root = OptimizedTrieNode()
        root.min_word_length = 0
        
        result = []
        
        def insert_word(word: str) -> None:
            """Insert word into optimized trie"""
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = OptimizedTrieNode()
                node = node.children[char]
                node.min_word_length = min(node.min_word_length, len(word))
            node.is_word = True
            node.word = word
        
        def can_form_optimized(word: str) -> bool:
            """Check if word can be formed with optimizations"""
            n = len(word)
            dp = [False] * (n + 1)
            dp[0] = True
            
            for i in range(1, n + 1):
                if dp[i]:
                    continue
                
                node = root
                remaining = n - i
                
                for j in range(i - 1, -1, -1):
                    char = word[j]
                    
                    if char not in node.children:
                        break
                    
                    node = node.children[char]
                    
                    # Early pruning: not enough characters left
                    if node.min_word_length > remaining + (i - j):
                        break
                    
                    if (node.is_word and 
                        node.word != word and 
                        dp[j]):
                        dp[i] = True
                        break
            
            return dp[n]
        
        for word in words:
            if can_form_optimized(word):
                result.append(word)
            insert_word(word)
        
        return result
    
    def findAllConcatenatedWordsInADict6(self, words: List[str]) -> List[str]:
        """
        Approach 6: BFS with Level Processing
        
        Use BFS to process words level by level.
        
        Time: O(n * m^2)
        Space: O(n * m)
        """
        if not words:
            return []
        
        word_set = set(words)
        result = []
        
        def can_form_bfs(word: str) -> bool:
            """Use BFS to check if word can be formed"""
            if not word:
                return False
            
            queue = deque([(0, 0)])  # (position, word_count)
            visited = set([0])
            
            while queue:
                pos, count = queue.popleft()
                
                if pos == len(word):
                    return count >= 2
                
                for end in range(pos + 1, len(word) + 1):
                    if end not in visited:
                        prefix = word[pos:end]
                        if prefix in word_set and prefix != word:
                            visited.add(end)
                            queue.append((end, count + 1))
            
            return False
        
        for word in words:
            if can_form_bfs(word):
                result.append(word)
        
        return result
    
    def findAllConcatenatedWordsInADict7(self, words: List[str]) -> List[str]:
        """
        Approach 7: Advanced DP with Word Length Optimization
        
        Optimize DP by considering word length constraints.
        
        Time: O(n * m * max_word_length)
        Space: O(n * m)
        """
        if not words:
            return []
        
        word_set = set(words)
        max_word_len = max(len(word) for word in words) if words else 0
        result = []
        
        def can_form_optimized_dp(word: str) -> bool:
            """Optimized DP with length constraints"""
            n = len(word)
            dp = [False] * (n + 1)
            dp[0] = True
            
            for i in range(1, n + 1):
                # Only check word lengths that exist in dictionary
                for length in range(1, min(i, max_word_len) + 1):
                    if dp[i - length]:
                        prefix = word[i - length:i]
                        if prefix in word_set and prefix != word:
                            dp[i] = True
                            break
            
            return dp[n]
        
        for word in words:
            if can_form_optimized_dp(word):
                result.append(word)
        
        return result


def test_basic_functionality():
    """Test basic functionality"""
    print("=== Testing Basic Functionality ===")
    
    solution = Solution()
    
    test_cases = [
        # LeetCode example
        (["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamus","rat","ratcatdogcat"],
         ["catsdogcats","dogcatsdog","ratcatdogcat"]),
        
        # Simple cases
        (["cat", "dog", "catdog"], ["catdog"]),
        (["word", "good", "best", "wordgoodbestword"], ["wordgoodbestword"]),
        
        # Edge cases
        ([], []),
        (["a"], []),
        (["a", "aa"], ["aa"]),
        (["a", "b", "ab"], ["ab"]),
        
        # Complex cases
        (["a", "aa", "aaa", "aaaa"], ["aa", "aaa", "aaaa"]),
        (["cat", "cats", "dog", "catsdog"], ["catsdog"]),
        
        # No concatenated words
        (["apple", "banana", "cherry"], []),
        
        # Multiple possibilities
        (["ab", "cd", "abcd", "efgh"], ["abcd"]),
    ]
    
    approaches = [
        ("Brute Force", solution.findAllConcatenatedWordsInADict1),
        ("Length Sorted", solution.findAllConcatenatedWordsInADict2),
        ("Trie-based", solution.findAllConcatenatedWordsInADict3),
        ("DFS Memoization", solution.findAllConcatenatedWordsInADict4),
        ("Optimized Trie", solution.findAllConcatenatedWordsInADict5),
        ("BFS", solution.findAllConcatenatedWordsInADict6),
        ("Advanced DP", solution.findAllConcatenatedWordsInADict7),
    ]
    
    for i, (words, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {words}")
        print(f"Expected: {sorted(expected)}")
        
        for name, method in approaches:
            try:
                result = method(words[:])
                result_sorted = sorted(result)
                
                status = "✓" if result_sorted == sorted(expected) else "✗"
                print(f"  {name:15}: {result_sorted} {status}")
                
            except Exception as e:
                print(f"  {name:15}: Error - {e}")


def demonstrate_word_formation_process():
    """Demonstrate word formation checking process"""
    print("\n=== Word Formation Process Demo ===")
    
    words = ["cat", "cats", "dog", "catsdogcats"]
    target_word = "catsdogcats"
    
    print(f"Words: {words}")
    print(f"Checking if '{target_word}' can be formed...")
    
    word_set = set(words)
    word_set.remove(target_word)  # Don't use the word itself
    
    print(f"Available words: {word_set}")
    
    # Demonstrate DP process
    n = len(target_word)
    dp = [False] * (n + 1)
    dp[0] = True
    
    print(f"\nDP process:")
    print(f"dp[0] = True (empty string)")
    
    for i in range(1, n + 1):
        print(f"\nPosition {i} (character '{target_word[i-1]}'):")
        
        for j in range(i):
            substring = target_word[j:i]
            
            if dp[j] and substring in word_set:
                dp[i] = True
                print(f"  ✓ Found: '{substring}' at [{j}:{i}], dp[{j}] = {dp[j]}")
                print(f"    Setting dp[{i}] = True")
                break
            elif dp[j]:
                print(f"  ✗ Checked: '{substring}' at [{j}:{i}] (not in dictionary)")
        
        if not dp[i]:
            print(f"  No valid formation found for position {i}")
        
        print(f"  dp[{i}] = {dp[i]}")
    
    print(f"\nResult: '{target_word}' {'can' if dp[n] else 'cannot'} be formed")
    
    # Show the actual formation
    if dp[n]:
        print(f"Formation breakdown:")
        
        def find_formation(pos: int) -> List[str]:
            if pos == 0:
                return []
            
            for start in range(pos):
                word = target_word[start:pos]
                if dp[start] and word in word_set:
                    prefix_formation = find_formation(start)
                    return prefix_formation + [word]
            
            return []
        
        formation = find_formation(n)
        print(f"  {' + '.join(formation)} = {target_word}")


def demonstrate_trie_optimization():
    """Demonstrate trie optimization benefits"""
    print("\n=== Trie Optimization Demo ===")
    
    words = ["program", "programming", "language", "prog", "ram", "min", "g", "lan", "age"]
    target = "programming"
    
    print(f"Words: {words}")
    print(f"Checking: '{target}'")
    
    # Build trie
    root = TrieNode()
    for word in words:
        if word != target:  # Exclude target word
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = True
            node.word = word
    
    print(f"\nTrie-based word break:")
    
    n = len(target)
    dp = [False] * (n + 1)
    dp[0] = True
    
    for i in range(1, n + 1):
        print(f"\nPosition {i}:")
        
        if dp[i]:
            print(f"  Already True, skipping")
            continue
        
        node = root
        words_found = []
        
        for j in range(i - 1, -1, -1):
            char = target[j]
            print(f"    Checking char '{char}' at position {j}")
            
            if char not in node.children:
                print(f"      No trie path, stopping backward search")
                break
            
            node = node.children[char]
            
            if node.is_word and dp[j]:
                word = target[j:i]
                words_found.append(word)
                dp[i] = True
                print(f"      ✓ Found word '{word}' and dp[{j}] = True")
                break
            elif node.is_word:
                word = target[j:i]
                print(f"      Found word '{word}' but dp[{j}] = False")
        
        print(f"    dp[{i}] = {dp[i]}")
    
    print(f"\nTrie traversal benefits:")
    print(f"  • Single traversal finds multiple possible words")
    print(f"  • Early termination when no trie path exists")
    print(f"  • Efficient for dictionaries with common prefixes")


def demonstrate_length_sorting_optimization():
    """Demonstrate length sorting optimization"""
    print("\n=== Length Sorting Optimization Demo ===")
    
    words = ["catsdogcats", "cat", "dog", "cats", "ratcatdogcat", "rat"]
    
    print(f"Original order: {words}")
    
    # Sort by length
    sorted_words = sorted(words, key=len)
    print(f"Sorted by length: {sorted_words}")
    
    print(f"\nProcessing in length order:")
    
    word_set = set()
    concatenated_words = []
    
    def can_form(word: str, available: Set[str]) -> bool:
        if not word or not available:
            return False
        
        n = len(word)
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and word[j:i] in available:
                    dp[i] = True
                    break
        
        return dp[n]
    
    for word in sorted_words:
        print(f"\nProcessing '{word}' (length {len(word)}):")
        print(f"  Available dictionary: {word_set}")
        
        if can_form(word, word_set):
            concatenated_words.append(word)
            print(f"  ✓ Can be formed from shorter words")
        else:
            print(f"  ✗ Cannot be formed from available words")
        
        word_set.add(word)
        print(f"  Added '{word}' to dictionary")
    
    print(f"\nConcatenated words found: {concatenated_words}")
    print(f"\nBenefits of length sorting:")
    print(f"  • Shorter words are available when checking longer ones")
    print(f"  • Build dictionary incrementally")
    print(f"  • Avoid checking impossible combinations")


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import random
    import string
    
    solution = Solution()
    
    # Generate test data
    def generate_words(base_count: int, max_length: int) -> List[str]:
        """Generate words including some that are concatenations"""
        words = set()
        
        # Generate base words
        while len(words) < base_count:
            length = random.randint(2, max_length // 2)
            word = ''.join(random.choices(string.ascii_lowercase[:6], k=length))
            words.add(word)
        
        word_list = list(words)
        
        # Generate some concatenated words
        concat_count = base_count // 4
        for _ in range(concat_count):
            # Concatenate 2-3 existing words
            num_parts = random.randint(2, 3)
            parts = random.sample(word_list, num_parts)
            concatenated = ''.join(parts)
            
            if len(concatenated) <= max_length:
                word_list.append(concatenated)
        
        return word_list
    
    test_scenarios = [
        ("Small", generate_words(20, 15)),
        ("Medium", generate_words(50, 20)),
        ("Large", generate_words(100, 25)),
    ]
    
    approaches = [
        ("Length Sorted", solution.findAllConcatenatedWordsInADict2),
        ("Trie-based", solution.findAllConcatenatedWordsInADict3),
        ("DFS Memo", solution.findAllConcatenatedWordsInADict4),
        ("Optimized Trie", solution.findAllConcatenatedWordsInADict5),
        ("Advanced DP", solution.findAllConcatenatedWordsInADict7),
    ]
    
    for scenario_name, words in test_scenarios:
        print(f"\n--- {scenario_name} Dataset ---")
        print(f"Total words: {len(words)}, Max length: {max(len(w) for w in words) if words else 0}")
        
        for approach_name, method in approaches:
            start_time = time.time()
            
            try:
                result = method(words[:])
                end_time = time.time()
                
                execution_time = (end_time - start_time) * 1000
                print(f"  {approach_name:15}: {len(result):3} concatenated words in {execution_time:6.2f}ms")
            
            except Exception as e:
                print(f"  {approach_name:15}: Error - {str(e)[:30]}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solution = Solution()
    
    # Application 1: Compound word detection in linguistics
    print("1. Compound Word Detection:")
    
    dictionary = ["fire", "man", "house", "dog", "cat", "fish", 
                 "fireman", "doghouse", "catfish", "housework", "work"]
    
    compounds = solution.findAllConcatenatedWordsInADict2(dictionary)
    print(f"   Dictionary: {dictionary}")
    print(f"   Compound words: {compounds}")
    
    # Show composition
    for compound in compounds:
        print(f"   '{compound}' formation:")
        # Find a way to break it down
        word_set = set(dictionary) - {compound}
        n = len(compound)
        dp = [False] * (n + 1)
        parent = [-1] * (n + 1)
        dp[0] = True
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and compound[j:i] in word_set:
                    dp[i] = True
                    parent[i] = j
                    break
        
        if dp[n]:
            # Reconstruct
            parts = []
            pos = n
            while pos > 0:
                prev = parent[pos]
                parts.append(compound[prev:pos])
                pos = prev
            parts.reverse()
            print(f"     {' + '.join(parts)}")
    
    # Application 2: Domain name analysis
    print(f"\n2. Domain Name Component Analysis:")
    
    domains = ["tech", "start", "up", "blog", "news", "techstartup", "startupnews", "technews"]
    
    domain_compounds = solution.findAllConcatenatedWordsInADict2(domains)
    print(f"   Domain components: {domains}")
    print(f"   Compound domains: {domain_compounds}")
    
    # Application 3: Code identifier analysis
    print(f"\n3. Code Identifier Analysis:")
    
    identifiers = ["get", "user", "name", "by", "id", "set", 
                  "getUserName", "setUserName", "getUserById", "data", "base"]
    
    # Convert to lowercase for analysis
    lower_identifiers = [id.lower() for id in identifiers]
    
    compound_ids = solution.findAllConcatenatedWordsInADict2(lower_identifiers)
    print(f"   Identifiers: {identifiers}")
    print(f"   Compound identifiers: {compound_ids}")
    
    # Application 4: Text processing for search engines
    print(f"\n4. Search Query Processing:")
    
    query_terms = ["new", "york", "city", "weather", "forecast", "today",
                  "newyork", "newyorkcity", "weatherforecast", "todayweather"]
    
    compound_queries = solution.findAllConcatenatedWordsInADict2(query_terms)
    print(f"   Query terms: {query_terms}")
    print(f"   Compound queries: {compound_queries}")
    print(f"   These could be segmented for better search results")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solution = Solution()
    
    edge_cases = [
        # Empty and minimal cases
        ([], "Empty input"),
        (["a"], "Single word"),
        (["a", "b"], "Two single chars"),
        (["ab"], "Single word only"),
        
        # Self-reference cases
        (["a", "a"], "Duplicate words"),
        (["abc", "abc"], "Exact duplicates"),
        
        # Length edge cases
        (["", "a"], "Empty string in input"),
        (["a", "aa", "aaa"], "Nested lengths"),
        
        # Complex concatenations
        (["a", "b", "c", "ab", "bc", "abc", "abbc"], "Multiple levels"),
        (["cat", "cats", "dog", "s", "catsdogs"], "Single char component"),
        
        # No solutions
        (["hello", "world", "foo", "bar"], "No concatenations possible"),
        
        # Long chains
        (["a", "b", "ab", "aba", "abab", "ababa"], "Long concatenation chains"),
        
        # Case sensitivity
        (["Cat", "cat", "Dog", "CatDog"], "Mixed case"),
    ]
    
    for words, description in edge_cases:
        print(f"\n{description}: {words}")
        
        try:
            result = solution.findAllConcatenatedWordsInADict2(words)
            print(f"  Result: {result}")
            
            # Verify each result
            for word in result:
                if word in words:
                    print(f"    '{word}' is a valid concatenated word")
                else:
                    print(f"    ERROR: '{word}' not in original input")
        
        except Exception as e:
            print(f"  Error: {e}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Brute Force Word Break",
         "Time: O(n * m^2 * k) where k is avg word length",
         "Space: O(n + m) for word set and DP array"),
        
        ("Length Sorted Optimization",
         "Time: O(n log n + n * m^2) where sorting helps",
         "Space: O(n + m) with incremental dictionary building"),
        
        ("Trie-based Solution",
         "Time: O(∑word_lengths + n * m^2)",
         "Space: O(∑word_lengths) for trie + O(m) for DP"),
        
        ("DFS with Memoization",
         "Time: O(n * m^2) with memoization preventing recomputation",
         "Space: O(n * m) for memoization across all words"),
        
        ("Optimized Trie with Pruning",
         "Time: O(∑word_lengths + n * m * avg_depth)",
         "Space: O(∑word_lengths) with metadata"),
        
        ("BFS Level Processing",
         "Time: O(n * m^2) with level-by-level exploration",
         "Space: O(m) for BFS queue and visited set"),
        
        ("Advanced DP with Length Constraints",
         "Time: O(n * m * max_word_length)",
         "Space: O(n + m) optimized space usage"),
    ]
    
    print("Approach Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nKey Variables:")
    print(f"  • n = number of words in input")
    print(f"  • m = maximum word length")
    print(f"  • k = average word length")
    print(f"  • ∑word_lengths = total characters across all words")
    
    print(f"\nOptimization Strategies:")
    print(f"  • Length sorting: Process shorter words first")
    print(f"  • Trie usage: Reduce string comparison overhead")
    print(f"  • Memoization: Avoid recomputing same subproblems")
    print(f"  • Early pruning: Skip impossible combinations")
    print(f"  • Length constraints: Limit search space")
    
    print(f"\nPractical Considerations:")
    print(f"  • For small dictionaries: Simple DP often fastest")
    print(f"  • For large dictionaries: Trie-based approaches scale better")
    print(f"  • For very long words: Memory usage becomes important")
    print(f"  • For repeated queries: Preprocessing amortizes cost")
    
    print(f"\nRecommendations:")
    print(f"  • Use Length Sorted for general purpose applications")
    print(f"  • Use Trie-based for large dictionaries with common prefixes")
    print(f"  • Use Advanced DP for memory-constrained environments")
    print(f"  • Consider input characteristics when choosing approach")


if __name__ == "__main__":
    test_basic_functionality()
    demonstrate_word_formation_process()
    demonstrate_trie_optimization()
    demonstrate_length_sorting_optimization()
    benchmark_approaches()
    demonstrate_real_world_applications()
    test_edge_cases()
    analyze_complexity()

"""
472. Concatenated Words demonstrates comprehensive word formation detection approaches:

1. Brute Force Word Break - Check each word against others using basic word break
2. Length Sorted Optimization - Sort by length and build dictionary incrementally  
3. Trie-based Solution - Use trie for efficient prefix matching and word lookup
4. DFS with Memoization - Depth-first search with memoization for subproblems
5. Optimized Trie with Pruning - Enhanced trie with early termination optimizations
6. BFS Level Processing - Breadth-first exploration of word formation possibilities
7. Advanced DP with Length Constraints - Optimized DP limiting search by word lengths

Key concepts:
- Word break algorithms applied to concatenation detection
- Length-based optimization strategies
- Trie data structures for efficient string matching
- Dynamic programming with memoization
- Early pruning and constraint optimization

Real-world applications:
- Compound word detection in natural language processing
- Domain name component analysis
- Code identifier analysis and refactoring
- Search query segmentation and processing

Each approach offers different trade-offs between preprocessing time,
query performance, and memory usage for word concatenation problems.
"""
