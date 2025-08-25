"""
140. Word Break II - Multiple Approaches
Difficulty: Medium

Given a string s and a dictionary of strings wordDict, add spaces in s to construct 
a sentence where each word is a valid dictionary word. Return all such possible sentences 
in any order.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

LeetCode Problem: https://leetcode.com/problems/word-break-ii/

Example:
Input: s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
Output: ["cats and dog","cat sand dog"]
"""

from typing import List, Set, Dict, Optional
from collections import defaultdict, deque
import time

class TrieNode:
    """Trie node for word break II problem"""
    def __init__(self):
        self.children = {}
        self.is_word = False

class Solution:
    
    def wordBreak1(self, s: str, wordDict: List[str]) -> List[str]:
        """
        Approach 1: Backtracking with Pruning
        
        Use backtracking to find all possible sentence constructions.
        
        Time: O(2^n) worst case, better with pruning
        Space: O(n) recursion depth + O(result size)
        """
        word_set = set(wordDict)
        
        # First check if word break is possible
        def can_break(s: str) -> bool:
            n = len(s)
            dp = [False] * (n + 1)
            dp[0] = True
            
            for i in range(1, n + 1):
                for j in range(i):
                    if dp[j] and s[j:i] in word_set:
                        dp[i] = True
                        break
            
            return dp[n]
        
        if not can_break(s):
            return []
        
        def backtrack(start: int, path: List[str]) -> List[str]:
            if start == len(s):
                return [' '.join(path)]
            
            results = []
            for end in range(start + 1, len(s) + 1):
                word = s[start:end]
                if word in word_set:
                    path.append(word)
                    results.extend(backtrack(end, path))
                    path.pop()
            
            return results
        
        return backtrack(0, [])
    
    def wordBreak2(self, s: str, wordDict: List[str]) -> List[str]:
        """
        Approach 2: Memoized Recursion
        
        Add memoization to avoid recomputing the same subproblems.
        
        Time: O(n^2 + result_size)
        Space: O(n^2) for memoization + O(result_size)
        """
        word_set = set(wordDict)
        memo = {}
        
        def word_break_helper(start: int) -> List[str]:
            if start == len(s):
                return [""]
            
            if start in memo:
                return memo[start]
            
            results = []
            for end in range(start + 1, len(s) + 1):
                word = s[start:end]
                if word in word_set:
                    suffixes = word_break_helper(end)
                    for suffix in suffixes:
                        if suffix:
                            results.append(word + " " + suffix)
                        else:
                            results.append(word)
            
            memo[start] = results
            return results
        
        return word_break_helper(0)
    
    def wordBreak3(self, s: str, wordDict: List[str]) -> List[str]:
        """
        Approach 3: Trie + Memoized Recursion
        
        Use trie for efficient word lookup with memoization.
        
        Time: O(n^2 + result_size)
        Space: O(trie_size + n^2) for memoization
        """
        # Build trie
        root = TrieNode()
        for word in wordDict:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = True
        
        memo = {}
        
        def word_break_helper(start: int) -> List[str]:
            if start == len(s):
                return [""]
            
            if start in memo:
                return memo[start]
            
            results = []
            node = root
            
            for end in range(start, len(s)):
                char = s[end]
                if char not in node.children:
                    break
                
                node = node.children[char]
                
                if node.is_word:
                    word = s[start:end + 1]
                    suffixes = word_break_helper(end + 1)
                    
                    for suffix in suffixes:
                        if suffix:
                            results.append(word + " " + suffix)
                        else:
                            results.append(word)
            
            memo[start] = results
            return results
        
        return word_break_helper(0)
    
    def wordBreak4(self, s: str, wordDict: List[str]) -> List[str]:
        """
        Approach 4: Dynamic Programming + Reconstruction
        
        Use DP to build valid breakpoints, then reconstruct all solutions.
        
        Time: O(n^2 + result_size * sentence_length)
        Space: O(n^2) for DP table
        """
        word_set = set(wordDict)
        n = len(s)
        
        # DP to check if word break is possible
        dp = [False] * (n + 1)
        dp[0] = True
        
        # Store valid breakpoints
        valid_breaks = [[] for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    valid_breaks[i].append(j)
        
        if not dp[n]:
            return []
        
        # Reconstruct all valid sentences
        def reconstruct(pos: int) -> List[str]:
            if pos == 0:
                return [""]
            
            results = []
            for prev_pos in valid_breaks[pos]:
                word = s[prev_pos:pos]
                prev_sentences = reconstruct(prev_pos)
                
                for sentence in prev_sentences:
                    if sentence:
                        results.append(sentence + " " + word)
                    else:
                        results.append(word)
            
            return results
        
        return reconstruct(n)
    
    def wordBreak5(self, s: str, wordDict: List[str]) -> List[str]:
        """
        Approach 5: BFS with Path Tracking
        
        Use BFS to explore all possible breakdowns level by level.
        
        Time: O(2^n) worst case
        Space: O(2^n) for storing all paths
        """
        word_set = set(wordDict)
        
        # BFS with path tracking
        queue = deque([(0, [])])  # (position, path)
        results = []
        visited = {}  # position -> list of paths reaching this position
        
        while queue:
            pos, path = queue.popleft()
            
            if pos == len(s):
                results.append(' '.join(path))
                continue
            
            # Avoid processing same position with same path multiple times
            if pos in visited:
                # Only process if this path is new
                path_key = tuple(path)
                if path_key in visited[pos]:
                    continue
                visited[pos].add(path_key)
            else:
                visited[pos] = {tuple(path)}
            
            for end in range(pos + 1, len(s) + 1):
                word = s[pos:end]
                if word in word_set:
                    new_path = path + [word]
                    queue.append((end, new_path))
        
        return results
    
    def wordBreak6(self, s: str, wordDict: List[str]) -> List[str]:
        """
        Approach 6: Optimized Backtracking with Early Pruning
        
        Enhanced backtracking with multiple pruning strategies.
        
        Time: O(result_size * sentence_length)
        Space: O(n) recursion + O(result_size)
        """
        word_set = set(wordDict)
        max_word_len = max(len(word) for word in wordDict) if wordDict else 0
        
        # Precompute which positions can lead to valid solutions
        n = len(s)
        can_break = [False] * (n + 1)
        can_break[n] = True
        
        for i in range(n - 1, -1, -1):
            for length in range(1, min(max_word_len, n - i) + 1):
                if s[i:i + length] in word_set and can_break[i + length]:
                    can_break[i] = True
                    break
        
        if not can_break[0]:
            return []
        
        def backtrack(start: int, path: List[str]) -> List[str]:
            if start == len(s):
                return [' '.join(path)]
            
            if not can_break[start]:
                return []
            
            results = []
            for length in range(1, min(max_word_len, len(s) - start) + 1):
                word = s[start:start + length]
                if word in word_set and can_break[start + length]:
                    path.append(word)
                    results.extend(backtrack(start + length, path))
                    path.pop()
            
            return results
        
        return backtrack(0, [])
    
    def wordBreak7(self, s: str, wordDict: List[str]) -> List[str]:
        """
        Approach 7: Advanced Trie with Path Compression
        
        Use compressed trie with optimized path reconstruction.
        
        Time: O(n^2 + result_size)
        Space: O(trie_size + result_size)
        """
        class AdvancedTrieNode:
            def __init__(self):
                self.children = {}
                self.is_word = False
                self.words = []  # All words ending at this node
        
        # Build advanced trie
        root = AdvancedTrieNode()
        for word in wordDict:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = AdvancedTrieNode()
                node = node.children[char]
            node.is_word = True
            node.words.append(word)
        
        # Memoized recursion with advanced trie
        memo = {}
        
        def find_words_at_position(start: int) -> List[str]:
            """Find all words that can start at given position"""
            if start >= len(s):
                return []
            
            words = []
            node = root
            
            for end in range(start, len(s)):
                char = s[end]
                if char not in node.children:
                    break
                
                node = node.children[char]
                if node.is_word:
                    words.extend(node.words)
            
            return words
        
        def word_break_helper(start: int) -> List[str]:
            if start == len(s):
                return [""]
            
            if start in memo:
                return memo[start]
            
            results = []
            node = root
            
            for end in range(start, len(s)):
                char = s[end]
                if char not in node.children:
                    break
                
                node = node.children[char]
                
                if node.is_word:
                    for word in node.words:
                        suffixes = word_break_helper(end + 1)
                        for suffix in suffixes:
                            if suffix:
                                results.append(word + " " + suffix)
                            else:
                                results.append(word)
            
            memo[start] = results
            return results
        
        return word_break_helper(0)


def test_basic_functionality():
    """Test basic functionality"""
    print("=== Testing Basic Functionality ===")
    
    solution = Solution()
    
    test_cases = [
        # LeetCode examples
        ("catsanddog", ["cat","cats","and","sand","dog"], ["cats and dog","cat sand dog"]),
        ("pineapplepenapple", ["apple","pen","applepen","pine","pineapple"], 
         ["pine apple pen apple","pineapple pen apple","pine applepen apple"]),
        ("catsandog", ["cats","dog","sand","and","cat"], []),
        
        # Edge cases
        ("", [], [""]),
        ("a", ["a"], ["a"]),
        ("ab", ["a", "b"], ["a b"]),
        
        # Multiple solutions
        ("aaaaaaa", ["aaaa","aaa","aa"], ["aa aa aaa", "aa aaa aa", "aaa aa aa", "aaaa aaa"]),
        
        # No solution
        ("abcd", ["ab", "cd"], []),
        
        # Overlapping words
        ("wordwordword", ["word","wordword"], ["word word word", "word wordword", "wordword word"]),
    ]
    
    approaches = [
        ("Backtracking", solution.wordBreak1),
        ("Memoized Recursion", solution.wordBreak2),
        ("Trie + Memoization", solution.wordBreak3),
        ("DP + Reconstruction", solution.wordBreak4),
        ("BFS Path Tracking", solution.wordBreak5),
        ("Optimized Backtracking", solution.wordBreak6),
        ("Advanced Trie", solution.wordBreak7),
    ]
    
    for i, (s, wordDict, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: s='{s}', words={wordDict}")
        print(f"Expected: {expected}")
        
        for name, method in approaches:
            try:
                result = method(s, wordDict[:])
                
                # Sort results for comparison
                result_sorted = sorted(result)
                expected_sorted = sorted(expected)
                
                status = "✓" if result_sorted == expected_sorted else "✗"
                print(f"  {name:20}: {len(result)} solutions {status}")
                
                if len(result) <= 5:  # Show results if not too many
                    print(f"    {result}")
                
            except Exception as e:
                print(f"  {name:20}: Error - {e}")


def demonstrate_backtracking_process():
    """Demonstrate backtracking process step by step"""
    print("\n=== Backtracking Process Demo ===")
    
    s = "catsand"
    wordDict = ["cat", "cats", "and", "sand"]
    
    print(f"String: '{s}'")
    print(f"Dictionary: {wordDict}")
    word_set = set(wordDict)
    
    print(f"\nBacktracking exploration:")
    
    def backtrack_with_trace(start: int, path: List[str], depth: int = 0) -> List[str]:
        indent = "  " * depth
        print(f"{indent}Exploring from position {start}, path: {path}")
        
        if start == len(s):
            result = ' '.join(path)
            print(f"{indent}✓ Found complete solution: '{result}'")
            return [result]
        
        results = []
        found_word = False
        
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            print(f"{indent}  Trying word: '{word}' (positions {start}:{end})")
            
            if word in word_set:
                print(f"{indent}  ✓ '{word}' is in dictionary")
                path.append(word)
                sub_results = backtrack_with_trace(end, path, depth + 1)
                results.extend(sub_results)
                path.pop()
                found_word = True
            else:
                print(f"{indent}  ✗ '{word}' not in dictionary")
        
        if not found_word:
            print(f"{indent}No valid words found from position {start}")
        
        return results
    
    solutions = backtrack_with_trace(0, [])
    print(f"\nFinal solutions: {solutions}")


def demonstrate_memoization_benefit():
    """Demonstrate memoization benefits"""
    print("\n=== Memoization Benefits Demo ===")
    
    s = "aaaaaaaaab"
    wordDict = ["a", "aa", "aaa", "aaaa", "aaaaa", "b"]
    
    print(f"String: '{s}' (length {len(s)})")
    print(f"Dictionary: {wordDict}")
    
    # Count recursive calls without memoization
    call_count_without_memo = [0]
    
    def backtrack_count_calls(start: int) -> List[str]:
        call_count_without_memo[0] += 1
        
        if start == len(s):
            return [""]
        
        results = []
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in set(wordDict):
                suffixes = backtrack_count_calls(end)
                for suffix in suffixes:
                    if suffix:
                        results.append(word + " " + suffix)
                    else:
                        results.append(word)
        
        return results
    
    # Simulate without memoization (but limit calls to avoid timeout)
    max_calls = 1000
    try:
        backtrack_count_calls(0)
    except:
        call_count_without_memo[0] = max_calls  # Approximate
    
    if call_count_without_memo[0] >= max_calls:
        call_count_without_memo[0] = max_calls
        print(f"Without memoization: >{max_calls} recursive calls (stopped early)")
    else:
        print(f"Without memoization: {call_count_without_memo[0]} recursive calls")
    
    # Count with memoization
    memo = {}
    call_count_with_memo = [0]
    
    def memoized_word_break(start: int) -> List[str]:
        call_count_with_memo[0] += 1
        
        if start == len(s):
            return [""]
        
        if start in memo:
            return memo[start]
        
        results = []
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in set(wordDict):
                suffixes = memoized_word_break(end)
                for suffix in suffixes:
                    if suffix:
                        results.append(word + " " + suffix)
                    else:
                        results.append(word)
        
        memo[start] = results
        return results
    
    solutions = memoized_word_break(0)
    print(f"With memoization: {call_count_with_memo[0]} recursive calls")
    print(f"Found {len(solutions)} solutions")
    print(f"Memoization cache size: {len(memo)}")
    
    if call_count_without_memo[0] > 0:
        speedup = call_count_without_memo[0] / call_count_with_memo[0]
        print(f"Speedup factor: {speedup:.1f}x")


def demonstrate_trie_optimization():
    """Demonstrate trie optimization benefits"""
    print("\n=== Trie Optimization Demo ===")
    
    s = "programminglanguage"
    wordDict = ["program", "programming", "language", "lang", "age", "pro", "gram", "min", "g"]
    
    print(f"String: '{s}'")
    print(f"Dictionary: {wordDict}")
    
    # Build trie
    root = TrieNode()
    for word in wordDict:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
    
    print(f"\nTrie traversal from position 0:")
    
    # Show trie traversal process
    def show_trie_traversal(start_pos: int, max_chars: int = 10):
        print(f"  Starting at position {start_pos}: '{s[start_pos:][:max_chars]}...'")
        
        node = root
        words_found = []
        
        for i in range(start_pos, min(start_pos + max_chars, len(s))):
            char = s[i]
            print(f"    Position {i}: checking character '{char}'")
            
            if char not in node.children:
                print(f"      No trie path for '{char}', stopping")
                break
            
            node = node.children[char]
            
            if node.is_word:
                word = s[start_pos:i + 1]
                words_found.append(word)
                print(f"      ✓ Found word: '{word}'")
            else:
                print(f"      Continue in trie...")
        
        print(f"    Words found from position {start_pos}: {words_found}")
        return words_found
    
    # Show traversal from a few positions
    for pos in [0, 4, 11]:
        if pos < len(s):
            show_trie_traversal(pos)
            print()
    
    print("Benefits of trie approach:")
    print("  • Single traversal finds all possible words starting at position")
    print("  • Early termination when no valid path exists")
    print("  • Efficient for dictionaries with common prefixes")


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import random
    import string
    
    solution = Solution()
    
    # Generate test cases with controlled complexity
    def generate_test_case(string_length: int, dict_size: int, avg_word_len: int) -> tuple:
        # Generate dictionary
        words = set()
        while len(words) < dict_size:
            length = max(1, avg_word_len + random.randint(-2, 2))
            word = ''.join(random.choices(string.ascii_lowercase[:4], k=length))
            words.add(word)
        
        word_list = list(words)
        
        # Generate string that might be breakable
        if random.random() < 0.8:  # 80% chance of being breakable
            s = ""
            while len(s) < string_length:
                word = random.choice(word_list)
                if len(s) + len(word) <= string_length:
                    s += word
                else:
                    # Fill remaining with partial word or random chars
                    remaining = string_length - len(s)
                    if remaining > 0:
                        if random.random() < 0.5:
                            s += word[:remaining]
                        else:
                            s += ''.join(random.choices(string.ascii_lowercase[:4], k=remaining))
                    break
        else:
            s = ''.join(random.choices(string.ascii_lowercase[:4], k=string_length))
        
        return s, word_list
    
    test_scenarios = [
        ("Small", 15, 8, 3),
        ("Medium", 25, 12, 4),
        ("Large", 35, 15, 5),
    ]
    
    approaches = [
        ("Memoized Recursion", solution.wordBreak2),
        ("Trie + Memoization", solution.wordBreak3),
        ("DP + Reconstruction", solution.wordBreak4),
        ("Optimized Backtracking", solution.wordBreak6),
    ]
    
    for scenario_name, str_len, dict_size, avg_word_len in test_scenarios:
        s, wordDict = generate_test_case(str_len, dict_size, avg_word_len)
        
        print(f"\n--- {scenario_name} Test Case ---")
        print(f"String length: {len(s)}, Dictionary size: {len(wordDict)}")
        
        for approach_name, method in approaches:
            start_time = time.time()
            
            try:
                result = method(s, wordDict)
                end_time = time.time()
                
                execution_time = (end_time - start_time) * 1000
                print(f"  {approach_name:20}: {len(result):3} solutions in {execution_time:6.2f}ms")
            
            except Exception as e:
                print(f"  {approach_name:20}: Error - {str(e)[:30]}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solution = Solution()
    
    # Application 1: Text segmentation for search
    print("1. Search Query Segmentation:")
    
    query = "newyorkcityweather"
    entities = ["new", "york", "city", "weather", "new york", "new york city", "nyc"]
    
    segmentations = solution.wordBreak2(query, entities)
    print(f"   Query: '{query}'")
    print(f"   Entities: {entities}")
    print(f"   Possible segmentations:")
    for seg in segmentations:
        print(f"     • {seg}")
    
    # Application 2: Domain name suggestions
    print(f"\n2. Domain Name Breakdown:")
    
    domain = "techstartupblog"
    keywords = ["tech", "start", "startup", "up", "blog", "log", "tech startup"]
    
    breakdowns = solution.wordBreak2(domain, keywords)
    print(f"   Domain: '{domain}'")
    print(f"   Keywords: {keywords}")
    print(f"   Possible meanings:")
    for breakdown in breakdowns:
        print(f"     • {breakdown}")
    
    # Application 3: Code identifier decomposition
    print(f"\n3. Code Identifier Decomposition:")
    
    identifier = "getusernamebyid"
    code_words = ["get", "user", "name", "by", "id", "username", "get user", "by id"]
    
    decompositions = solution.wordBreak2(identifier, code_words)
    print(f"   Identifier: '{identifier}'")
    print(f"   Code vocabulary: {code_words}")
    print(f"   Possible decompositions:")
    for decomp in decompositions:
        print(f"     • {decomp}")
    
    # Application 4: Natural language processing
    print(f"\n4. Compound Word Analysis:")
    
    compound = "firetruckdriver"
    components = ["fire", "truck", "driver", "fire truck", "truck driver"]
    
    analyses = solution.wordBreak2(compound, components)
    print(f"   Compound word: '{compound}'")
    print(f"   Components: {components}")
    print(f"   Possible analyses:")
    for analysis in analyses:
        print(f"     • {analysis}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solution = Solution()
    
    edge_cases = [
        # Empty cases
        ("", [], "Empty string and dictionary"),
        ("", ["a"], "Empty string, non-empty dictionary"),
        ("a", [], "Non-empty string, empty dictionary"),
        
        # Single elements
        ("a", ["a"], "Single character match"),
        ("a", ["b"], "Single character no match"),
        
        # Repeated patterns
        ("aaaa", ["a"], "All single character"),
        ("aaaa", ["aa"], "Pairs only"),
        ("aaaa", ["aaa"], "Not evenly divisible"),
        ("aaaa", ["a", "aa", "aaa", "aaaa"], "Multiple options"),
        
        # Overlapping words
        ("abab", ["a", "ab", "aba", "abab"], "Multiple overlapping"),
        
        # Long strings
        ("a" * 20, ["a"], "Very long string"),
        ("abc" * 10, ["abc"], "Repeated pattern"),
        
        # Case sensitivity
        ("ABC", ["abc"], "Case mismatch"),
        ("ABC", ["ABC"], "Case match"),
        
        # Special characters (if applicable)
        ("a-b", ["a", "b", "-"], "With special characters"),
    ]
    
    for s, wordDict, description in edge_cases:
        print(f"\n{description}:")
        print(f"  s = '{s[:20]}{'...' if len(s) > 20 else ''}'")
        print(f"  wordDict = {wordDict}")
        
        try:
            result = solution.wordBreak2(s, wordDict)
            print(f"  Result: {len(result)} solutions")
            
            if len(result) <= 3:
                for sol in result:
                    print(f"    '{sol}'")
            elif len(result) > 3:
                print(f"    First 3: {result[:3]}")
        
        except Exception as e:
            print(f"  Error: {e}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Backtracking (no pruning)",
         "Time: O(2^n) - exponential exploration",
         "Space: O(n) recursion + O(result_size)"),
        
        ("Memoized Recursion",
         "Time: O(n^2 + result_size * avg_sentence_length)",
         "Space: O(n^2) memoization + O(result_size)"),
        
        ("Trie + Memoization",
         "Time: O(n^2 + W + result_size) where W = total word chars",
         "Space: O(W) trie + O(n^2) memo + O(result_size)"),
        
        ("DP + Reconstruction",
         "Time: O(n^2) DP + O(result_size * avg_sentence_length)",
         "Space: O(n^2) for DP table and breakpoints"),
        
        ("BFS Path Tracking",
         "Time: O(2^n) worst case",
         "Space: O(2^n) for storing all partial paths"),
        
        ("Optimized Backtracking",
         "Time: O(result_size * avg_sentence_length) with pruning",
         "Space: O(n) recursion + O(result_size)"),
        
        ("Advanced Trie",
         "Time: O(n^2 + W + result_size)",
         "Space: O(W) enhanced trie + O(n^2) memo"),
    ]
    
    print("Approach Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nKey Insights:")
    print(f"  • Result size can be exponential (2^n worst case)")
    print(f"  • Memoization is crucial for avoiding recomputation")
    print(f"  • Trie reduces string comparison overhead")
    print(f"  • Early pruning significantly improves practical performance")
    
    print(f"\nOptimization Strategies:")
    print(f"  • Pre-check if any solution exists (Word Break I)")
    print(f"  • Use trie for efficient prefix matching")
    print(f"  • Memoize intermediate results")
    print(f"  • Prune impossible branches early")
    print(f"  • Limit word length checks to maximum dictionary word length")
    
    print(f"\nRecommendations:")
    print(f"  • Use Memoized Recursion for general cases")
    print(f"  • Use Trie + Memoization for large dictionaries")
    print(f"  • Use Optimized Backtracking when result size matters")
    print(f"  • Always check if solution exists before generating all solutions")


if __name__ == "__main__":
    test_basic_functionality()
    demonstrate_backtracking_process()
    demonstrate_memoization_benefit()
    demonstrate_trie_optimization()
    benchmark_approaches()
    demonstrate_real_world_applications()
    test_edge_cases()
    analyze_complexity()

"""
140. Word Break II demonstrates comprehensive sentence reconstruction approaches:

1. Backtracking with Pruning - Explore all possibilities with basic pruning
2. Memoized Recursion - Add memoization to avoid recomputing subproblems
3. Trie + Memoization - Use trie for efficient word lookup with memoization
4. DP + Reconstruction - Build breakpoints with DP, then reconstruct solutions
5. BFS Path Tracking - Level-by-level exploration with path tracking
6. Optimized Backtracking - Enhanced pruning with reachability analysis
7. Advanced Trie - Enhanced trie with multiple words per node optimization

Key concepts:
- Memoization for avoiding exponential recomputation
- Trie data structure for efficient string matching
- Dynamic programming for pre-validation
- Backtracking with intelligent pruning
- Path reconstruction techniques

Real-world applications:
- Search query segmentation
- Domain name analysis
- Code identifier decomposition
- Natural language compound word analysis

Each approach demonstrates different strategies for handling the exponential
nature of generating all possible sentence constructions efficiently.
"""
