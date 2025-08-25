"""
139. Word Break - Multiple Approaches
Difficulty: Easy

Given a string s and a dictionary of strings wordDict, return true if s can be segmented 
into a space-separated sequence of one or more dictionary words.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

LeetCode Problem: https://leetcode.com/problems/word-break/

Example:
Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
"""

from typing import List, Set, Dict, Optional
from collections import deque, defaultdict
import time

class TrieNode:
    """Trie node for word break problem"""
    def __init__(self):
        self.children = {}
        self.is_word = False

class Solution:
    
    def wordBreak1(self, s: str, wordDict: List[str]) -> bool:
        """
        Approach 1: Brute Force with Recursion
        
        Try all possible ways to break the string.
        
        Time: O(2^n) exponential without memoization
        Space: O(n) recursion depth
        """
        word_set = set(wordDict)
        
        def can_break(start: int) -> bool:
            """Check if string from start can be broken"""
            if start == len(s):
                return True
            
            for end in range(start + 1, len(s) + 1):
                prefix = s[start:end]
                if prefix in word_set and can_break(end):
                    return True
            
            return False
        
        return can_break(0)
    
    def wordBreak2(self, s: str, wordDict: List[str]) -> bool:
        """
        Approach 2: Dynamic Programming
        
        Use DP to avoid recomputing subproblems.
        
        Time: O(n^2 * m) where n=len(s), m=max(len(word))
        Space: O(n)
        """
        word_set = set(wordDict)
        n = len(s)
        
        # dp[i] = True if s[0:i] can be broken into words
        dp = [False] * (n + 1)
        dp[0] = True  # Empty string can always be broken
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        
        return dp[n]
    
    def wordBreak3(self, s: str, wordDict: List[str]) -> bool:
        """
        Approach 3: BFS Approach
        
        Use BFS to explore all possible break points.
        
        Time: O(n^2 * m)
        Space: O(n)
        """
        word_set = set(wordDict)
        queue = deque([0])  # Start positions
        visited = set()
        
        while queue:
            start = queue.popleft()
            
            if start == len(s):
                return True
            
            if start in visited:
                continue
            
            visited.add(start)
            
            for end in range(start + 1, len(s) + 1):
                if s[start:end] in word_set:
                    queue.append(end)
        
        return False
    
    def wordBreak4(self, s: str, wordDict: List[str]) -> bool:
        """
        Approach 4: Trie + DP
        
        Build trie for efficient word lookup, then use DP.
        
        Time: O(n^2 + total_word_length)
        Space: O(total_word_length) for trie + O(n) for DP
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
        
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(1, n + 1):
            node = root
            
            # Check all possible words ending at position i
            for j in range(i - 1, -1, -1):
                char = s[j]
                if char not in node.children:
                    break
                
                node = node.children[char]
                
                if node.is_word and dp[j]:
                    dp[i] = True
                    break
        
        return dp[n]
    
    def wordBreak5(self, s: str, wordDict: List[str]) -> bool:
        """
        Approach 5: Optimized DP with Word Length Filtering
        
        Optimize by only checking relevant substring lengths.
        
        Time: O(n * max_word_length)
        Space: O(n)
        """
        word_set = set(wordDict)
        max_len = max(len(word) for word in wordDict) if wordDict else 0
        n = len(s)
        
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(1, n + 1):
            # Only check substrings of valid lengths
            for length in range(1, min(i, max_len) + 1):
                if dp[i - length] and s[i - length:i] in word_set:
                    dp[i] = True
                    break
        
        return dp[n]
    
    def wordBreak6(self, s: str, wordDict: List[str]) -> bool:
        """
        Approach 6: Memoized Recursion
        
        Add memoization to the recursive approach.
        
        Time: O(n^2)
        Space: O(n) for memoization + O(n) recursion
        """
        word_set = set(wordDict)
        memo = {}
        
        def can_break(start: int) -> bool:
            if start == len(s):
                return True
            
            if start in memo:
                return memo[start]
            
            for end in range(start + 1, len(s) + 1):
                prefix = s[start:end]
                if prefix in word_set and can_break(end):
                    memo[start] = True
                    return True
            
            memo[start] = False
            return False
        
        return can_break(0)
    
    def wordBreak7(self, s: str, wordDict: List[str]) -> bool:
        """
        Approach 7: Advanced Trie with Early Termination
        
        Use trie with optimizations for early termination.
        
        Time: O(n^2)
        Space: O(total_word_length)
        """
        class AdvancedTrieNode:
            def __init__(self):
                self.children = {}
                self.is_word = False
                self.min_remaining = float('inf')  # Minimum chars needed from this node
        
        # Build advanced trie with preprocessing
        root = AdvancedTrieNode()
        
        for word in wordDict:
            node = root
            for i, char in enumerate(word):
                if char not in node.children:
                    node.children[char] = AdvancedTrieNode()
                node = node.children[char]
                node.min_remaining = min(node.min_remaining, len(word) - i - 1)
            node.is_word = True
            node.min_remaining = 0
        
        # DP with trie traversal
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(n):
            if not dp[i]:
                continue
            
            node = root
            for j in range(i, n):
                char = s[j]
                if char not in node.children:
                    break
                
                node = node.children[char]
                
                # Early termination: not enough characters left
                if j - i + 1 + node.min_remaining > n - i:
                    break
                
                if node.is_word:
                    dp[j + 1] = True
        
        return dp[n]


def test_basic_functionality():
    """Test basic functionality"""
    print("=== Testing Basic Functionality ===")
    
    solution = Solution()
    
    test_cases = [
        # LeetCode examples
        ("leetcode", ["leet", "code"], True),
        ("applepenapple", ["apple", "pen"], True),
        ("catsandog", ["cats", "dog", "sand", "and", "cat"], False),
        
        # Edge cases
        ("", [], True),  # Empty string
        ("a", ["a"], True),  # Single character
        ("a", ["b"], False),  # No match
        
        # Complex cases
        ("aaaaaaa", ["aaaa", "aaa"], True),
        ("aaaaaaa", ["aaaa", "aa"], False),
        ("abcd", ["a", "abc", "b", "cd"], True),
        
        # Overlapping words
        ("wordwordword", ["word", "wordword"], True),
        ("catscatcat", ["cats", "cat", "catcat"], True),
    ]
    
    approaches = [
        ("Brute Force", solution.wordBreak1),
        ("Dynamic Programming", solution.wordBreak2),
        ("BFS", solution.wordBreak3),
        ("Trie + DP", solution.wordBreak4),
        ("Optimized DP", solution.wordBreak5),
        ("Memoized Recursion", solution.wordBreak6),
        ("Advanced Trie", solution.wordBreak7),
    ]
    
    for i, (s, wordDict, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: s='{s}', words={wordDict}")
        print(f"Expected: {expected}")
        
        for name, method in approaches:
            try:
                result = method(s, wordDict[:])
                status = "✓" if result == expected else "✗"
                print(f"  {name:20}: {result} {status}")
            except Exception as e:
                print(f"  {name:20}: Error - {e}")


def demonstrate_dp_approach():
    """Demonstrate DP approach step by step"""
    print("\n=== DP Approach Demo ===")
    
    s = "leetcode"
    wordDict = ["leet", "code", "le", "et", "co", "de"]
    
    print(f"String: '{s}'")
    print(f"Dictionary: {wordDict}")
    
    word_set = set(wordDict)
    n = len(s)
    
    # DP table
    dp = [False] * (n + 1)
    dp[0] = True
    
    print(f"\nDP Table Evolution:")
    print(f"dp[0] = True (empty string)")
    
    for i in range(1, n + 1):
        print(f"\nChecking position {i} (character '{s[i-1]}'):")
        
        for j in range(i):
            substring = s[j:i]
            
            if dp[j] and substring in word_set:
                dp[i] = True
                print(f"  Found: s[{j}:{i}] = '{substring}' (dp[{j}] = {dp[j]})")
                print(f"  Setting dp[{i}] = True")
                break
            elif dp[j]:
                print(f"  Checked: s[{j}:{i}] = '{substring}' (not in dict)")
        
        print(f"  dp[{i}] = {dp[i]}")
    
    print(f"\nFinal result: {dp[n]}")
    print(f"DP array: {dp}")


def demonstrate_trie_approach():
    """Demonstrate trie-based approach"""
    print("\n=== Trie Approach Demo ===")
    
    s = "catsanddog"
    wordDict = ["cat", "cats", "and", "sand", "dog"]
    
    print(f"String: '{s}'")
    print(f"Dictionary: {wordDict}")
    
    # Build trie
    root = TrieNode()
    
    print(f"\nBuilding trie:")
    for word in wordDict:
        print(f"Inserting '{word}':")
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
                print(f"  Created node for '{char}'")
            node = node.children[char]
        node.is_word = True
        print(f"  Marked end of word '{word}'")
    
    # DP with trie
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    
    print(f"\nDP with trie traversal:")
    
    for i in range(1, n + 1):
        print(f"\nPosition {i}:")
        node = root
        found = False
        
        for j in range(i - 1, -1, -1):
            char = s[j]
            print(f"  Checking character '{char}' at position {j}")
            
            if char not in node.children:
                print(f"    No path in trie, stopping")
                break
            
            node = node.children[char]
            
            if node.is_word and dp[j]:
                print(f"    Found word s[{j}:{i}] = '{s[j:i]}' and dp[{j}] = True")
                dp[i] = True
                found = True
                break
            elif node.is_word:
                print(f"    Found word s[{j}:{i}] = '{s[j:i]}' but dp[{j}] = False")
        
        print(f"  dp[{i}] = {dp[i]}")
    
    print(f"\nResult: {dp[n]}")


def demonstrate_optimization_techniques():
    """Demonstrate optimization techniques"""
    print("\n=== Optimization Techniques Demo ===")
    
    print("1. Word Length Filtering:")
    
    s = "aaaaaaaaab"
    wordDict = ["a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa", "b"]
    
    print(f"   String: '{s}' (length {len(s)})")
    print(f"   Dictionary: {wordDict}")
    
    max_word_len = max(len(word) for word in wordDict)
    print(f"   Max word length: {max_word_len}")
    
    # Without optimization: check all substrings
    total_checks_naive = 0
    for i in range(1, len(s) + 1):
        total_checks_naive += i
    
    # With optimization: only check relevant lengths
    total_checks_optimized = 0
    for i in range(1, len(s) + 1):
        total_checks_optimized += min(i, max_word_len)
    
    print(f"   Naive approach: {total_checks_naive} substring checks")
    print(f"   Optimized: {total_checks_optimized} substring checks")
    print(f"   Reduction: {(1 - total_checks_optimized/total_checks_naive)*100:.1f}%")
    
    print("\n2. Early Termination in Trie:")
    
    # Show how trie can terminate early
    trie_root = TrieNode()
    for word in ["hello", "help", "world"]:
        node = trie_root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
    
    test_string = "helloworld"
    print(f"   String: '{test_string}'")
    print(f"   Dictionary: ['hello', 'help', 'world']")
    
    print(f"   Trie traversal at position 0:")
    node = trie_root
    for i, char in enumerate(test_string):
        if char in node.children:
            node = node.children[char]
            if node.is_word:
                print(f"     Found word: '{test_string[0:i+1]}'")
        else:
            print(f"     No path for '{char}' at position {i}, early termination")
            break
    
    print("\n3. Memoization Benefits:")
    
    # Show overlapping subproblems
    s = "ababa"
    print(f"   String: '{s}'")
    print(f"   Overlapping subproblems in recursive approach:")
    
    # Simulate recursive calls (without actually implementing)
    subproblems = set()
    
    def simulate_calls(start: int, path: str = ""):
        if start >= len(s):
            return
        
        call_signature = f"canBreak({start})"
        if call_signature in subproblems:
            print(f"     {path} -> {call_signature} (REPEATED)")
        else:
            print(f"     {path} -> {call_signature}")
            subproblems.add(call_signature)
        
        # Simulate trying different word lengths
        for length in [1, 2]:
            if start + length <= len(s):
                simulate_calls(start + length, path + f"  ")
    
    simulate_calls(0)
    print(f"   Total unique subproblems: {len(subproblems)}")


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import random
    import string
    
    solution = Solution()
    
    # Generate test data
    def generate_test_case(string_length: int, dict_size: int, word_length_range: tuple) -> tuple:
        # Generate dictionary
        words = set()
        while len(words) < dict_size:
            length = random.randint(*word_length_range)
            word = ''.join(random.choices(string.ascii_lowercase[:5], k=length))
            words.add(word)
        
        word_list = list(words)
        
        # Generate string that can be broken (sometimes)
        if random.random() < 0.7:  # 70% chance of valid break
            s = ""
            while len(s) < string_length:
                word = random.choice(word_list)
                if len(s) + len(word) <= string_length:
                    s += word
                else:
                    # Fill remaining with random chars
                    remaining = string_length - len(s)
                    s += ''.join(random.choices(string.ascii_lowercase[:5], k=remaining))
                    break
        else:
            # Generate random string
            s = ''.join(random.choices(string.ascii_lowercase[:5], k=string_length))
        
        return s, word_list
    
    test_scenarios = [
        ("Small", 20, 10, (2, 4)),
        ("Medium", 50, 20, (3, 6)),
        ("Large", 100, 30, (4, 8)),
    ]
    
    approaches = [
        ("DP", solution.wordBreak2),
        ("BFS", solution.wordBreak3),
        ("Trie+DP", solution.wordBreak4),
        ("Optimized DP", solution.wordBreak5),
        ("Memoized", solution.wordBreak6),
        ("Advanced Trie", solution.wordBreak7),
    ]
    
    for scenario_name, str_len, dict_size, word_len_range in test_scenarios:
        s, wordDict = generate_test_case(str_len, dict_size, word_len_range)
        
        print(f"\n--- {scenario_name} Test Case ---")
        print(f"String length: {len(s)}, Dictionary size: {len(wordDict)}")
        
        for approach_name, method in approaches:
            start_time = time.time()
            
            try:
                result = method(s, wordDict)
                end_time = time.time()
                
                execution_time = (end_time - start_time) * 1000
                print(f"  {approach_name:15}: {result!s:5} in {execution_time:6.2f}ms")
            
            except Exception as e:
                print(f"  {approach_name:15}: Error - {str(e)[:20]}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solution = Solution()
    
    # Application 1: Text processing and tokenization
    print("1. Text Tokenization:")
    
    text = "machinelearning"
    dictionary = ["machine", "learning", "learn", "ing", "mac", "hine"]
    
    can_tokenize = solution.wordBreak2(text, dictionary)
    print(f"   Text: '{text}'")
    print(f"   Dictionary: {dictionary}")
    print(f"   Can tokenize: {can_tokenize}")
    
    if can_tokenize:
        # Find one possible tokenization
        def find_tokenization(s: str, words: List[str]) -> List[str]:
            word_set = set(words)
            n = len(s)
            dp = [False] * (n + 1)
            parent = [-1] * (n + 1)
            
            dp[0] = True
            
            for i in range(1, n + 1):
                for j in range(i):
                    if dp[j] and s[j:i] in word_set:
                        dp[i] = True
                        parent[i] = j
                        break
            
            if not dp[n]:
                return []
            
            # Reconstruct path
            result = []
            pos = n
            while pos > 0:
                prev = parent[pos]
                result.append(s[prev:pos])
                pos = prev
            
            return result[::-1]
        
        tokens = find_tokenization(text, dictionary)
        print(f"   Tokenization: {tokens}")
    
    # Application 2: Domain name validation
    print(f"\n2. Domain Name Validation:")
    
    domain = "googlemailcom"
    valid_parts = ["google", "mail", "com", "gmail", "yahoo", "hotmail"]
    
    is_valid = solution.wordBreak2(domain, valid_parts)
    print(f"   Domain: '{domain}'")
    print(f"   Valid parts: {valid_parts}")
    print(f"   Is valid combination: {is_valid}")
    
    # Application 3: Password strength checking
    print(f"\n3. Password Pattern Analysis:")
    
    password = "password123"
    weak_patterns = ["password", "123", "abc", "qwerty", "admin", "user"]
    
    contains_weak = solution.wordBreak2(password, weak_patterns)
    print(f"   Password: '{password}'")
    print(f"   Weak patterns: {weak_patterns}")
    print(f"   Contains weak patterns: {contains_weak}")
    
    # Application 4: Code obfuscation detection
    print(f"\n4. Code Obfuscation Detection:")
    
    code_string = "ifelseforiloop"
    keywords = ["if", "else", "for", "while", "do", "loop", "function", "return"]
    
    contains_keywords = solution.wordBreak2(code_string, keywords)
    print(f"   Code string: '{code_string}'")
    print(f"   Keywords: {keywords}")
    print(f"   Contains concatenated keywords: {contains_keywords}")


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
        ("aaaa", ["a"], "Repeated single character"),
        ("aaaa", ["aa"], "Repeated double character"),
        ("aaaa", ["aaa"], "Not evenly divisible"),
        
        # Overlapping words
        ("abab", ["a", "ab", "aba", "abab"], "Multiple valid breakdowns"),
        
        # Long string with short words
        ("a" * 100, ["a"], "Very long string, single word"),
        
        # Many small words
        ("abcdefghijk", [chr(ord('a') + i) for i in range(11)], "Each character is a word"),
        
        # No solution possible
        ("abc", ["ab", "bc"], "Impossible to break completely"),
        
        # Case sensitivity (assuming case sensitive)
        ("ABC", ["abc"], "Case mismatch"),
    ]
    
    for s, wordDict, description in edge_cases:
        print(f"\n{description}:")
        print(f"  s = '{s[:20]}{'...' if len(s) > 20 else ''}'")
        print(f"  wordDict = {wordDict}")
        
        try:
            result = solution.wordBreak2(s, wordDict)
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {e}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Brute Force Recursion",
         "Time: O(2^n) - exponential without memoization",
         "Space: O(n) - recursion depth"),
        
        ("Dynamic Programming",
         "Time: O(n^2 * m) where m is max word length",
         "Space: O(n) - DP array"),
        
        ("BFS Approach",
         "Time: O(n^2 * m) - similar to DP",
         "Space: O(n) - queue and visited set"),
        
        ("Trie + DP",
         "Time: O(n^2 + W) where W is total word characters",
         "Space: O(W) for trie + O(n) for DP"),
        
        ("Optimized DP",
         "Time: O(n * max_word_length)",
         "Space: O(n) - DP array"),
        
        ("Memoized Recursion",
         "Time: O(n^2) - each subproblem solved once",
         "Space: O(n) - memoization + recursion"),
        
        ("Advanced Trie",
         "Time: O(n^2) with early termination optimizations",
         "Space: O(W) - trie with additional metadata"),
    ]
    
    print("Approach Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nKey Optimization Insights:")
    print(f"  • Word length filtering reduces time from O(n^2*m) to O(n*max_len)")
    print(f"  • Trie eliminates repeated string hashing/comparison")
    print(f"  • Memoization prevents exponential blowup in recursion")
    print(f"  • Early termination in trie reduces unnecessary traversals")
    
    print(f"\nPractical Considerations:")
    print(f"  • For small dictionaries: Optimized DP is often fastest")
    print(f"  • For large dictionaries: Trie-based approaches win")
    print(f"  • For very long strings: Memory usage becomes important")
    print(f"  • For repeated queries: Pre-build trie and reuse")
    
    print(f"\nRecommendations:")
    print(f"  • Use Optimized DP for general purpose applications")
    print(f"  • Use Trie + DP when dictionary is large or reused")
    print(f"  • Use Memoized Recursion for easy implementation")
    print(f"  • Avoid Brute Force for anything but tiny inputs")


if __name__ == "__main__":
    test_basic_functionality()
    demonstrate_dp_approach()
    demonstrate_trie_approach()
    demonstrate_optimization_techniques()
    benchmark_approaches()
    demonstrate_real_world_applications()
    test_edge_cases()
    analyze_complexity()

"""
139. Word Break demonstrates comprehensive dynamic programming approaches with trie optimization:

1. Brute Force Recursion - Exponential time exploration of all possibilities
2. Dynamic Programming - O(n²) bottom-up approach with memoization
3. BFS Approach - Level-by-level exploration of break points
4. Trie + DP - Efficient word lookup with trie data structure
5. Optimized DP - Word length filtering for performance improvement
6. Memoized Recursion - Top-down DP with recursive structure
7. Advanced Trie - Enhanced trie with early termination optimizations

Key concepts:
- Dynamic programming for avoiding recomputation
- Trie data structure for efficient string matching
- Optimization techniques (filtering, memoization, early termination)
- Multiple algorithmic paradigms (DP, BFS, recursion)

Real-world applications:
- Text tokenization and natural language processing
- Domain name validation
- Password pattern analysis
- Code obfuscation detection

Each approach demonstrates different trade-offs between implementation
complexity, time efficiency, and space usage for string segmentation problems.
"""
