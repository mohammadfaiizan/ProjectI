"""
1858. Longest Word With All Prefixes - Multiple Approaches
Difficulty: Medium

Given an array of strings words, find the longest string in words such that every prefix 
of it is also in words.

For example, let words = ["a", "app", "ap", "appl", "apply"]. The string "apply" has 
prefixes "a", "ap", "app", "appl", and "apply". All of these are in words.

Return the string described above. If there is more than one string with the same length, 
return the lexicographically smallest one. If no such string exists, return "".

LeetCode Problem: https://leetcode.com/problems/longest-word-with-all-prefixes/

Example:
Input: words = ["k","ki","kir","kira", "kiran"]
Output: "kiran"
"""

from typing import List, Set, Dict, Optional
from collections import defaultdict
import heapq

class TrieNode:
    """Trie node for prefix checking"""
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word = ""

class Solution:
    
    def longestWord1(self, words: List[str]) -> str:
        """
        Approach 1: Brute Force with Set
        
        For each word, check if all its prefixes exist in the word set.
        
        Time: O(sum of word lengths squared)
        Space: O(sum of word lengths)
        """
        if not words:
            return ""
        
        word_set = set(words)
        
        def has_all_prefixes(word: str) -> bool:
            """Check if all prefixes of word exist in word_set"""
            for i in range(1, len(word)):
                if word[:i] not in word_set:
                    return False
            return True
        
        # Find longest word with all prefixes
        longest = ""
        
        for word in words:
            if has_all_prefixes(word):
                if len(word) > len(longest) or (len(word) == len(longest) and word < longest):
                    longest = word
        
        return longest
    
    def longestWord2(self, words: List[str]) -> str:
        """
        Approach 2: Trie Construction and DFS
        
        Build trie and use DFS to find longest buildable path.
        
        Time: O(sum of word lengths)
        Space: O(sum of word lengths)
        """
        if not words:
            return ""
        
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
        
        # DFS to find longest buildable word
        def dfs(node: TrieNode) -> str:
            """Find longest buildable word from current node"""
            if not node.is_word and node != root:
                return ""  # Can't continue if current position is not a word
            
            longest = node.word
            
            # Try all children that are complete words
            for child in node.children.values():
                if child.is_word:  # Can only continue if child is a word
                    candidate = dfs(child)
                    if len(candidate) > len(longest) or (len(candidate) == len(longest) and candidate < longest):
                        longest = candidate
            
            return longest
        
        return dfs(root)
    
    def longestWord3(self, words: List[str]) -> str:
        """
        Approach 3: Sort and Build Incrementally
        
        Sort words and build valid words incrementally.
        
        Time: O(n log n + sum of word lengths)
        Space: O(n)
        """
        if not words:
            return ""
        
        # Sort by length first, then lexicographically
        words.sort(key=lambda x: (len(x), x))
        
        buildable = set()
        longest = ""
        
        for word in words:
            if len(word) == 1 or word[:-1] in buildable:
                buildable.add(word)
                if len(word) > len(longest):
                    longest = word
        
        return longest
    
    def longestWord4(self, words: List[str]) -> str:
        """
        Approach 4: BFS Level by Level
        
        Use BFS to explore words level by level.
        
        Time: O(sum of word lengths)
        Space: O(sum of word lengths)
        """
        if not words:
            return ""
        
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
        
        # BFS to find longest word
        from collections import deque
        
        queue = deque([root])
        longest_word = ""
        
        while queue:
            node = queue.popleft()
            
            # Update longest word if current is longer or lexicographically smaller
            if node.word and (len(node.word) > len(longest_word) or 
                             (len(node.word) == len(longest_word) and node.word < longest_word)):
                longest_word = node.word
            
            # Add children that are complete words (can be extended)
            for child in node.children.values():
                if child.is_word:
                    queue.append(child)
        
        return longest_word
    
    def longestWord5(self, words: List[str]) -> str:
        """
        Approach 5: Union-Find for Path Building
        
        Use Union-Find to track buildable word chains.
        
        Time: O(sum of word lengths * α(n))
        Space: O(n)
        """
        if not words:
            return ""
        
        class UnionFind:
            def __init__(self, words):
                self.parent = {}
                self.rank = {}
                self.word_length = {}
                
                for word in words:
                    self.parent[word] = word
                    self.rank[word] = 0
                    self.word_length[word] = len(word)
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return
                
                if self.rank[px] < self.rank[py]:
                    px, py = py, px
                
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
        
        word_set = set(words)
        uf = UnionFind(words)
        
        # Connect words that can extend each other
        for word in words:
            if len(word) > 1:
                prefix = word[:-1]
                if prefix in word_set:
                    uf.union(word, prefix)
        
        # Find longest buildable chain
        longest = ""
        
        for word in words:
            # Check if this word can be built (all prefixes exist)
            can_build = True
            for i in range(1, len(word)):
                if word[:i] not in word_set:
                    can_build = False
                    break
            
            if can_build:
                if len(word) > len(longest) or (len(word) == len(longest) and word < longest):
                    longest = word
        
        return longest
    
    def longestWord6(self, words: List[str]) -> str:
        """
        Approach 6: Dynamic Programming with Memoization
        
        Use DP to cache results of prefix checking.
        
        Time: O(sum of word lengths)
        Space: O(sum of word lengths)
        """
        if not words:
            return ""
        
        word_set = set(words)
        memo = {}
        
        def can_build(word: str) -> bool:
            """Check if word can be built with memoization"""
            if word in memo:
                return memo[word]
            
            if len(word) == 1:
                result = word in word_set
                memo[word] = result
                return result
            
            # Check if word exists and prefix can be built
            prefix = word[:-1]
            result = word in word_set and can_build(prefix)
            memo[word] = result
            return result
        
        longest = ""
        
        for word in words:
            if can_build(word):
                if len(word) > len(longest) or (len(word) == len(longest) and word < longest):
                    longest = word
        
        return longest
    
    def longestWord7(self, words: List[str]) -> str:
        """
        Approach 7: Optimized Trie with Early Termination
        
        Build trie with optimizations and early termination.
        
        Time: O(sum of word lengths)
        Space: O(sum of word lengths)
        """
        if not words:
            return ""
        
        class OptimizedTrieNode:
            def __init__(self):
                self.children = {}
                self.is_word = False
                self.word = ""
                self.max_buildable_length = 0  # Maximum buildable length in subtree
        
        # Build optimized trie
        root = OptimizedTrieNode()
        
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = OptimizedTrieNode()
                node = node.children[char]
            node.is_word = True
            node.word = word
        
        # Calculate max buildable length for each node
        def calculate_max_buildable(node: OptimizedTrieNode) -> int:
            """Calculate maximum buildable length in subtree"""
            max_length = len(node.word) if node.is_word else 0
            
            for child in node.children.values():
                if child.is_word:  # Can only continue if child is buildable
                    child_max = calculate_max_buildable(child)
                    max_length = max(max_length, child_max)
            
            node.max_buildable_length = max_length
            return max_length
        
        calculate_max_buildable(root)
        
        # Find longest buildable word with early termination
        def find_longest_optimized(node: OptimizedTrieNode, current_best_length: int) -> str:
            """Find longest buildable word with pruning"""
            if node.max_buildable_length <= current_best_length:
                return ""  # Prune: can't improve
            
            if not node.is_word and node != root:
                return ""
            
            longest = node.word
            
            # Sort children to get lexicographically smallest first
            for char in sorted(node.children.keys()):
                child = node.children[char]
                if child.is_word and child.max_buildable_length > len(longest):
                    candidate = find_longest_optimized(child, len(longest))
                    if len(candidate) > len(longest) or (len(candidate) == len(longest) and candidate < longest):
                        longest = candidate
            
            return longest
        
        return find_longest_optimized(root, 0)


def test_basic_functionality():
    """Test basic functionality"""
    print("=== Testing Basic Functionality ===")
    
    solution = Solution()
    
    test_cases = [
        # LeetCode examples
        (["k","ki","kir","kira", "kiran"], "kiran"),
        (["a", "banana", "app", "appl", "ap", "apply", "application"], "application"),
        (["b", "br", "bre", "brea", "break", "breakf", "breakfa", "breakfast"], "breakfast"),
        
        # Edge cases
        ([], ""),
        (["a"], "a"),
        (["abc", "a"], "a"),
        (["ab", "a", "abc"], "abc"),
        
        # Multiple same length
        (["cat", "cats", "c", "ca"], "cats"),
        (["dog", "dogs", "d", "do"], "dogs"),
        
        # Lexicographic ordering
        (["a", "ab", "abc", "x", "xy", "xyz"], "abc"),  # Both abc and xyz valid, abc is smaller
        
        # No valid buildable word
        (["abc", "def"], ""),
        (["ab", "cd"], ""),
    ]
    
    approaches = [
        ("Brute Force", solution.longestWord1),
        ("Trie DFS", solution.longestWord2),
        ("Sort and Build", solution.longestWord3),
        ("BFS Level", solution.longestWord4),
        ("Union-Find", solution.longestWord5),
        ("DP Memoization", solution.longestWord6),
        ("Optimized Trie", solution.longestWord7),
    ]
    
    for i, (words, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {words}")
        print(f"Expected: '{expected}'")
        
        for name, method in approaches:
            try:
                result = method(words[:])
                status = "✓" if result == expected else "✗"
                print(f"  {name:15}: '{result}' {status}")
            except Exception as e:
                print(f"  {name:15}: Error - {e}")


def demonstrate_trie_construction():
    """Demonstrate trie construction and traversal"""
    print("\n=== Trie Construction Demo ===")
    
    words = ["a", "app", "ap", "appl", "apply"]
    
    print(f"Words: {words}")
    print(f"Building trie to find longest buildable word...")
    
    # Build trie step by step
    root = TrieNode()
    
    print(f"\nBuilding trie:")
    for word in words:
        print(f"\nInserting '{word}':")
        node = root
        
        for i, char in enumerate(word):
            if char not in node.children:
                node.children[char] = TrieNode()
                print(f"  Created node for '{char}' at depth {i+1}")
            
            node = node.children[char]
            print(f"  At node for prefix '{word[:i+1]}'")
        
        node.is_word = True
        node.word = word
        print(f"  Marked '{word}' as complete word")
    
    # Demonstrate DFS traversal
    print(f"\nDFS traversal to find buildable words:")
    
    def dfs_demo(node: TrieNode, depth: int = 0) -> str:
        """DFS with detailed output"""
        indent = "  " * depth
        
        if not node.is_word and node != root:
            print(f"{indent}Cannot continue from non-word node")
            return ""
        
        current_word = node.word if node.word else "[ROOT]"
        print(f"{indent}At node: '{current_word}'")
        
        longest = node.word
        
        for char, child in sorted(node.children.items()):
            print(f"{indent}Exploring child '{char}'")
            if child.is_word:
                candidate = dfs_demo(child, depth + 1)
                if len(candidate) > len(longest) or (len(candidate) == len(longest) and candidate < longest):
                    longest = candidate
                    print(f"{indent}New longest: '{longest}'")
            else:
                print(f"{indent}Child '{char}' is not a complete word, skipping")
        
        return longest
    
    result = dfs_demo(root)
    print(f"\nFinal result: '{result}'")


def demonstrate_building_process():
    """Demonstrate the word building process"""
    print("\n=== Word Building Process Demo ===")
    
    words = ["a", "app", "appl", "apply", "application"]
    target = "application"
    
    print(f"Words: {words}")
    print(f"Target word: '{target}'")
    print(f"Checking if '{target}' can be built step by step:")
    
    word_set = set(words)
    
    # Check each prefix
    buildable = True
    for i in range(1, len(target) + 1):
        prefix = target[:i]
        exists = prefix in word_set
        status = "✓" if exists else "✗"
        print(f"  Step {i}: '{prefix}' {'exists' if exists else 'missing'} {status}")
        
        if not exists:
            buildable = False
            print(f"    Cannot build '{target}' - missing prefix '{prefix}'")
            break
    
    if buildable:
        print(f"  '{target}' can be built successfully!")
    
    # Show all buildable words
    print(f"\nFinding all buildable words:")
    
    def can_build(word: str) -> bool:
        for i in range(1, len(word)):
            if word[:i] not in word_set:
                return False
        return True
    
    buildable_words = []
    for word in words:
        if can_build(word):
            buildable_words.append(word)
            print(f"  '{word}': buildable")
        else:
            print(f"  '{word}': not buildable")
    
    # Find longest
    if buildable_words:
        longest = max(buildable_words, key=lambda x: (len(x), -ord(x[0]) if x else 0))
        print(f"\nLongest buildable word: '{longest}'")
    else:
        print(f"\nNo buildable words found")


def demonstrate_optimization_techniques():
    """Demonstrate optimization techniques"""
    print("\n=== Optimization Techniques Demo ===")
    
    words = ["a", "app", "appl", "apply", "application", "b", "br", "bre", "brea", "break"]
    
    print("1. Sorting Optimization:")
    print(f"   Original words: {words}")
    
    # Sort by length then lexicographically
    sorted_words = sorted(words, key=lambda x: (len(x), x))
    print(f"   Sorted words: {sorted_words}")
    
    print(f"\n   Building incrementally:")
    buildable = set()
    
    for word in sorted_words:
        if len(word) == 1 or word[:-1] in buildable:
            buildable.add(word)
            print(f"     '{word}': added to buildable set")
        else:
            print(f"     '{word}': cannot build (missing prefix '{word[:-1]}')")
    
    print(f"   Final buildable set: {buildable}")
    
    print("\n2. Early Termination:")
    
    # Demonstrate how we can terminate early when we find a long enough word
    target_length = 5
    print(f"   Looking for word of length >= {target_length}")
    
    for word in sorted(words, key=lambda x: (-len(x), x)):  # Sort by length desc
        solution = Solution()
        if solution.longestWord6([w for w in words if len(w) <= len(word)]):
            # Check if this word is buildable
            can_build = True
            word_set = set(words)
            for i in range(1, len(word)):
                if word[:i] not in word_set:
                    can_build = False
                    break
            
            if can_build and len(word) >= target_length:
                print(f"     Found suitable word: '{word}' (length {len(word)})")
                break
        
        print(f"     '{word}': checking...")
    
    print("\n3. Memoization Benefits:")
    
    # Show how memoization helps with overlapping subproblems
    test_words = ["a", "ab", "abc", "abcd", "abcde", "ab", "abc"]  # Some duplicates
    
    print(f"   Words with potential overlap: {test_words}")
    
    memo = {}
    
    def can_build_memo(word: str, word_set: set) -> bool:
        if word in memo:
            print(f"     Cache hit for '{word}'")
            return memo[word]
        
        print(f"     Computing for '{word}'")
        
        if len(word) == 1:
            result = word in word_set
        else:
            prefix = word[:-1]
            result = word in word_set and can_build_memo(prefix, word_set)
        
        memo[word] = result
        return result
    
    word_set = set(test_words)
    for word in set(test_words):  # Remove duplicates for testing
        result = can_build_memo(word, word_set)
        print(f"     '{word}': {'buildable' if result else 'not buildable'}")


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import time
    import random
    import string
    
    solution = Solution()
    
    # Generate test data with buildable chains
    def generate_words_with_chains(base_count: int, max_length: int) -> List[str]:
        words = []
        
        # Generate some buildable chains
        for _ in range(base_count):
            # Start with random character
            base = random.choice(string.ascii_lowercase)
            words.append(base)
            
            # Build chain
            current = base
            for length in range(2, random.randint(2, max_length + 1)):
                current += random.choice(string.ascii_lowercase)
                words.append(current)
        
        # Add some random words that break chains
        for _ in range(base_count // 2):
            length = random.randint(2, max_length)
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.append(word)
        
        return list(set(words))  # Remove duplicates
    
    test_scenarios = [
        ("Small", generate_words_with_chains(5, 6)),
        ("Medium", generate_words_with_chains(10, 8)),
        ("Large", generate_words_with_chains(20, 10)),
    ]
    
    approaches = [
        ("Brute Force", solution.longestWord1),
        ("Trie DFS", solution.longestWord2),
        ("Sort+Build", solution.longestWord3),
        ("BFS Level", solution.longestWord4),
        ("DP Memo", solution.longestWord6),
    ]
    
    for scenario_name, words in test_scenarios:
        print(f"\n--- {scenario_name} Dataset ---")
        print(f"Words: {len(words)}, Max length: {max(len(w) for w in words) if words else 0}")
        
        for approach_name, method in approaches:
            start_time = time.time()
            
            # Run multiple times for better measurement
            for _ in range(5):
                result = method(words[:])
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 5
            
            print(f"  {approach_name:15}: {avg_time*1000:.2f}ms")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solution = Solution()
    
    edge_cases = [
        # Empty and single cases
        ([], "Empty list"),
        (["a"], "Single word"),
        (["ab"], "Single word - not buildable"),
        
        # No buildable words
        (["abc", "def", "ghi"], "No buildable words"),
        
        # All single characters
        (["a", "b", "c", "d"], "All single characters"),
        
        # Multiple chains
        (["a", "ab", "abc", "x", "xy", "xyz"], "Multiple chains"),
        
        # Same length words
        (["abc", "def", "a", "b", "c", "d", "ab", "de"], "Multiple same length"),
        
        # Lexicographic ordering
        (["z", "za", "zab", "a", "ab", "abc"], "Lexicographic ordering"),
        
        # Very long chain
        (["a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa"], "Long chain"),
        
        # Broken chain
        (["a", "ab", "abcd"], "Broken chain (missing abc)"),
        
        # Duplicate words
        (["a", "ab", "ab", "abc", "abc"], "Duplicate words"),
    ]
    
    for words, description in edge_cases:
        print(f"\n{description}:")
        print(f"  Words: {words}")
        
        try:
            result = solution.longestWord2(words)
            print(f"  Result: '{result}'")
            
            # Verify result
            if result:
                word_set = set(words)
                valid = all(result[:i] in word_set for i in range(1, len(result) + 1))
                print(f"  Valid: {valid}")
        except Exception as e:
            print(f"  Error: {e}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Brute Force Set Check",
         "Time: O(sum of word lengths²) - check all prefixes for each word",
         "Space: O(sum of word lengths) - set storage"),
        
        ("Trie with DFS",
         "Time: O(sum of word lengths) - build trie + DFS traversal",
         "Space: O(sum of word lengths) - trie storage"),
        
        ("Sort and Build Incrementally",
         "Time: O(n*log(n) + sum of word lengths)",
         "Space: O(n) - buildable set"),
        
        ("BFS Level by Level",
         "Time: O(sum of word lengths) - build trie + BFS",
         "Space: O(sum of word lengths) - trie + queue"),
        
        ("Union-Find",
         "Time: O(sum of word lengths * α(n))",
         "Space: O(n) - Union-Find structure"),
        
        ("DP with Memoization",
         "Time: O(sum of word lengths) - each prefix computed once",
         "Space: O(sum of unique prefixes) - memoization"),
        
        ("Optimized Trie",
         "Time: O(sum of word lengths) - with pruning optimizations",
         "Space: O(sum of word lengths) - trie with additional metadata"),
    ]
    
    print("Approach Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nKey Observations:")
    print(f"  • n = number of words")
    print(f"  • Trie-based approaches are generally most efficient")
    print(f"  • Sorting approach is simple but has O(n log n) overhead")
    print(f"  • Memoization helps when many words share prefixes")
    
    print(f"\nOptimization Strategies:")
    print(f"  • Use trie for efficient prefix checking")
    print(f"  • Early termination when longer words impossible")
    print(f"  • Memoization for overlapping prefix computations")
    print(f"  • Lexicographic processing for tie-breaking")
    
    print(f"\nRecommendations:")
    print(f"  • Use Trie DFS for optimal performance")
    print(f"  • Use Sort+Build for simple implementation")
    print(f"  • Use DP Memoization when prefix overlap is high")
    print(f"  • Consider Optimized Trie for very large inputs")


if __name__ == "__main__":
    test_basic_functionality()
    demonstrate_trie_construction()
    demonstrate_building_process()
    demonstrate_optimization_techniques()
    benchmark_approaches()
    test_edge_cases()
    analyze_complexity()

"""
1858. Longest Word With All Prefixes demonstrates multiple approaches for prefix validation:

1. Brute Force Set Check - Direct prefix validation for each word
2. Trie with DFS - Build trie and use depth-first search for buildable paths
3. Sort and Build Incrementally - Sort words and build valid set incrementally
4. BFS Level by Level - Use breadth-first search to explore buildable words
5. Union-Find - Track buildable word chains using Union-Find structure
6. DP with Memoization - Cache prefix validation results
7. Optimized Trie - Enhanced trie with pruning and early termination

Key concepts:
- Prefix validation and buildable word chains
- Trie traversal strategies (DFS vs BFS)
- Incremental building and validation
- Memoization for overlapping subproblems
- Lexicographic ordering for tie-breaking

Each approach offers different trade-offs between implementation complexity,
time efficiency, and space usage for prefix-based word validation problems.
"""
