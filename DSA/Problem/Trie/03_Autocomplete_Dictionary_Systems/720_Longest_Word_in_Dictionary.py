"""
720. Longest Word in Dictionary - Multiple Approaches
Difficulty: Easy

Given an array of strings words representing an English dictionary, return the longest 
word in words that can be built one character at a time by other words in words.

If there is more than one possible answer, return the longest word with the smallest 
lexicographical order. If there is no answer, return the empty string.

LeetCode Problem: https://leetcode.com/problems/longest-word-in-dictionary/

Example:
Input: words = ["w","wo","wor","worl","world"]
Output: "world"
Explanation: The word "world" can be built one character at a time by "w", "wo", "wor", and "worl".
"""

from typing import List, Set
from collections import defaultdict, deque

class TrieNode:
    """Trie node for building word chains"""
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word = ""

class Solution:
    
    def longestWord1(self, words: List[str]) -> str:
        """
        Approach 1: Trie with DFS
        
        Build trie and find longest buildable word using DFS.
        
        Time: O(sum of word lengths)
        Space: O(sum of word lengths)
        """
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
                return ""
            
            longest = node.word
            
            for child in node.children.values():
                if child.is_word:  # Can only continue if child is a complete word
                    candidate = dfs(child)
                    if len(candidate) > len(longest) or (len(candidate) == len(longest) and candidate < longest):
                        longest = candidate
            
            return longest
        
        return dfs(root)
    
    def longestWord2(self, words: List[str]) -> str:
        """
        Approach 2: BFS Level-by-Level
        
        Process words level by level using BFS.
        
        Time: O(sum of word lengths)
        Space: O(sum of word lengths)
        """
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
        queue = deque([root])
        longest_word = ""
        
        while queue:
            node = queue.popleft()
            
            # Update longest word if current is longer or lexicographically smaller
            if node.word and (len(node.word) > len(longest_word) or 
                             (len(node.word) == len(longest_word) and node.word < longest_word)):
                longest_word = node.word
            
            # Add children that are complete words
            for child in node.children.values():
                if child.is_word:
                    queue.append(child)
        
        return longest_word
    
    def longestWord3(self, words: List[str]) -> str:
        """
        Approach 3: Set-based Approach
        
        Use set to check if all prefixes exist.
        
        Time: O(sum of word lengths squared)
        Space: O(number of words)
        """
        word_set = set(words)
        
        def can_build(word: str) -> bool:
            """Check if word can be built character by character"""
            for i in range(1, len(word)):
                if word[:i] not in word_set:
                    return False
            return True
        
        # Find longest buildable word
        longest = ""
        
        for word in words:
            if can_build(word):
                if len(word) > len(longest) or (len(word) == len(longest) and word < longest):
                    longest = word
        
        return longest
    
    def longestWord4(self, words: List[str]) -> str:
        """
        Approach 4: Sort and Build Incrementally
        
        Sort words and build incrementally.
        
        Time: O(n log n + sum of word lengths)
        Space: O(number of words)
        """
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
    
    def longestWord5(self, words: List[str]) -> str:
        """
        Approach 5: Dynamic Programming with Memoization
        
        Use DP to cache buildability results.
        
        Time: O(sum of word lengths)
        Space: O(number of words)
        """
        word_set = set(words)
        memo = {}
        
        def can_build_dp(word: str) -> bool:
            """Check if word can be built using DP"""
            if word in memo:
                return memo[word]
            
            if len(word) == 1:
                memo[word] = word in word_set
                return memo[word]
            
            # Check if word exists and prefix can be built
            result = word in word_set and can_build_dp(word[:-1])
            memo[word] = result
            return result
        
        longest = ""
        
        for word in words:
            if can_build_dp(word):
                if len(word) > len(longest) or (len(word) == len(longest) and word < longest):
                    longest = word
        
        return longest
    
    def longestWord6(self, words: List[str]) -> str:
        """
        Approach 6: Trie with Bottom-up Building
        
        Build trie and mark buildable nodes bottom-up.
        
        Time: O(sum of word lengths)
        Space: O(sum of word lengths)
        """
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
        
        # Mark buildable nodes
        def mark_buildable(node: TrieNode, is_buildable: bool) -> None:
            """Mark nodes that can be built incrementally"""
            if node.is_word and is_buildable:
                node.buildable = True
            else:
                node.buildable = False
            
            for child in node.children.values():
                mark_buildable(child, node.buildable if node == root else node.buildable and node.is_word)
        
        # Add buildable attribute to nodes
        def add_buildable_attr(node: TrieNode) -> None:
            node.buildable = False
            for child in node.children.values():
                add_buildable_attr(child)
        
        add_buildable_attr(root)
        mark_buildable(root, True)
        
        # Find longest buildable word
        def find_longest(node: TrieNode) -> str:
            """Find longest buildable word in subtree"""
            longest = node.word if hasattr(node, 'buildable') and node.buildable else ""
            
            for child in node.children.values():
                if hasattr(child, 'buildable') and child.buildable:
                    candidate = find_longest(child)
                    if len(candidate) > len(longest) or (len(candidate) == len(longest) and candidate < longest):
                        longest = candidate
            
            return longest
        
        return find_longest(root)
    
    def longestWord7(self, words: List[str]) -> str:
        """
        Approach 7: Graph-based Approach
        
        Model as graph where edges represent "can extend" relationship.
        
        Time: O(sum of word lengths + V + E)
        Space: O(V + E) where V = words, E = extensions
        """
        # Build adjacency list
        word_to_extensions = defaultdict(list)
        word_set = set(words)
        
        for word in words:
            for i in range(len(word)):
                prefix = word[:i]
                if prefix in word_set:
                    word_to_extensions[prefix].append(word)
        
        # DFS to find longest path
        def dfs_longest_path(word: str, visited: Set[str]) -> str:
            """Find longest path starting from word"""
            longest = word
            visited.add(word)
            
            for extension in word_to_extensions[word]:
                if extension not in visited:
                    candidate = dfs_longest_path(extension, visited)
                    if len(candidate) > len(longest) or (len(candidate) == len(longest) and candidate < longest):
                        longest = candidate
            
            visited.remove(word)
            return longest
        
        # Try starting from each single character word
        overall_longest = ""
        
        for word in words:
            if len(word) == 1:
                candidate = dfs_longest_path(word, set())
                if len(candidate) > len(overall_longest) or (len(candidate) == len(overall_longest) and candidate < overall_longest):
                    overall_longest = candidate
        
        return overall_longest


def test_basic_cases():
    """Test basic functionality"""
    print("=== Testing Basic Cases ===")
    
    solution = Solution()
    
    test_cases = [
        # LeetCode examples
        (["w","wo","wor","worl","world"], "world"),
        (["a","banana","app","appl","ap","apply","application"], "application"),
        (["m","mo","moc","moch","mocha","l","la","lat","latt","latte","c","ca","cat"], "mocha"),
        
        # Edge cases
        ([], ""),
        (["a"], "a"),
        (["ab", "a"], "ab"),
        (["abc", "ab"], ""),  # "ab" not in dictionary
        
        # Multiple same length
        (["cat", "cats", "dog", "dogs"], "cats"),  # lexicographically first
        
        # Complex buildable chains
        (["a", "aa", "aaa", "aaaa"], "aaaa"),
        (["b", "ba", "bac", "back"], "back"),
    ]
    
    approaches = [
        ("Trie DFS", solution.longestWord1),
        ("BFS Level", solution.longestWord2),
        ("Set-based", solution.longestWord3),
        ("Sort+Build", solution.longestWord4),
        ("DP Memoization", solution.longestWord5),
        ("Trie Bottom-up", solution.longestWord6),
        ("Graph-based", solution.longestWord7),
    ]
    
    for i, (words, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: words={words}")
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
    
    words = ["w", "wo", "wor", "worl", "world"]
    
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
    for i in range(1, len(target) + 1):
        prefix = target[:i]
        exists = prefix in word_set
        status = "✓" if exists else "✗"
        print(f"  Step {i}: '{prefix}' {'exists' if exists else 'missing'} {status}")
        
        if not exists and i < len(target):
            print(f"    Cannot continue building - '{prefix}' not in dictionary")
            break
    
    # Show buildable words
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
    longest = max(buildable_words, key=lambda x: (len(x), -ord(x[0]))) if buildable_words else ""
    print(f"\nLongest buildable word: '{longest}'")


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
            for _ in range(random.randint(1, max_length - 1)):
                if random.random() < 0.7:  # 70% chance to continue chain
                    current += random.choice(string.ascii_lowercase)
                    words.append(current)
                else:
                    break
        
        # Add some random words
        for _ in range(base_count):
            length = random.randint(1, max_length)
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.append(word)
        
        return list(set(words))  # Remove duplicates
    
    test_scenarios = [
        ("Small", generate_words_with_chains(10, 5)),
        ("Medium", generate_words_with_chains(30, 8)),
        ("Large", generate_words_with_chains(100, 12)),
    ]
    
    approaches = [
        ("Trie DFS", solution.longestWord1),
        ("BFS Level", solution.longestWord2),
        ("Set-based", solution.longestWord3),
        ("Sort+Build", solution.longestWord4),
        ("DP Memo", solution.longestWord5),
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


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solution = Solution()
    
    # Application 1: Word learning progression
    print("1. Language Learning Progression:")
    
    learned_words = [
        "I", "It", "Its", 
        "a", "an", "and",
        "c", "ca", "cat", "cats",
        "d", "do", "dog", "dogs"
    ]
    
    print(f"   Learned words: {learned_words}")
    
    longest_learnable = solution.longestWord1(learned_words)
    print(f"   Longest word that can be learned step-by-step: '{longest_learnable}'")
    
    # Show learning path
    if longest_learnable:
        path = []
        for i in range(1, len(longest_learnable) + 1):
            prefix = longest_learnable[:i]
            if prefix in learned_words:
                path.append(prefix)
        print(f"   Learning path: {' → '.join(path)}")
    
    # Application 2: Skill building progression
    print(f"\n2. Programming Skill Progression:")
    
    skills = [
        "code", "coding",
        "debug", "debugging", 
        "test", "testing",
        "deploy", "deployment",
        "p", "pr", "pro", "prog", "progr", "progra", "program", "programming"
    ]
    
    print(f"   Available skills: {len(skills)} skills")
    
    advanced_skill = solution.longestWord1(skills)
    print(f"   Most advanced skill achievable: '{advanced_skill}'")
    
    # Application 3: Password strength progression
    print(f"\n3. Password Complexity Building:")
    
    password_components = [
        "a", "ab", "abc",
        "1", "12", "123",
        "A", "AB", "ABC",
        "!", "!@", "!@#"
    ]
    
    print(f"   Password components: {password_components}")
    
    strongest_password = solution.longestWord1(password_components)
    print(f"   Strongest buildable password: '{strongest_password}'")


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
    ]
    
    for words, description in edge_cases:
        print(f"\n{description}:")
        print(f"  Words: {words}")
        
        try:
            result = solution.longestWord1(words)
            print(f"  Result: '{result}'")
            
            # Show reasoning
            if result:
                print(f"  Reason: '{result}' can be built incrementally")
            else:
                print(f"  Reason: No word can be built incrementally")
        except Exception as e:
            print(f"  Error: {e}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Trie DFS",
         "Time: O(sum of word lengths) - build trie + DFS",
         "Space: O(sum of word lengths) - trie storage"),
        
        ("BFS Level-by-Level",
         "Time: O(sum of word lengths) - build trie + BFS",
         "Space: O(sum of word lengths) - trie + queue"),
        
        ("Set-based Check",
         "Time: O(sum of word lengths²) - check all prefixes",
         "Space: O(number of words) - set storage"),
        
        ("Sort and Build",
         "Time: O(n*log(n) + sum of word lengths)",
         "Space: O(number of words) - buildable set"),
        
        ("DP with Memoization",
         "Time: O(sum of word lengths) - memoized checks",
         "Space: O(number of words) - memoization table"),
        
        ("Graph-based",
         "Time: O(sum of word lengths + V + E)",
         "Space: O(V + E) - adjacency list"),
    ]
    
    print("Approach Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nWhere:")
    print(f"  n = number of words")
    print(f"  V = vertices (words)")
    print(f"  E = edges (buildable relationships)")
    
    print(f"\nRecommendations:")
    print(f"  • Use Trie DFS for optimal time complexity")
    print(f"  • Use Set-based for simple implementation")
    print(f"  • Use Sort+Build for memory-constrained environments")
    print(f"  • Use DP for overlapping subproblems scenarios")


if __name__ == "__main__":
    test_basic_cases()
    demonstrate_trie_construction()
    demonstrate_building_process()
    benchmark_approaches()
    demonstrate_real_world_applications()
    test_edge_cases()
    analyze_complexity()

"""
720. Longest Word in Dictionary demonstrates multiple approaches for finding buildable words:

1. Trie DFS - Build trie and use depth-first search to find longest buildable word
2. BFS Level-by-Level - Process trie nodes level by level using breadth-first search
3. Set-based Check - Use set lookup to verify all prefixes exist for each word
4. Sort and Build - Sort words and build incrementally with set tracking
5. DP with Memoization - Cache buildability results to avoid recomputation
6. Trie Bottom-up - Mark buildable nodes in trie using bottom-up approach
7. Graph-based - Model as graph with "can extend" relationships

Each approach offers different trade-offs between implementation complexity,
time efficiency, and space usage for incremental word building problems.
"""

