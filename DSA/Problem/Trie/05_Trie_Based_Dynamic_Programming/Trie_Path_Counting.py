"""
Trie Path Counting - Multiple Approaches
Difficulty: Easy

Count the number of paths in a trie from root to any node, or from root to terminal nodes.
This problem involves various path counting scenarios in trie data structures.

Problem Variations:
1. Count all paths from root to terminal nodes (complete words)
2. Count all paths from root to any node
3. Count paths of specific length
4. Count paths with specific properties
5. Dynamic programming on trie paths

Applications:
- Word counting in dictionaries
- Path analysis in tree structures  
- Combinatorial counting problems
- Dynamic programming on trees
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import time

class TrieNode:
    """Basic trie node for path counting"""
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word_count = 0  # Number of times this word was inserted
        self.path_count = 0  # Number of paths through this node

class TriePathCounter:
    
    def __init__(self):
        """Initialize trie for path counting"""
        self.root = TrieNode()
        self.total_words = 0
    
    def insert(self, word: str, count: int = 1) -> None:
        """
        Insert word into trie with given count.
        
        Time: O(len(word))
        Space: O(len(word)) worst case
        """
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.path_count += count
        
        node.is_word = True
        node.word_count += count
        self.total_words += count
    
    def count_all_paths_to_words(self) -> int:
        """
        Approach 1: Count All Paths to Terminal Nodes
        
        Count paths from root to all complete words.
        
        Time: O(total characters in all words)
        Space: O(height of trie)
        """
        def dfs(node: TrieNode) -> int:
            count = 0
            
            if node.is_word:
                count += node.word_count
            
            for child in node.children.values():
                count += dfs(child)
            
            return count
        
        return dfs(self.root)
    
    def count_all_paths_to_nodes(self) -> int:
        """
        Approach 2: Count All Paths to Any Node
        
        Count paths from root to every node in the trie.
        
        Time: O(number of nodes)
        Space: O(height of trie)
        """
        def dfs(node: TrieNode) -> int:
            count = 1  # Count current node
            
            for child in node.children.values():
                count += dfs(child)
            
            return count
        
        return dfs(self.root) - 1  # Exclude root
    
    def count_paths_of_length(self, target_length: int) -> int:
        """
        Approach 3: Count Paths of Specific Length
        
        Count paths from root with exactly target_length edges.
        
        Time: O(nodes at target depth)
        Space: O(height of trie)
        """
        def dfs(node: TrieNode, current_length: int) -> int:
            if current_length == target_length:
                return 1
            
            if current_length > target_length:
                return 0
            
            count = 0
            for child in node.children.values():
                count += dfs(child, current_length + 1)
            
            return count
        
        return dfs(self.root, 0)
    
    def count_paths_with_prefix(self, prefix: str) -> int:
        """
        Approach 4: Count Paths Starting with Prefix
        
        Count all paths that start with given prefix.
        
        Time: O(len(prefix) + nodes in subtree)
        Space: O(height of trie)
        """
        # Navigate to prefix node
        node = self.root
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        
        # Count all paths in subtree
        def dfs(node: TrieNode) -> int:
            count = 0
            
            if node.is_word:
                count += node.word_count
            
            for child in node.children.values():
                count += dfs(child)
            
            return count
        
        return dfs(node)
    
    def count_paths_with_pattern(self, pattern: str, wildcard: str = '*') -> int:
        """
        Approach 5: Count Paths Matching Pattern
        
        Count paths that match a pattern with wildcards.
        
        Time: O(nodes * pattern_length)
        Space: O(height of trie * pattern_length)
        """
        def dfs(node: TrieNode, pattern_idx: int) -> int:
            if pattern_idx == len(pattern):
                return node.word_count if node.is_word else 0
            
            char = pattern[pattern_idx]
            count = 0
            
            if char == wildcard:
                # Wildcard matches any character
                for child in node.children.values():
                    count += dfs(child, pattern_idx + 1)
            elif char in node.children:
                # Exact character match
                count += dfs(node.children[char], pattern_idx + 1)
            
            return count
        
        return dfs(self.root, 0)


class AdvancedTriePathCounter:
    """Advanced trie with enhanced path counting capabilities"""
    
    def __init__(self):
        self.root = TrieNode()
        self.total_nodes = 0
    
    def insert_with_metadata(self, word: str, metadata: dict = None) -> None:
        """Insert word with additional metadata"""
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
                self.total_nodes += 1
            node = node.children[char]
        
        node.is_word = True
        node.word_count += 1
        
        # Store metadata if provided
        if metadata:
            if not hasattr(node, 'metadata'):
                node.metadata = []
            node.metadata.append(metadata)
    
    def count_paths_by_length_distribution(self) -> Dict[int, int]:
        """
        Approach 6: Count Paths by Length Distribution
        
        Return distribution of path lengths to terminal nodes.
        
        Time: O(total characters)
        Space: O(unique path lengths)
        """
        distribution = defaultdict(int)
        
        def dfs(node: TrieNode, depth: int) -> None:
            if node.is_word:
                distribution[depth] += node.word_count
            
            for child in node.children.values():
                dfs(child, depth + 1)
        
        dfs(self.root, 0)
        return dict(distribution)
    
    def count_paths_with_dp(self) -> Dict[str, int]:
        """
        Approach 7: Dynamic Programming on Trie Paths
        
        Use DP to count various path properties efficiently.
        
        Time: O(nodes)
        Space: O(nodes)
        """
        # Memoization for different counting problems
        memo_word_paths = {}  # node -> number of word paths in subtree
        memo_total_paths = {}  # node -> total paths in subtree
        memo_max_depth = {}   # node -> maximum depth in subtree
        
        def count_word_paths(node: TrieNode) -> int:
            if id(node) in memo_word_paths:
                return memo_word_paths[id(node)]
            
            count = node.word_count if node.is_word else 0
            
            for child in node.children.values():
                count += count_word_paths(child)
            
            memo_word_paths[id(node)] = count
            return count
        
        def count_total_paths(node: TrieNode) -> int:
            if id(node) in memo_total_paths:
                return memo_total_paths[id(node)]
            
            count = 1  # Current node
            
            for child in node.children.values():
                count += count_total_paths(child)
            
            memo_total_paths[id(node)] = count
            return count
        
        def max_depth(node: TrieNode) -> int:
            if id(node) in memo_max_depth:
                return memo_max_depth[id(node)]
            
            if not node.children:
                memo_max_depth[id(node)] = 0
                return 0
            
            depth = max(max_depth(child) for child in node.children.values()) + 1
            memo_max_depth[id(node)] = depth
            return depth
        
        return {
            'word_paths': count_word_paths(self.root),
            'total_paths': count_total_paths(self.root) - 1,  # Exclude root
            'max_depth': max_depth(self.root)
        }
    
    def count_paths_with_constraints(self, min_length: int = 0, max_length: int = float('inf'),
                                   required_chars: Set[str] = None, forbidden_chars: Set[str] = None) -> int:
        """
        Approach 8: Count Paths with Multiple Constraints
        
        Count paths satisfying multiple constraints simultaneously.
        
        Time: O(nodes * constraint_checking_time)
        Space: O(height of trie)
        """
        required_chars = required_chars or set()
        forbidden_chars = forbidden_chars or set()
        
        def dfs(node: TrieNode, depth: int, seen_chars: Set[str], path: str) -> int:
            # Check constraints
            if depth > max_length:
                return 0
            
            if forbidden_chars & seen_chars:
                return 0
            
            count = 0
            
            if (node.is_word and 
                depth >= min_length and 
                required_chars.issubset(seen_chars)):
                count += node.word_count
            
            # Continue to children
            for char, child in node.children.items():
                new_seen = seen_chars | {char}
                count += dfs(child, depth + 1, new_seen, path + char)
            
            return count
        
        return dfs(self.root, 0, set(), "")


class OptimizedTriePathCounter:
    """Optimized trie with precomputed path statistics"""
    
    def __init__(self):
        self.root = TrieNode()
        self._precomputed = False
    
    def insert(self, word: str) -> None:
        """Insert word and mark for recomputation"""
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_word = True
        node.word_count += 1
        self._precomputed = False
    
    def _precompute_statistics(self) -> None:
        """
        Approach 9: Precompute All Path Statistics
        
        Precompute various statistics for O(1) queries.
        
        Time: O(nodes) for precomputation
        Space: O(nodes) for statistics storage
        """
        if self._precomputed:
            return
        
        # Add statistics to each node
        def postorder_dfs(node: TrieNode) -> Dict[str, int]:
            stats = {
                'word_paths': node.word_count if node.is_word else 0,
                'total_nodes': 1,
                'max_depth': 0,
                'min_depth': float('inf') if node.is_word else 0,
                'leaf_count': 1 if not node.children else 0
            }
            
            for child in node.children.values():
                child_stats = postorder_dfs(child)
                
                stats['word_paths'] += child_stats['word_paths']
                stats['total_nodes'] += child_stats['total_nodes']
                stats['max_depth'] = max(stats['max_depth'], child_stats['max_depth'] + 1)
                stats['leaf_count'] += child_stats['leaf_count']
                
                if child_stats['min_depth'] != float('inf'):
                    stats['min_depth'] = min(stats['min_depth'], child_stats['min_depth'] + 1)
            
            if stats['min_depth'] == float('inf'):
                stats['min_depth'] = 0
            
            # Store statistics in node
            node.stats = stats
            return stats
        
        postorder_dfs(self.root)
        self._precomputed = True
    
    def get_path_statistics(self) -> Dict[str, int]:
        """Get precomputed path statistics"""
        self._precompute_statistics()
        return self.root.stats
    
    def get_subtree_statistics(self, prefix: str) -> Dict[str, int]:
        """Get statistics for subtree rooted at prefix"""
        self._precompute_statistics()
        
        # Navigate to prefix
        node = self.root
        for char in prefix:
            if char not in node.children:
                return {'word_paths': 0, 'total_nodes': 0, 'max_depth': 0, 'min_depth': 0, 'leaf_count': 0}
            node = node.children[char]
        
        return node.stats


def test_basic_path_counting():
    """Test basic path counting functionality"""
    print("=== Testing Basic Path Counting ===")
    
    # Create test trie
    trie = TriePathCounter()
    
    words = ["cat", "car", "card", "care", "careful", "cats", "dog", "do"]
    
    print("Inserting words:")
    for word in words:
        trie.insert(word)
        print(f"  Inserted: '{word}'")
    
    print(f"\nPath counting results:")
    
    # Test different counting methods
    word_paths = trie.count_all_paths_to_words()
    print(f"  Paths to words: {word_paths}")
    
    all_paths = trie.count_all_paths_to_nodes()
    print(f"  Paths to all nodes: {all_paths}")
    
    # Test length-specific counting
    for length in [1, 2, 3, 4]:
        count = trie.count_paths_of_length(length)
        print(f"  Paths of length {length}: {count}")
    
    # Test prefix counting
    prefixes = ["ca", "car", "do", "x"]
    for prefix in prefixes:
        count = trie.count_paths_with_prefix(prefix)
        print(f"  Paths with prefix '{prefix}': {count}")


def test_pattern_matching():
    """Test pattern matching in path counting"""
    print("\n=== Testing Pattern Matching ===")
    
    trie = TriePathCounter()
    
    words = ["cat", "car", "bat", "bar", "rat", "rag"]
    
    for word in words:
        trie.insert(word)
    
    print(f"Words: {words}")
    
    # Test wildcard patterns
    patterns = ["*at", "?ar", "ca*", "*a*", "***"]
    
    for pattern in patterns:
        count = trie.count_paths_with_pattern(pattern, '*')
        print(f"  Pattern '{pattern}': {count} matches")
        
        # Show which words match (for verification)
        matching_words = [word for word in words if matches_pattern(word, pattern)]
        print(f"    Matching words: {matching_words}")


def matches_pattern(word: str, pattern: str) -> bool:
    """Helper function to check if word matches pattern"""
    if len(word) != len(pattern):
        return False
    
    for w_char, p_char in zip(word, pattern):
        if p_char != '*' and p_char != '?' and w_char != p_char:
            return False
    
    return True


def test_advanced_counting():
    """Test advanced counting methods"""
    print("\n=== Testing Advanced Counting ===")
    
    trie = AdvancedTriePathCounter()
    
    # Insert words with metadata
    words_with_metadata = [
        ("apple", {"category": "fruit", "length": 5}),
        ("app", {"category": "software", "length": 3}),
        ("application", {"category": "software", "length": 11}),
        ("banana", {"category": "fruit", "length": 6}),
        ("band", {"category": "music", "length": 4}),
    ]
    
    for word, metadata in words_with_metadata:
        trie.insert_with_metadata(word, metadata)
    
    # Test length distribution
    distribution = trie.count_paths_by_length_distribution()
    print(f"Length distribution: {distribution}")
    
    # Test DP-based counting
    dp_stats = trie.count_paths_with_dp()
    print(f"DP statistics: {dp_stats}")
    
    # Test constraint-based counting
    constraints_tests = [
        {"min_length": 4, "max_length": 6},
        {"required_chars": {'a', 'p'}},
        {"forbidden_chars": {'x', 'z'}},
        {"min_length": 3, "required_chars": {'a'}, "forbidden_chars": {'z'}},
    ]
    
    for constraints in constraints_tests:
        count = trie.count_paths_with_constraints(**constraints)
        print(f"  Constraints {constraints}: {count} paths")


def test_optimized_counting():
    """Test optimized precomputed counting"""
    print("\n=== Testing Optimized Counting ===")
    
    trie = OptimizedTriePathCounter()
    
    words = ["programming", "program", "pro", "progress", "project", "problem"]
    
    print("Building trie:")
    for word in words:
        trie.insert(word)
        print(f"  Inserted: '{word}'")
    
    # Test overall statistics
    stats = trie.get_path_statistics()
    print(f"\nOverall statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test subtree statistics
    prefixes = ["pro", "prog", "proj", "xyz"]
    
    print(f"\nSubtree statistics:")
    for prefix in prefixes:
        subtree_stats = trie.get_subtree_statistics(prefix)
        print(f"  Prefix '{prefix}': {subtree_stats}")


def demonstrate_dp_optimization():
    """Demonstrate dynamic programming optimization"""
    print("\n=== DP Optimization Demo ===")
    
    print("Problem: Count paths in trie where each path represents a valid word")
    print("Challenge: Avoid recomputing statistics for overlapping subtrees")
    
    trie = AdvancedTriePathCounter()
    
    # Create trie with overlapping prefixes
    words = ["test", "testing", "tester", "tests", "tea", "teach", "teacher"]
    
    for word in words:
        trie.insert_with_metadata(word)
    
    print(f"\nWords: {words}")
    print("Notice common prefixes: 'te', 'test', 'teach'")
    
    # Show how DP avoids recomputation
    print(f"\nWithout DP: would recompute statistics for each subtree multiple times")
    print(f"With DP: compute each subtree statistics only once")
    
    dp_stats = trie.count_paths_with_dp()
    print(f"\nDP Results: {dp_stats}")
    
    # Show the efficiency gain
    print(f"\nEfficiency analysis:")
    print(f"  Total nodes: {dp_stats['total_paths']}")
    print(f"  Word paths: {dp_stats['word_paths']}")
    print(f"  Max depth: {dp_stats['max_depth']}")
    print(f"  Each node's statistics computed exactly once using memoization")


def benchmark_counting_approaches():
    """Benchmark different counting approaches"""
    print("\n=== Benchmarking Counting Approaches ===")
    
    import random
    import string
    
    # Generate test data
    def generate_words(count: int, max_length: int) -> List[str]:
        words = set()
        while len(words) < count:
            length = random.randint(3, max_length)
            word = ''.join(random.choices(string.ascii_lowercase[:5], k=length))
            words.add(word)
        return list(words)
    
    test_scenarios = [
        ("Small", generate_words(50, 6)),
        ("Medium", generate_words(200, 8)), 
        ("Large", generate_words(500, 10)),
    ]
    
    approaches = [
        ("Basic Trie", TriePathCounter),
        ("Advanced Trie", AdvancedTriePathCounter),
        ("Optimized Trie", OptimizedTriePathCounter),
    ]
    
    for scenario_name, words in test_scenarios:
        print(f"\n--- {scenario_name} Dataset ({len(words)} words) ---")
        
        for approach_name, TrieClass in approaches:
            start_time = time.time()
            
            # Build trie
            if TrieClass == AdvancedTriePathCounter:
                trie = TrieClass()
                for word in words:
                    trie.insert_with_metadata(word)
                
                # Test DP counting
                stats = trie.count_paths_with_dp()
                result = stats['word_paths']
            
            elif TrieClass == OptimizedTriePathCounter:
                trie = TrieClass()
                for word in words:
                    trie.insert(word)
                
                # Test optimized counting
                stats = trie.get_path_statistics()
                result = stats['word_paths']
            
            else:
                trie = TrieClass()
                for word in words:
                    trie.insert(word)
                
                # Test basic counting
                result = trie.count_all_paths_to_words()
            
            end_time = time.time()
            
            execution_time = (end_time - start_time) * 1000
            print(f"  {approach_name:15}: {result:4} paths in {execution_time:6.2f}ms")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Dictionary analysis
    print("1. Dictionary Word Count Analysis:")
    
    dictionary_words = [
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "its", "may", "new", "now", "old", "see", "two", "who", "boy", "did", "man", "men", "oil", "run", "sat", "say", "she", "too", "use"
    ]
    
    trie = AdvancedTriePathCounter()
    for word in dictionary_words:
        trie.insert_with_metadata(word, {"type": "common_word"})
    
    distribution = trie.count_paths_by_length_distribution()
    print(f"   Word length distribution: {distribution}")
    
    dp_stats = trie.count_paths_with_dp()
    print(f"   Total dictionary entries: {dp_stats['word_paths']}")
    print(f"   Trie compression ratio: {dp_stats['word_paths'] / dp_stats['total_paths']:.2%}")
    
    # Application 2: Auto-completion efficiency
    print(f"\n2. Auto-completion Efficiency Analysis:")
    
    prefixes = ["th", "an", "fo", "ca", "ma"]
    
    for prefix in prefixes:
        count = trie.count_paths_with_constraints(min_length=len(prefix))
        print(f"   Prefix '{prefix}': {count} possible completions")
    
    # Application 3: Text processing statistics
    print(f"\n3. Text Processing Statistics:")
    
    # Analyze character frequency in paths
    char_constraints = [
        ({'a'}, "containing 'a'"),
        ({'e'}, "containing 'e'"),
        ({'a', 'e'}, "containing both 'a' and 'e'"),
        (set(), "all words")
    ]
    
    for required_chars, description in char_constraints:
        count = trie.count_paths_with_constraints(required_chars=required_chars)
        print(f"   Words {description}: {count}")
    
    # Application 4: Performance monitoring
    print(f"\n4. Trie Performance Metrics:")
    
    optimized_trie = OptimizedTriePathCounter()
    for word in dictionary_words:
        optimized_trie.insert(word)
    
    stats = optimized_trie.get_path_statistics()
    
    print(f"   Total nodes: {stats['total_nodes']}")
    print(f"   Max depth: {stats['max_depth']}")
    print(f"   Average branching factor: {(stats['total_nodes'] - 1) / max(1, stats['total_nodes'] - stats['leaf_count']):.2f}")
    print(f"   Space efficiency: {stats['word_paths'] / stats['total_nodes']:.2%} words per node")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    edge_cases = [
        # Empty trie
        ([], "Empty trie"),
        
        # Single word
        (["hello"], "Single word"),
        
        # All same length
        (["cat", "dog", "rat", "bat"], "All same length"),
        
        # Nested words
        (["a", "ab", "abc", "abcd"], "Nested words"),
        
        # No common prefixes
        (["apple", "banana", "cherry"], "No common prefixes"),
        
        # Very long words
        (["a" * 50, "b" * 50], "Very long words"),
        
        # Single character words
        (["a", "b", "c", "d", "e"], "Single characters"),
    ]
    
    for words, description in edge_cases:
        print(f"\n{description}: {words}")
        
        try:
            trie = TriePathCounter()
            for word in words:
                trie.insert(word)
            
            word_paths = trie.count_all_paths_to_words()
            all_paths = trie.count_all_paths_to_nodes()
            
            print(f"  Word paths: {word_paths}")
            print(f"  All paths: {all_paths}")
            
            if words:
                max_len = max(len(word) for word in words)
                for length in range(1, min(max_len + 1, 4)):
                    count = trie.count_paths_of_length(length)
                    print(f"  Paths of length {length}: {count}")
        
        except Exception as e:
            print(f"  Error: {e}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Basic Path Counting",
         "Time: O(nodes) for DFS traversal",
         "Space: O(height) for recursion stack"),
        
        ("Length-Specific Counting",
         "Time: O(nodes at target depth)",
         "Space: O(height) for recursion"),
        
        ("Prefix-Based Counting", 
         "Time: O(prefix_length + subtree_nodes)",
         "Space: O(height) for recursion"),
        
        ("Pattern Matching",
         "Time: O(nodes × pattern_length)",
         "Space: O(height × pattern_length)"),
        
        ("DP-Based Counting",
         "Time: O(nodes) with memoization",
         "Space: O(nodes) for memoization"),
        
        ("Constraint-Based Counting",
         "Time: O(nodes × constraint_checking)",
         "Space: O(height + constraint_state)"),
        
        ("Precomputed Statistics",
         "Time: O(nodes) for precomputation, O(1) for queries",
         "Space: O(nodes) for storing statistics"),
    ]
    
    print("Approach Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nOptimization Strategies:")
    print(f"  • Memoization: Cache results for repeated subproblems")
    print(f"  • Precomputation: Calculate statistics once, query many times")
    print(f"  • Early termination: Stop when constraints cannot be satisfied")
    print(f"  • Constraint filtering: Eliminate impossible paths early")
    
    print(f"\nTrade-offs:")
    print(f"  • Memory vs Speed: Precomputation uses more memory for faster queries")
    print(f"  • Flexibility vs Performance: Generic solutions vs specialized optimizations")
    print(f"  • Build time vs Query time: One-time setup cost vs repeated query cost")
    
    print(f"\nRecommendations:")
    print(f"  • Use basic counting for simple, one-time queries")
    print(f"  • Use DP approach for complex constraints with overlap")
    print(f"  • Use precomputed statistics for repeated queries")
    print(f"  • Consider constraint complexity when choosing approach")


if __name__ == "__main__":
    test_basic_path_counting()
    test_pattern_matching()
    test_advanced_counting()
    test_optimized_counting()
    demonstrate_dp_optimization()
    benchmark_counting_approaches()
    demonstrate_real_world_applications()
    test_edge_cases()
    analyze_complexity()

"""
Trie Path Counting demonstrates comprehensive path counting approaches in trie data structures:

1. Basic Path Counting - Count paths to terminal nodes and all nodes
2. Length-Specific Counting - Count paths of exact lengths  
3. Prefix-Based Counting - Count paths with specific prefixes
4. Pattern Matching - Count paths matching wildcard patterns
5. Advanced Constraint Counting - Multiple simultaneous constraints
6. DP-Based Optimization - Memoization for efficient recomputation
7. Precomputed Statistics - O(1) queries after O(n) preprocessing

Key concepts:
- Dynamic programming on tree structures
- Path enumeration and counting algorithms
- Constraint satisfaction in tree traversal
- Memory-time trade-offs in optimization
- Statistical analysis of trie structures

Real-world applications:
- Dictionary analysis and statistics
- Auto-completion efficiency measurement
- Text processing and character frequency analysis
- Performance monitoring of trie-based systems

Each approach demonstrates different algorithmic techniques for counting
and analyzing paths in trie data structures with various constraints.
"""
