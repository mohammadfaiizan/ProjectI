"""
Basic Autocomplete - Multiple Approaches
Difficulty: Easy

Implement a basic autocomplete system that suggests words based on prefix input.
The system should support adding words to dictionary and getting suggestions.
"""

from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict, deque
import heapq
import bisect

class TrieNode:
    """Basic trie node for autocomplete"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.word = ""
        self.frequency = 0

class BasicAutocomplete:
    
    def __init__(self):
        self.words = []
        self.trie_root = None
        
    def approach1_simple_list(self, dictionary: List[str], prefix: str, max_suggestions: int = 5) -> List[str]:
        """
        Approach 1: Simple List Filtering
        
        Filter dictionary words that start with prefix.
        
        Time: O(n * m) where n=words, m=average word length
        Space: O(1)
        """
        suggestions = []
        
        for word in dictionary:
            if word.startswith(prefix):
                suggestions.append(word)
                if len(suggestions) >= max_suggestions:
                    break
        
        return sorted(suggestions)[:max_suggestions]
    
    def approach2_binary_search(self, dictionary: List[str], prefix: str, max_suggestions: int = 5) -> List[str]:
        """
        Approach 2: Binary Search on Sorted Dictionary
        
        Use binary search to find prefix range efficiently.
        
        Time: O(log n + k) where k=number of matches
        Space: O(1)
        """
        if not dictionary:
            return []
        
        # Ensure dictionary is sorted
        sorted_dict = sorted(dictionary)
        
        # Find first position where prefix could be inserted
        left = bisect.bisect_left(sorted_dict, prefix)
        
        suggestions = []
        
        # Collect words starting with prefix
        for i in range(left, len(sorted_dict)):
            if sorted_dict[i].startswith(prefix):
                suggestions.append(sorted_dict[i])
                if len(suggestions) >= max_suggestions:
                    break
            else:
                break  # No more matches possible
        
        return suggestions
    
    def approach3_trie_based(self, dictionary: List[str], prefix: str, max_suggestions: int = 5) -> List[str]:
        """
        Approach 3: Trie-based Autocomplete
        
        Build trie and traverse to find suggestions.
        
        Time: O(sum of word lengths) for build + O(prefix_length + k) for query
        Space: O(sum of word lengths)
        """
        # Build trie
        root = TrieNode()
        
        for word in dictionary:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
            node.word = word
        
        # Navigate to prefix node
        node = root
        for char in prefix:
            if char not in node.children:
                return []  # Prefix not found
            node = node.children[char]
        
        # Collect suggestions using DFS
        suggestions = []
        
        def dfs(current_node: TrieNode):
            if len(suggestions) >= max_suggestions:
                return
            
            if current_node.is_end:
                suggestions.append(current_node.word)
            
            # Traverse children in alphabetical order
            for char in sorted(current_node.children.keys()):
                dfs(current_node.children[char])
        
        dfs(node)
        return suggestions
    
    def approach4_frequency_based(self, dictionary: List[Tuple[str, int]], prefix: str, max_suggestions: int = 5) -> List[str]:
        """
        Approach 4: Frequency-based Suggestions
        
        Rank suggestions by frequency/popularity.
        
        Time: O(n log k) where k=max_suggestions
        Space: O(n)
        """
        # Filter words with prefix and sort by frequency
        matching_words = []
        
        for word, frequency in dictionary:
            if word.startswith(prefix):
                matching_words.append((word, frequency))
        
        # Sort by frequency (descending) then alphabetically
        matching_words.sort(key=lambda x: (-x[1], x[0]))
        
        return [word for word, _ in matching_words[:max_suggestions]]
    
    def approach5_heap_based(self, dictionary: List[Tuple[str, int]], prefix: str, max_suggestions: int = 5) -> List[str]:
        """
        Approach 5: Heap-based Top-K Selection
        
        Use min-heap to maintain top-k suggestions by frequency.
        
        Time: O(n log k)
        Space: O(k)
        """
        # Use min-heap to track top k suggestions
        heap = []
        
        for word, frequency in dictionary:
            if word.startswith(prefix):
                if len(heap) < max_suggestions:
                    heapq.heappush(heap, (frequency, word))
                elif frequency > heap[0][0]:
                    heapq.heapreplace(heap, (frequency, word))
        
        # Extract results and sort
        suggestions = []
        while heap:
            frequency, word = heapq.heappop(heap)
            suggestions.append(word)
        
        # Sort by frequency (desc) then alphabetically
        suggestions.sort(key=lambda x: (-dict(dictionary)[x], x))
        
        return suggestions
    
    def approach6_prefix_tree_with_cache(self, dictionary: List[str], prefix: str, max_suggestions: int = 5) -> List[str]:
        """
        Approach 6: Trie with Cached Suggestions
        
        Pre-compute and cache suggestions at each node.
        
        Time: O(sum of word lengths) for build + O(prefix_length) for query
        Space: O(sum of word lengths * max_suggestions)
        """
        # Build trie with cached suggestions
        root = TrieNode()
        
        # Sort dictionary first for consistent ordering
        sorted_dict = sorted(dictionary)
        
        for word in sorted_dict:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                
                # Add word to suggestions if not already full
                if not hasattr(node, 'suggestions'):
                    node.suggestions = []
                
                if len(node.suggestions) < max_suggestions:
                    node.suggestions.append(word)
            
            node.is_end = True
            node.word = word
        
        # Navigate to prefix and return cached suggestions
        node = root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        return getattr(node, 'suggestions', [])
    
    def approach7_fuzzy_autocomplete(self, dictionary: List[str], prefix: str, max_suggestions: int = 5, max_distance: int = 1) -> List[str]:
        """
        Approach 7: Fuzzy Autocomplete with Edit Distance
        
        Allow suggestions with small edit distance from prefix.
        
        Time: O(n * m * k) where k=max_distance
        Space: O(m * k) for DP table
        """
        def edit_distance(s1: str, s2: str) -> int:
            """Calculate minimum edit distance"""
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            # Initialize base cases
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            
            # Fill DP table
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            
            return dp[m][n]
        
        suggestions = []
        
        for word in dictionary:
            # Check if word starts with similar prefix
            word_prefix = word[:len(prefix)] if len(word) >= len(prefix) else word
            distance = edit_distance(prefix, word_prefix)
            
            if distance <= max_distance:
                suggestions.append((word, distance))
        
        # Sort by distance then alphabetically
        suggestions.sort(key=lambda x: (x[1], x[0]))
        
        return [word for word, _ in suggestions[:max_suggestions]]


class AutocompleteSystem:
    """Complete autocomplete system with all features"""
    
    def __init__(self):
        self.root = TrieNode()
        self.word_frequencies = {}
    
    def add_word(self, word: str, frequency: int = 1) -> None:
        """Add word to autocomplete system"""
        # Add to trie
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end = True
        node.word = word
        node.frequency = frequency
        
        # Update frequency map
        self.word_frequencies[word] = frequency
    
    def get_suggestions(self, prefix: str, max_suggestions: int = 5, sort_by_frequency: bool = True) -> List[str]:
        """Get autocomplete suggestions for prefix"""
        # Navigate to prefix node
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Collect all words with DFS
        suggestions = []
        
        def dfs(current_node: TrieNode):
            if current_node.is_end:
                suggestions.append((current_node.word, current_node.frequency))
            
            for child in current_node.children.values():
                dfs(child)
        
        dfs(node)
        
        # Sort suggestions
        if sort_by_frequency:
            suggestions.sort(key=lambda x: (-x[1], x[0]))  # Frequency desc, then alphabetical
        else:
            suggestions.sort(key=lambda x: x[0])  # Alphabetical only
        
        return [word for word, _ in suggestions[:max_suggestions]]
    
    def update_frequency(self, word: str, new_frequency: int) -> None:
        """Update frequency of existing word"""
        if word in self.word_frequencies:
            self.word_frequencies[word] = new_frequency
            
            # Update in trie
            node = self.root
            for char in word:
                if char in node.children:
                    node = node.children[char]
                else:
                    return  # Word not found
            
            if node.is_end:
                node.frequency = new_frequency


def test_basic_autocomplete():
    """Test basic autocomplete functionality"""
    print("=== Testing Basic Autocomplete ===")
    
    autocomplete = BasicAutocomplete()
    
    # Test dictionary
    dictionary = ["apple", "application", "apply", "banana", "band", "bandana", "can", "cat", "car"]
    prefix = "app"
    
    print(f"Dictionary: {dictionary}")
    print(f"Prefix: '{prefix}'")
    
    approaches = [
        ("Simple List", autocomplete.approach1_simple_list),
        ("Binary Search", autocomplete.approach2_binary_search),
        ("Trie-based", autocomplete.approach3_trie_based),
        ("Cached Trie", autocomplete.approach6_prefix_tree_with_cache),
    ]
    
    for name, method in approaches:
        try:
            result = method(dictionary, prefix)
            print(f"\n{name:15}: {result}")
        except Exception as e:
            print(f"\n{name:15}: Error - {e}")


def test_frequency_based_suggestions():
    """Test frequency-based suggestions"""
    print("\n=== Testing Frequency-based Suggestions ===")
    
    autocomplete = BasicAutocomplete()
    
    # Dictionary with frequencies (word, frequency)
    dictionary_with_freq = [
        ("apple", 100), ("application", 50), ("apply", 80),
        ("appreciate", 20), ("approach", 30), ("appropriate", 10)
    ]
    
    prefix = "app"
    
    print(f"Dictionary with frequencies: {dictionary_with_freq}")
    print(f"Prefix: '{prefix}'")
    
    result_freq = autocomplete.approach4_frequency_based(dictionary_with_freq, prefix)
    result_heap = autocomplete.approach5_heap_based(dictionary_with_freq, prefix)
    
    print(f"\nFrequency-based: {result_freq}")
    print(f"Heap-based:      {result_heap}")


def test_fuzzy_autocomplete():
    """Test fuzzy autocomplete with edit distance"""
    print("\n=== Testing Fuzzy Autocomplete ===")
    
    autocomplete = BasicAutocomplete()
    
    dictionary = ["apple", "aple", "apply", "banana", "aply"]  # Note typos
    prefix = "app"
    
    print(f"Dictionary: {dictionary}")
    print(f"Prefix: '{prefix}' (with fuzzy matching)")
    
    # Test different edit distances
    for max_distance in [0, 1, 2]:
        result = autocomplete.approach7_fuzzy_autocomplete(dictionary, prefix, max_distance=max_distance)
        print(f"  Max distance {max_distance}: {result}")


def test_complete_system():
    """Test complete autocomplete system"""
    print("\n=== Testing Complete Autocomplete System ===")
    
    system = AutocompleteSystem()
    
    # Add words with frequencies
    words_freq = [
        ("python", 100), ("programming", 80), ("program", 90),
        ("project", 60), ("problem", 70), ("process", 50),
        ("java", 85), ("javascript", 75)
    ]
    
    print("Adding words to autocomplete system:")
    for word, freq in words_freq:
        system.add_word(word, freq)
        print(f"  Added '{word}' with frequency {freq}")
    
    # Test suggestions
    test_prefixes = ["pro", "java", "p"]
    
    for prefix in test_prefixes:
        print(f"\nSuggestions for '{prefix}':")
        
        # Frequency-sorted suggestions
        freq_suggestions = system.get_suggestions(prefix, max_suggestions=3, sort_by_frequency=True)
        print(f"  By frequency: {freq_suggestions}")
        
        # Alphabetical suggestions
        alpha_suggestions = system.get_suggestions(prefix, max_suggestions=3, sort_by_frequency=False)
        print(f"  Alphabetical: {alpha_suggestions}")


def demonstrate_real_world_usage():
    """Demonstrate real-world autocomplete usage"""
    print("\n=== Real-World Usage Demo ===")
    
    # Scenario 1: Search engine autocomplete
    print("1. Search Engine Autocomplete:")
    
    search_queries = [
        ("python programming", 1000), ("python tutorial", 800),
        ("python examples", 600), ("java programming", 900),
        ("java tutorial", 700), ("javascript", 500)
    ]
    
    autocomplete = BasicAutocomplete()
    user_input = "python"
    
    suggestions = autocomplete.approach4_frequency_based(search_queries, user_input)
    print(f"   User types: '{user_input}'")
    print(f"   Suggestions: {suggestions}")
    
    # Scenario 2: Code editor autocomplete
    print(f"\n2. Code Editor Autocomplete:")
    
    code_symbols = [
        "function", "function_call", "func_parameter",
        "variable", "var_name", "var_type",
        "class", "class_method", "class_property"
    ]
    
    user_code = "func"
    suggestions = autocomplete.approach3_trie_based(code_symbols, user_code)
    print(f"   Developer types: '{user_code}'")
    print(f"   Code suggestions: {suggestions}")
    
    # Scenario 3: E-commerce product search
    print(f"\n3. E-commerce Product Search:")
    
    products = [
        ("laptop computer", 50), ("laptop bag", 30), ("laptop stand", 20),
        ("smartphone", 80), ("smartphone case", 40), ("tablet", 60)
    ]
    
    product_search = "laptop"
    suggestions = autocomplete.approach4_frequency_based(products, product_search)
    print(f"   Customer searches: '{product_search}'")
    print(f"   Product suggestions: {suggestions}")


def benchmark_approaches():
    """Benchmark different autocomplete approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import time
    import random
    import string
    
    autocomplete = BasicAutocomplete()
    
    # Generate test data
    def generate_dictionary(size: int, avg_length: int) -> List[str]:
        words = []
        for _ in range(size):
            length = max(1, avg_length + random.randint(-2, 2))
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.append(word)
        return list(set(words))  # Remove duplicates
    
    test_scenarios = [
        ("Small", generate_dictionary(100, 6)),
        ("Medium", generate_dictionary(1000, 8)),
        ("Large", generate_dictionary(10000, 10)),
    ]
    
    approaches = [
        ("Simple List", autocomplete.approach1_simple_list),
        ("Binary Search", autocomplete.approach2_binary_search),
        ("Trie-based", autocomplete.approach3_trie_based),
        ("Cached Trie", autocomplete.approach6_prefix_tree_with_cache),
    ]
    
    prefix = "test"
    
    for scenario_name, dictionary in test_scenarios:
        print(f"\n--- {scenario_name} Dataset ---")
        print(f"Dictionary size: {len(dictionary)}, Prefix: '{prefix}'")
        
        for approach_name, method in approaches:
            start_time = time.time()
            
            # Run multiple times for better measurement
            for _ in range(10):
                result = method(dictionary, prefix)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 10
            
            print(f"  {approach_name:15}: {avg_time*1000:.2f}ms")


def demonstrate_optimization_techniques():
    """Demonstrate optimization techniques"""
    print("\n=== Optimization Techniques ===")
    
    print("1. Preprocessing Optimizations:")
    print("   • Sort dictionary once for binary search")
    print("   • Build trie structure for faster prefix navigation")
    print("   • Pre-compute suggestions at each trie node")
    
    print("\n2. Memory Optimizations:")
    print("   • Use compressed tries for memory efficiency")
    print("   • Limit cached suggestions per node")
    print("   • Share common prefixes across words")
    
    print("\n3. Query Optimizations:")
    print("   • Early termination when max suggestions reached")
    print("   • Use heaps for top-k selection")
    print("   • Cache frequently requested prefixes")
    
    print("\n4. Ranking Optimizations:")
    print("   • Incorporate user search history")
    print("   • Apply popularity/frequency weighting")
    print("   • Use machine learning for personalization")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    autocomplete = BasicAutocomplete()
    
    edge_cases = [
        # Empty cases
        ([], "test", "Empty dictionary"),
        (["apple", "banana"], "", "Empty prefix"),
        
        # Single cases
        (["apple"], "app", "Single word match"),
        (["apple"], "ban", "Single word no match"),
        
        # Prefix longer than words
        (["a", "ab"], "abc", "Prefix longer than words"),
        
        # All words have same prefix
        (["test1", "test2", "test3"], "test", "All same prefix"),
        
        # No matches
        (["apple", "banana"], "xyz", "No matches"),
        
        # Case sensitivity
        (["Apple", "apple"], "app", "Case sensitivity"),
    ]
    
    for dictionary, prefix, description in edge_cases:
        print(f"\n{description}:")
        print(f"  Dictionary: {dictionary}, Prefix: '{prefix}'")
        
        try:
            result = autocomplete.approach3_trie_based(dictionary, prefix)
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {e}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Simple List Filtering",
         "Time: O(n * m) - check each word",
         "Space: O(1) - no extra storage"),
        
        ("Binary Search",
         "Time: O(log n + k) - search + collect matches",
         "Space: O(1) - constant extra space"),
        
        ("Trie-based",
         "Time: O(sum of word lengths) build + O(prefix + k) query",
         "Space: O(sum of word lengths) - trie storage"),
        
        ("Frequency-based",
         "Time: O(n log n) - sort by frequency",
         "Space: O(n) - store frequency pairs"),
        
        ("Heap-based Top-K",
         "Time: O(n log k) - maintain heap of size k",
         "Space: O(k) - heap storage"),
        
        ("Cached Trie",
         "Time: O(sum of word lengths) build + O(prefix) query",
         "Space: O(sum of word lengths * k) - cached suggestions"),
    ]
    
    print("Approach Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nWhere:")
    print(f"  n = number of words in dictionary")
    print(f"  m = average word length")
    print(f"  k = number of suggestions returned")
    
    print(f"\nRecommendations:")
    print(f"  • Use Trie-based for frequent queries on same dictionary")
    print(f"  • Use Binary Search for sorted dictionaries with few queries")
    print(f"  • Use Cached Trie for maximum query performance")
    print(f"  • Use Frequency-based for popularity-aware suggestions")


if __name__ == "__main__":
    test_basic_autocomplete()
    test_frequency_based_suggestions()
    test_fuzzy_autocomplete()
    test_complete_system()
    demonstrate_real_world_usage()
    benchmark_approaches()
    demonstrate_optimization_techniques()
    test_edge_cases()
    analyze_complexity()

"""
Basic Autocomplete demonstrates fundamental autocomplete implementation approaches:

1. Simple List Filtering - Brute force approach checking each word
2. Binary Search - Efficient search on pre-sorted dictionary
3. Trie-based - Build prefix tree for fast navigation and collection
4. Frequency-based - Rank suggestions by popularity/frequency
5. Heap-based Top-K - Use min-heap to maintain best suggestions
6. Cached Trie - Pre-compute suggestions at each node for speed
7. Fuzzy Autocomplete - Allow approximate matches with edit distance

Each approach demonstrates different optimization strategies for building
efficient and user-friendly autocomplete systems.
"""

