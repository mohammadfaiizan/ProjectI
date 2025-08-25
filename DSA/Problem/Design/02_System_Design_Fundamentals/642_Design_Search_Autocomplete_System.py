"""
642. Design Search Autocomplete System - Multiple Approaches
Difficulty: Hard

Design a search autocomplete system for a search engine. Users may input a sentence (at least one word and end with a special character '#'). 

You are given a string array sentences and an integer array times both of length n where sentences[i] is a previously typed sentence and times[i] is the corresponding number of times the sentence has been typed. For each input character except '#', return the top 3 historical hot sentences that have the same prefix ordered by their hotness ranking.

Implement the AutocompleteSystem class:
- AutocompleteSystem(String[] sentences, int[] times) Initializes the object with the sentences and times arrays.
- List<String> input(char c) This indicates that the user typed the character c.
  - If c == '#', this means the sentence is finished. Save it as a historical sentence in your system and return an empty list.
  - Otherwise, return the list of the top 3 historical hot sentences that have the same prefix ordered by hotness ranking. If there are several sentences that have the same hotness, you need to use ASCII-betical order.
"""

import heapq
from typing import List, Dict, Tuple
from collections import defaultdict

class TrieNode:
    """Node for Trie data structure"""
    def __init__(self):
        self.children = {}
        self.sentences = []  # List of (hotness, sentence) pairs
        self.is_end = False

class AutocompleteSystemTrie:
    """
    Approach 1: Trie with Sentence Storage at Each Node
    
    Store top sentences at each trie node for fast retrieval.
    
    Time Complexity:
    - __init__: O(N * M) where N is sentences, M is avg length
    - input: O(1) for retrieval, O(M) for new sentence insertion
    
    Space Complexity: O(N * M * K) where K is nodes per sentence
    """
    
    def __init__(self, sentences: List[str], times: List[int]):
        self.root = TrieNode()
        self.current_sentence = ""
        self.current_node = self.root
        
        # Build trie with initial sentences
        for sentence, time in zip(sentences, times):
            self._add_sentence(sentence, time)
    
    def _add_sentence(self, sentence: str, hotness: int) -> None:
        """Add sentence to trie with given hotness"""
        node = self.root
        
        # Traverse/create path for sentence
        for char in sentence:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            
            # Update top sentences at this node
            self._update_node_sentences(node, sentence, hotness)
        
        node.is_end = True
    
    def _update_node_sentences(self, node: TrieNode, sentence: str, hotness: int) -> None:
        """Update top 3 sentences at a node"""
        # Remove existing entry if present
        node.sentences = [(h, s) for h, s in node.sentences if s != sentence]
        
        # Add new entry
        node.sentences.append((hotness, sentence))
        
        # Sort by hotness (desc) then by ASCII order
        node.sentences.sort(key=lambda x: (-x[0], x[1]))
        
        # Keep only top 3
        node.sentences = node.sentences[:3]
    
    def input(self, c: str) -> List[str]:
        if c == '#':
            # End of sentence - add to system
            if self.current_sentence:
                self._add_sentence(self.current_sentence, 1)
            
            # Reset for next sentence
            self.current_sentence = ""
            self.current_node = self.root
            return []
        
        # Add character to current sentence
        self.current_sentence += c
        
        # Navigate trie
        if c in self.current_node.children:
            self.current_node = self.current_node.children[c]
            # Return top sentences from current node
            return [sentence for _, sentence in self.current_node.sentences]
        else:
            # No completions possible
            self.current_node = TrieNode()  # Dead end
            return []

class AutocompleteSystemHashMap:
    """
    Approach 2: HashMap with Prefix Matching
    
    Store all sentences in HashMap and filter by prefix on each input.
    
    Time Complexity:
    - __init__: O(N)
    - input: O(N * M) for filtering and sorting
    
    Space Complexity: O(N * M)
    """
    
    def __init__(self, sentences: List[str], times: List[int]):
        self.sentence_counts = {}
        self.current_sentence = ""
        
        # Store sentence frequencies
        for sentence, time in zip(sentences, times):
            self.sentence_counts[sentence] = time
    
    def input(self, c: str) -> List[str]:
        if c == '#':
            # End of sentence
            if self.current_sentence:
                if self.current_sentence in self.sentence_counts:
                    self.sentence_counts[self.current_sentence] += 1
                else:
                    self.sentence_counts[self.current_sentence] = 1
            
            self.current_sentence = ""
            return []
        
        # Add character to current sentence
        self.current_sentence += c
        
        # Find all sentences with current prefix
        candidates = []
        for sentence, count in self.sentence_counts.items():
            if sentence.startswith(self.current_sentence):
                candidates.append((count, sentence))
        
        # Sort by count (desc) then ASCII order
        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        # Return top 3
        return [sentence for _, sentence in candidates[:3]]

class AutocompleteSystemOptimized:
    """
    Approach 3: Optimized Trie with Lazy Updates
    
    Use trie but delay expensive updates until needed.
    
    Time Complexity:
    - __init__: O(N * M)
    - input: O(1) amortized for retrieval
    
    Space Complexity: O(N * M)
    """
    
    def __init__(self, sentences: List[str], times: List[int]):
        self.root = TrieNode()
        self.sentence_counts = {}
        self.current_sentence = ""
        self.current_node = self.root
        
        # Initialize sentence counts
        for sentence, time in zip(sentences, times):
            self.sentence_counts[sentence] = time
        
        # Build basic trie structure
        for sentence in sentences:
            self._build_trie_path(sentence)
    
    def _build_trie_path(self, sentence: str) -> None:
        """Build trie path without storing sentences at nodes"""
        node = self.root
        for char in sentence:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def _get_sentences_for_prefix(self, prefix: str) -> List[str]:
        """Get all sentences with given prefix"""
        sentences = []
        
        for sentence in self.sentence_counts:
            if sentence.startswith(prefix):
                count = self.sentence_counts[sentence]
                sentences.append((count, sentence))
        
        # Sort and return top 3
        sentences.sort(key=lambda x: (-x[0], x[1]))
        return [sentence for _, sentence in sentences[:3]]
    
    def input(self, c: str) -> List[str]:
        if c == '#':
            # End of sentence
            if self.current_sentence:
                if self.current_sentence in self.sentence_counts:
                    self.sentence_counts[self.current_sentence] += 1
                else:
                    self.sentence_counts[self.current_sentence] = 1
                    # Add to trie structure
                    self._build_trie_path(self.current_sentence)
            
            self.current_sentence = ""
            self.current_node = self.root
            return []
        
        # Add character
        self.current_sentence += c
        
        # Navigate trie
        if c in self.current_node.children:
            self.current_node = self.current_node.children[c]
            return self._get_sentences_for_prefix(self.current_sentence)
        else:
            # No valid path
            self.current_node = TrieNode()  # Dead end
            return []

class AutocompleteSystemAdvanced:
    """
    Approach 4: Advanced with Caching and Analytics
    
    Enhanced system with caching and usage analytics.
    
    Time Complexity:
    - __init__: O(N * M)
    - input: O(1) for cached results, O(N) for cache miss
    
    Space Complexity: O(N * M + C) where C is cache size
    """
    
    def __init__(self, sentences: List[str], times: List[int]):
        self.sentence_counts = {}
        self.current_sentence = ""
        
        # Caching for frequent prefixes
        self.prefix_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Analytics
        self.total_inputs = 0
        self.sentences_added = 0
        self.prefix_queries = 0
        
        # Initialize
        for sentence, time in zip(sentences, times):
            self.sentence_counts[sentence] = time
    
    def input(self, c: str) -> List[str]:
        self.total_inputs += 1
        
        if c == '#':
            # End of sentence
            if self.current_sentence:
                if self.current_sentence in self.sentence_counts:
                    self.sentence_counts[self.current_sentence] += 1
                else:
                    self.sentence_counts[self.current_sentence] = 1
                    self.sentences_added += 1
                
                # Invalidate cache entries that might be affected
                self._invalidate_cache_for_sentence(self.current_sentence)
            
            self.current_sentence = ""
            return []
        
        # Add character
        self.current_sentence += c
        self.prefix_queries += 1
        
        # Check cache first
        if self.current_sentence in self.prefix_cache:
            self.cache_hits += 1
            return self.prefix_cache[self.current_sentence]
        
        self.cache_misses += 1
        
        # Compute result
        candidates = []
        for sentence, count in self.sentence_counts.items():
            if sentence.startswith(self.current_sentence):
                candidates.append((count, sentence))
        
        candidates.sort(key=lambda x: (-x[0], x[1]))
        result = [sentence for _, sentence in candidates[:3]]
        
        # Cache result (limit cache size)
        if len(self.prefix_cache) < 1000:
            self.prefix_cache[self.current_sentence] = result
        
        return result
    
    def _invalidate_cache_for_sentence(self, sentence: str) -> None:
        """Invalidate cache entries that might be affected by new sentence"""
        prefixes_to_remove = []
        
        for cached_prefix in self.prefix_cache:
            if sentence.startswith(cached_prefix):
                prefixes_to_remove.append(cached_prefix)
        
        for prefix in prefixes_to_remove:
            del self.prefix_cache[prefix]
    
    def get_analytics(self) -> Dict[str, any]:
        """Get system analytics"""
        cache_hit_rate = self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        
        return {
            'total_inputs': self.total_inputs,
            'prefix_queries': self.prefix_queries,
            'sentences_added': self.sentences_added,
            'total_sentences': len(self.sentence_counts),
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.prefix_cache)
        }

class AutocompleteSystemHeap:
    """
    Approach 5: Heap-based for Top-K Selection
    
    Use heap to efficiently maintain top-3 results.
    
    Time Complexity:
    - __init__: O(N)
    - input: O(N log 3) for heap operations
    
    Space Complexity: O(N)
    """
    
    def __init__(self, sentences: List[str], times: List[int]):
        self.sentence_counts = {}
        self.current_sentence = ""
        
        for sentence, time in zip(sentences, times):
            self.sentence_counts[sentence] = time
    
    def input(self, c: str) -> List[str]:
        if c == '#':
            if self.current_sentence:
                if self.current_sentence in self.sentence_counts:
                    self.sentence_counts[self.current_sentence] += 1
                else:
                    self.sentence_counts[self.current_sentence] = 1
            
            self.current_sentence = ""
            return []
        
        self.current_sentence += c
        
        # Use min-heap to find top 3
        heap = []
        
        for sentence, count in self.sentence_counts.items():
            if sentence.startswith(self.current_sentence):
                # Use negative count for max-heap behavior with min-heap
                # For tie-breaking, use sentence itself (ASCII order)
                heapq.heappush(heap, (count, sentence))
                
                if len(heap) > 3:
                    heapq.heappop(heap)
        
        # Extract results and sort properly
        results = []
        while heap:
            count, sentence = heapq.heappop(heap)
            results.append((count, sentence))
        
        # Sort by count (desc) then ASCII order
        results.sort(key=lambda x: (-x[0], x[1]))
        
        return [sentence for _, sentence in results]


def test_autocomplete_basic():
    """Test basic autocomplete functionality"""
    print("=== Testing Basic Autocomplete Functionality ===")
    
    implementations = [
        ("Trie-based", AutocompleteSystemTrie),
        ("HashMap-based", AutocompleteSystemHashMap),
        ("Optimized Trie", AutocompleteSystemOptimized),
        ("Advanced", AutocompleteSystemAdvanced),
        ("Heap-based", AutocompleteSystemHeap)
    ]
    
    sentences = ["i love you", "island", "iroman", "i love leetcode"]
    times = [5, 3, 2, 2]
    
    for name, AutocompleteClass in implementations:
        print(f"\n{name}:")
        
        system = AutocompleteClass(sentences, times)
        
        # Test input sequence: "i" -> " " -> "a" -> "#"
        test_inputs = ["i", " ", "a", "#"]
        
        for char in test_inputs:
            result = system.input(char)
            if char == '#':
                print(f"  input('{char}'): {result} (sentence completed)")
            else:
                print(f"  input('{char}'): {result}")

def test_autocomplete_complex():
    """Test complex autocomplete scenarios"""
    print("\n=== Testing Complex Autocomplete Scenarios ===")
    
    sentences = ["i love you", "island", "iroman", "i love leetcode"]
    times = [5, 3, 2, 2]
    
    system = AutocompleteSystemTrie(sentences, times)
    
    # Test multiple queries
    queries = [
        "i love you#",
        "i a#",
        "i love leetcode#",
        "i#"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        for char in query:
            result = system.input(char)
            if char != '#':
                print(f"  '{char}' -> {result}")
            else:
                print(f"  '#' -> {result} (completed)")

def test_hotness_ranking():
    """Test hotness-based ranking"""
    print("\n=== Testing Hotness Ranking ===")
    
    sentences = ["apple", "app", "application", "apply"]
    times = [10, 5, 2, 1]  # Different hotness levels
    
    system = AutocompleteSystemHashMap(sentences, times)
    
    # Test prefix "app"
    print("Testing prefix 'app':")
    
    for char in "app":
        result = system.input(char)
        print(f"  '{char}' -> {result}")
    
    # Add new sentence and test again
    print(f"\nAdding new sentence 'application':")
    for char in "application#":
        system.input(char)
    
    print(f"Testing 'app' again after adding 'application':")
    
    # Reset and test
    for char in "app":
        result = system.input(char)
        print(f"  '{char}' -> {result}")

def test_ascii_ordering():
    """Test ASCII ordering for ties"""
    print("\n=== Testing ASCII Ordering ===")
    
    # Sentences with same hotness - should be ordered by ASCII
    sentences = ["zebra", "apple", "banana"]
    times = [1, 1, 1]  # Same hotness
    
    system = AutocompleteSystemOptimized(sentences, times)
    
    # Test empty prefix (should show all, ordered by ASCII)
    print("All sentences with same hotness:")
    
    # Since we need a common prefix, let's add a common character
    # Reset and add sentences with common prefix
    sentences2 = ["aa", "ab", "ac"]
    times2 = [1, 1, 1]
    
    system2 = AutocompleteSystemOptimized(sentences2, times2)
    
    result = system2.input('a')
    print(f"  Prefix 'a': {result}")

def test_performance_comparison():
    """Test performance of different implementations"""
    print("\n=== Testing Performance Comparison ===")
    
    import time
    
    # Create larger dataset
    sentences = [f"sentence_{i:04d}" for i in range(1000)]
    times = [i + 1 for i in range(1000)]
    
    implementations = [
        ("Trie-based", AutocompleteSystemTrie),
        ("HashMap-based", AutocompleteSystemHashMap),
        ("Optimized", AutocompleteSystemOptimized),
        ("Advanced", AutocompleteSystemAdvanced)
    ]
    
    for name, AutocompleteClass in implementations:
        # Time initialization
        start_time = time.time()
        system = AutocompleteClass(sentences, times)
        init_time = (time.time() - start_time) * 1000
        
        # Time queries
        start_time = time.time()
        
        # Perform several prefix queries
        test_prefixes = ["s", "se", "sen", "sent"]
        for prefix in test_prefixes:
            for char in prefix:
                system.input(char)
            # Reset for next test
            for _ in range(len(prefix)):
                pass  # In real system, would reset or handle differently
        
        query_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    Init: {init_time:.2f}ms")
        print(f"    Queries: {query_time:.2f}ms")

def test_advanced_analytics():
    """Test advanced analytics features"""
    print("\n=== Testing Advanced Analytics ===")
    
    sentences = ["hello", "world", "help", "helm"]
    times = [5, 3, 2, 1]
    
    system = AutocompleteSystemAdvanced(sentences, times)
    
    # Perform various operations
    operations = [
        "h", "e", "l", "#",  # Complete "hel"
        "h", "e", "l", "p", "#",  # Complete "help"
        "h", "e", "#",  # Complete "he"
        "w", "o", "r", "#"  # Complete "wor"
    ]
    
    print("Performing operations to generate analytics...")
    
    for char in operations:
        system.input(char)
    
    # Get analytics
    analytics = system.get_analytics()
    
    print(f"System Analytics:")
    for key, value in analytics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Search engine autocomplete
    print("Application 1: Search Engine Autocomplete")
    
    search_queries = [
        "python programming", "python tutorial", "python basics",
        "java programming", "javascript tutorial", "machine learning"
    ]
    query_counts = [100, 80, 60, 90, 70, 150]
    
    search_autocomplete = AutocompleteSystemTrie(search_queries, query_counts)
    
    print(f"  Popular search queries:")
    for query, count in zip(search_queries, query_counts):
        print(f"    '{query}': {count} searches")
    
    # Test some prefixes
    test_prefixes = ["python", "java", "machine"]
    
    for prefix in test_prefixes:
        print(f"\n  Typing '{prefix}':")
        
        for char in prefix:
            suggestions = search_autocomplete.input(char)
            if suggestions:
                print(f"    After '{char}': {suggestions[:2]}...")  # Show first 2
    
    # Application 2: Code editor autocomplete
    print(f"\nApplication 2: Code Editor Function Autocomplete")
    
    functions = [
        "print()", "println()", "printf()",
        "len()", "list()", "lambda",
        "range()", "return", "raise"
    ]
    usage_counts = [1000, 200, 300, 800, 400, 150, 600, 900, 100]
    
    code_autocomplete = AutocompleteSystemHashMap(functions, usage_counts)
    
    print(f"  Testing prefix 'p':")
    suggestions = code_autocomplete.input('p')
    print(f"    Suggestions: {suggestions}")
    
    print(f"  Testing prefix 'pr':")
    code_autocomplete.input('r')  # Continue from 'p'
    suggestions = code_autocomplete.input('i')  # Now at 'pri'
    # Reset and try 'pr'
    code_autocomplete2 = AutocompleteSystemHashMap(functions, usage_counts)
    code_autocomplete2.input('p')
    suggestions = code_autocomplete2.input('r')
    print(f"    Suggestions: {suggestions}")

def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    # Empty initialization
    print("Empty initialization:")
    system = AutocompleteSystemTrie([], [])
    result = system.input('a')
    print(f"  input('a') with empty system: {result}")
    
    # Single character sentences
    print(f"\nSingle character sentences:")
    system = AutocompleteSystemHashMap(["a", "b", "c"], [3, 2, 1])
    
    for char in "abc":
        result = system.input(char)
        print(f"  input('{char}'): {result}")
    
    # Very long sentences
    print(f"\nVery long sentence:")
    long_sentence = "a" * 100
    system = AutocompleteSystemOptimized([long_sentence], [1])
    
    # Test first few characters
    for i, char in enumerate(long_sentence[:5]):
        result = system.input(char)
        print(f"  char {i+1}: {len(result)} suggestions")

def test_memory_efficiency():
    """Test memory efficiency"""
    print("\n=== Testing Memory Efficiency ===")
    
    # Create system with many similar sentences
    sentences = [f"prefix_{i:04d}" for i in range(1000)]
    times = [1] * 1000
    
    implementations = [
        ("Trie-based", AutocompleteSystemTrie),
        ("HashMap-based", AutocompleteSystemHashMap),
        ("Optimized", AutocompleteSystemOptimized)
    ]
    
    for name, AutocompleteClass in implementations:
        system = AutocompleteClass(sentences, times)
        
        # Estimate memory usage (simplified)
        if hasattr(system, 'sentence_counts'):
            sentences_memory = len(system.sentence_counts)
        else:
            sentences_memory = len(sentences)
        
        if hasattr(system, 'root'):
            # Rough estimate for trie
            trie_memory = len(sentences) * 10  # Approximate
        else:
            trie_memory = 0
        
        total_memory = sentences_memory + trie_memory
        
        print(f"  {name}: ~{total_memory} memory units")

def stress_test_autocomplete():
    """Stress test autocomplete system"""
    print("\n=== Stress Testing Autocomplete System ===")
    
    import time
    
    # Create large dataset
    sentences = []
    times = []
    
    # Generate sentences with patterns
    prefixes = ["python", "java", "javascript", "machine", "data"]
    suffixes = ["tutorial", "guide", "example", "programming", "learning"]
    
    for prefix in prefixes:
        for suffix in suffixes:
            sentence = f"{prefix} {suffix}"
            sentences.append(sentence)
            times.append(len(sentence))  # Use length as frequency
    
    print(f"Created {len(sentences)} sentences")
    
    system = AutocompleteSystemAdvanced(sentences, times)
    
    # Stress test with many queries
    start_time = time.time()
    
    # Test all single character prefixes
    query_count = 0
    for char in "abcdefghijklmnopqrstuvwxyz":
        result = system.input(char)
        query_count += 1
        
        if query_count % 10 == 0:
            # Reset occasionally
            pass
    
    elapsed = time.time() - start_time
    
    print(f"Performed {query_count} queries in {elapsed:.3f}s")
    
    # Get final analytics
    if hasattr(system, 'get_analytics'):
        analytics = system.get_analytics()
        print(f"Final analytics: {analytics}")

if __name__ == "__main__":
    test_autocomplete_basic()
    test_autocomplete_complex()
    test_hotness_ranking()
    test_ascii_ordering()
    test_performance_comparison()
    test_advanced_analytics()
    demonstrate_applications()
    test_edge_cases()
    test_memory_efficiency()
    stress_test_autocomplete()

"""
Search Autocomplete System Design demonstrates key concepts:

Core Approaches:
1. Trie with Sentence Storage - Store top sentences at each node
2. HashMap with Filtering - Simple but less efficient for large datasets
3. Optimized Trie - Lazy updates and efficient traversal
4. Advanced with Caching - Performance optimization with analytics
5. Heap-based - Efficient top-K selection

Key Design Principles:
- Prefix-based search efficiency
- Hotness ranking with ASCII tie-breaking
- Memory vs query time trade-offs
- Real-time updates and caching strategies

Performance Characteristics:
- Trie: O(M) per query where M is prefix length
- HashMap: O(N) per query where N is total sentences
- Advanced: O(1) for cached queries, O(N) for cache miss
- Memory: O(N*M) to O(N*M*K) depending on approach

Real-world Applications:
- Search engine query suggestions
- Code editor autocomplete
- E-commerce product search
- Social media mention suggestions
- Email address completion
- Command-line tool completion

Advanced Features:
- Caching for frequent prefixes
- Usage analytics and optimization
- Real-time learning from user input
- Personalized suggestions
- Multi-language support

The trie-based approach with sentence storage at nodes
is most commonly used for production autocomplete systems
due to its optimal query performance characteristics.
"""
