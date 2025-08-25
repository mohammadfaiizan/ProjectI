"""
642. Design Search Autocomplete System - Multiple Approaches
Difficulty: Medium

Design a search autocomplete system for a search engine. Users may input a sentence 
(at least one word and end with a special character '#'). For each character they type 
except '#', you need to return the top 3 historical hot sentences that have prefix 
the same as the part of sentence already typed.

LeetCode Problem: https://leetcode.com/problems/design-search-autocomplete-system/

Example:
AutocompleteSystem(["i love you", "island","ironman", "i love leetcode"], [5,3,2,2])
The system have already tracked down the following sentences and their corresponding times:
"i love you" : 5 times
"island" : 3 times  
"ironman" : 2 times
"i love leetcode" : 2 times
"""

from typing import List, Dict, Tuple, Optional
import heapq
from collections import defaultdict

class TrieNode:
    """Trie node for autocomplete system"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.sentence = ""
        self.count = 0
        self.hot_sentences = []  # Top 3 hot sentences

class AutocompleteSystem1:
    """
    Approach 1: Trie with DFS Collection
    
    Build trie and collect suggestions using DFS on each input.
    """
    
    def __init__(self, sentences: List[str], times: List[int]):
        """
        Time: O(sum of sentence lengths)
        Space: O(sum of sentence lengths)
        """
        self.root = TrieNode()
        self.current_input = ""
        
        # Build trie with historical data
        for sentence, count in zip(sentences, times):
            self._add_sentence(sentence, count)
    
    def _add_sentence(self, sentence: str, count: int) -> None:
        """Add sentence to trie with count"""
        node = self.root
        for char in sentence:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end = True
        node.sentence = sentence
        node.count += count
    
    def input(self, c: str) -> List[str]:
        """
        Process input character and return top 3 suggestions.
        
        Time: O(prefix_length + total_sentences * max_sentence_length)
        Space: O(total_sentences)
        """
        if c == '#':
            # End of input - add current sentence and reset
            if self.current_input:
                self._add_sentence(self.current_input, 1)
            self.current_input = ""
            return []
        
        self.current_input += c
        
        # Navigate to current prefix in trie
        node = self.root
        for char in self.current_input:
            if char not in node.children:
                return []  # No suggestions for this prefix
            node = node.children[char]
        
        # Collect all sentences with DFS
        suggestions = []
        
        def dfs(current_node: TrieNode):
            if current_node.is_end:
                suggestions.append((current_node.sentence, current_node.count))
            
            for child in current_node.children.values():
                dfs(child)
        
        dfs(node)
        
        # Sort by count (desc) then lexicographically
        suggestions.sort(key=lambda x: (-x[1], x[0]))
        
        return [sentence for sentence, _ in suggestions[:3]]


class AutocompleteSystem2:
    """
    Approach 2: Trie with Pre-computed Hot Sentences
    
    Store top 3 hot sentences at each trie node.
    """
    
    def __init__(self, sentences: List[str], times: List[int]):
        """
        Time: O(sum of sentence lengths * log(hot_count))
        Space: O(sum of sentence lengths * 3)
        """
        self.root = TrieNode()
        self.current_input = ""
        
        # Build trie with hot sentences
        for sentence, count in zip(sentences, times):
            self._add_sentence(sentence, count)
    
    def _add_sentence(self, sentence: str, count: int) -> None:
        """Add sentence and update hot sentences along the path"""
        node = self.root
        for char in sentence:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            
            # Update hot sentences at this node
            self._update_hot_sentences(node, sentence, count)
        
        node.is_end = True
        node.sentence = sentence
        node.count += count
        
        # Final update at leaf node
        self._update_hot_sentences(node, sentence, count)
    
    def _update_hot_sentences(self, node: TrieNode, sentence: str, count: int) -> None:
        """Update top 3 hot sentences at node"""
        # Remove existing entry if present
        node.hot_sentences = [(s, c) for s, c in node.hot_sentences if s != sentence]
        
        # Add new entry
        node.hot_sentences.append((sentence, count))
        
        # Sort and keep top 3
        node.hot_sentences.sort(key=lambda x: (-x[1], x[0]))
        node.hot_sentences = node.hot_sentences[:3]
    
    def input(self, c: str) -> List[str]:
        """
        Process input character and return cached suggestions.
        
        Time: O(prefix_length)
        Space: O(1)
        """
        if c == '#':
            if self.current_input:
                self._add_sentence(self.current_input, 1)
            self.current_input = ""
            return []
        
        self.current_input += c
        
        # Navigate to current prefix
        node = self.root
        for char in self.current_input:
            if char not in node.children:
                return []
            node = node.children[char]
        
        return [sentence for sentence, _ in node.hot_sentences]


class AutocompleteSystem3:
    """
    Approach 3: Hash Map with Prefix Matching
    
    Use hash map to store sentences and filter by prefix.
    """
    
    def __init__(self, sentences: List[str], times: List[int]):
        """
        Time: O(1)
        Space: O(sum of sentence lengths)
        """
        self.sentence_counts = {}
        self.current_input = ""
        
        for sentence, count in zip(sentences, times):
            self.sentence_counts[sentence] = count
    
    def input(self, c: str) -> List[str]:
        """
        Process input and filter sentences by prefix.
        
        Time: O(total_sentences * prefix_length)
        Space: O(matching_sentences)
        """
        if c == '#':
            if self.current_input:
                self.sentence_counts[self.current_input] = self.sentence_counts.get(self.current_input, 0) + 1
            self.current_input = ""
            return []
        
        self.current_input += c
        
        # Filter sentences by current prefix
        matching = []
        for sentence, count in self.sentence_counts.items():
            if sentence.startswith(self.current_input):
                matching.append((sentence, count))
        
        # Sort and return top 3
        matching.sort(key=lambda x: (-x[1], x[0]))
        return [sentence for sentence, _ in matching[:3]]


class AutocompleteSystem4:
    """
    Approach 4: Optimized with Lazy Computation
    
    Compute suggestions only when needed using lazy evaluation.
    """
    
    def __init__(self, sentences: List[str], times: List[int]):
        """
        Time: O(sum of sentence lengths)
        Space: O(sum of sentence lengths)
        """
        self.root = TrieNode()
        self.current_input = ""
        self.current_node = self.root
        self.cache = {}  # Cache for prefix -> suggestions
        
        for sentence, count in zip(sentences, times):
            self._add_sentence(sentence, count)
    
    def _add_sentence(self, sentence: str, count: int) -> None:
        """Add sentence to trie"""
        node = self.root
        for char in sentence:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end = True
        node.sentence = sentence
        node.count += count
        
        # Invalidate cache for affected prefixes
        for i in range(1, len(sentence) + 1):
            prefix = sentence[:i]
            if prefix in self.cache:
                del self.cache[prefix]
    
    def input(self, c: str) -> List[str]:
        """
        Process input with caching and lazy computation.
        
        Time: O(prefix_length) with cache hit, O(total_sentences) with cache miss
        Space: O(cache_size)
        """
        if c == '#':
            if self.current_input:
                self._add_sentence(self.current_input, 1)
            self.current_input = ""
            self.current_node = self.root
            return []
        
        self.current_input += c
        
        # Check cache first
        if self.current_input in self.cache:
            return self.cache[self.current_input]
        
        # Navigate trie
        if c in self.current_node.children:
            self.current_node = self.current_node.children[c]
        else:
            self.current_node = None
            self.cache[self.current_input] = []
            return []
        
        # Collect suggestions
        suggestions = []
        
        def dfs(node: TrieNode):
            if node.is_end:
                suggestions.append((node.sentence, node.count))
            for child in node.children.values():
                dfs(child)
        
        dfs(self.current_node)
        suggestions.sort(key=lambda x: (-x[1], x[0]))
        
        result = [sentence for sentence, _ in suggestions[:3]]
        self.cache[self.current_input] = result
        
        return result


class AutocompleteSystem5:
    """
    Approach 5: Heap-based Top-K with Streaming
    
    Use min-heap to maintain top-k suggestions efficiently.
    """
    
    def __init__(self, sentences: List[str], times: List[int]):
        """
        Time: O(sum of sentence lengths)
        Space: O(sum of sentence lengths)
        """
        self.sentence_counts = {}
        self.current_input = ""
        
        for sentence, count in zip(sentences, times):
            self.sentence_counts[sentence] = count
    
    def input(self, c: str) -> List[str]:
        """
        Process input using heap for top-k selection.
        
        Time: O(total_sentences * log(3))
        Space: O(3) for heap
        """
        if c == '#':
            if self.current_input:
                self.sentence_counts[self.current_input] = self.sentence_counts.get(self.current_input, 0) + 1
            self.current_input = ""
            return []
        
        self.current_input += c
        
        # Use min-heap to find top 3
        heap = []
        
        for sentence, count in self.sentence_counts.items():
            if sentence.startswith(self.current_input):
                if len(heap) < 3:
                    heapq.heappush(heap, (count, sentence))
                elif count > heap[0][0] or (count == heap[0][0] and sentence < heap[0][1]):
                    heapq.heapreplace(heap, (count, sentence))
        
        # Extract and sort results
        results = []
        while heap:
            count, sentence = heapq.heappop(heap)
            results.append((sentence, count))
        
        results.sort(key=lambda x: (-x[1], x[0]))
        return [sentence for sentence, _ in results]


class AutocompleteSystem6:
    """
    Approach 6: Inverted Index with Ranking
    
    Build inverted index for efficient prefix queries.
    """
    
    def __init__(self, sentences: List[str], times: List[int]):
        """
        Time: O(sum of sentence lengths squared)
        Space: O(sum of sentence lengths squared)
        """
        self.sentence_counts = {}
        self.prefix_index = defaultdict(list)  # prefix -> list of sentences
        self.current_input = ""
        
        for sentence, count in zip(sentences, times):
            self.sentence_counts[sentence] = count
            self._index_sentence(sentence)
    
    def _index_sentence(self, sentence: str) -> None:
        """Add sentence to inverted index"""
        for i in range(1, len(sentence) + 1):
            prefix = sentence[:i]
            if sentence not in self.prefix_index[prefix]:
                self.prefix_index[prefix].append(sentence)
    
    def input(self, c: str) -> List[str]:
        """
        Process input using inverted index.
        
        Time: O(sentences_with_prefix * log(sentences_with_prefix))
        Space: O(sentences_with_prefix)
        """
        if c == '#':
            if self.current_input:
                self.sentence_counts[self.current_input] = self.sentence_counts.get(self.current_input, 0) + 1
                self._index_sentence(self.current_input)
            self.current_input = ""
            return []
        
        self.current_input += c
        
        # Get sentences with current prefix
        sentences_with_prefix = self.prefix_index.get(self.current_input, [])
        
        # Create list with counts and sort
        candidates = [(sentence, self.sentence_counts[sentence]) for sentence in sentences_with_prefix]
        candidates.sort(key=lambda x: (-x[1], x[0]))
        
        return [sentence for sentence, _ in candidates[:3]]


def test_autocomplete_systems():
    """Test all autocomplete system approaches"""
    print("=== Testing Autocomplete Systems ===")
    
    # Test data
    sentences = ["i love you", "island", "ironman", "i love leetcode"]
    times = [5, 3, 2, 2]
    
    # Test inputs
    test_inputs = [
        ('i', ["i love you", "island", "i love leetcode"]),
        (' ', ["i love you", "i love leetcode"]),
        ('a', []),
        ('#', []),
        ('i', ["i love you", "island", "i love leetcode"]),
        (' ', ["i love you", "i love leetcode"]),
        ('a', ["i a"]),  # After adding "i a"
        ('#', []),
    ]
    
    systems = [
        ("Trie DFS", AutocompleteSystem1),
        ("Pre-computed Hot", AutocompleteSystem2),
        ("Hash Map", AutocompleteSystem3),
        ("Lazy Computation", AutocompleteSystem4),
        ("Heap-based", AutocompleteSystem5),
        ("Inverted Index", AutocompleteSystem6),
    ]
    
    for name, SystemClass in systems:
        print(f"\n{name}:")
        system = SystemClass(sentences[:], times[:])
        
        results = []
        for char, expected in test_inputs:
            result = system.input(char)
            results.append(result)
            
            # Check if result matches expected (allowing for some flexibility)
            status = "✓" if set(result) <= set(expected) and len(result) <= 3 else "?"
            print(f"  Input '{char}': {result} {status}")


def demonstrate_system_usage():
    """Demonstrate autocomplete system usage"""
    print("\n=== System Usage Demo ===")
    
    # Initialize system
    sentences = ["i love you", "island", "ironman", "i love leetcode", "i love programming"]
    times = [5, 3, 2, 2, 1]
    
    system = AutocompleteSystem2(sentences, times)
    
    print("Historical sentences:")
    for sentence, count in zip(sentences, times):
        print(f"  '{sentence}': {count} times")
    
    # Simulate user typing
    user_sessions = [
        "i love coding#",
        "iron#",
        "i am#"
    ]
    
    for session in user_sessions:
        print(f"\nUser session: typing '{session[:-1]}'")
        
        for char in session:
            suggestions = system.input(char)
            if char == '#':
                print(f"  [ENTER] - sentence completed")
            else:
                print(f"  Type '{char}': suggestions = {suggestions}")


def demonstrate_real_world_scenarios():
    """Demonstrate real-world autocomplete scenarios"""
    print("\n=== Real-World Scenarios ===")
    
    # Scenario 1: Search engine queries
    print("1. Search Engine Autocomplete:")
    
    search_queries = [
        "python programming tutorial", "python data science",
        "java programming", "javascript tutorial",
        "machine learning", "machine learning python"
    ]
    query_counts = [100, 80, 90, 70, 120, 60]
    
    search_system = AutocompleteSystem2(search_queries, query_counts)
    
    user_query = "python"
    print(f"   User starts typing: '{user_query}'")
    
    for char in user_query:
        suggestions = search_system.input(char)
        print(f"     After '{user_query[:user_query.index(char)+1]}': {suggestions}")
    
    # Scenario 2: Code completion
    print(f"\n2. Code Editor Autocomplete:")
    
    code_completions = [
        "function main()", "function calculateSum()",
        "for loop", "for i in range",
        "if condition", "if __name__ == '__main__'"
    ]
    completion_counts = [50, 30, 40, 35, 45, 25]
    
    code_system = AutocompleteSystem2(code_completions, completion_counts)
    
    code_input = "func"
    print(f"   Developer types: '{code_input}'")
    
    for char in code_input:
        suggestions = code_system.input(char)
        print(f"     After '{code_input[:code_input.index(char)+1]}': {suggestions}")


def benchmark_systems():
    """Benchmark different autocomplete systems"""
    print("\n=== Benchmarking Systems ===")
    
    import time
    import random
    import string
    
    # Generate test data
    def generate_sentences(count: int, avg_length: int) -> Tuple[List[str], List[int]]:
        sentences = []
        times = []
        
        for _ in range(count):
            # Generate sentence
            word_count = random.randint(1, 5)
            words = []
            for _ in range(word_count):
                word_length = random.randint(3, 8)
                word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
                words.append(word)
            
            sentence = ' '.join(words)
            sentences.append(sentence)
            times.append(random.randint(1, 100))
        
        return sentences, times
    
    # Test scenarios
    test_scenarios = [
        ("Small", *generate_sentences(50, 20)),
        ("Medium", *generate_sentences(200, 25)),
        ("Large", *generate_sentences(1000, 30)),
    ]
    
    systems = [
        ("Pre-computed Hot", AutocompleteSystem2),
        ("Hash Map", AutocompleteSystem3),
        ("Lazy Computation", AutocompleteSystem4),
    ]
    
    for scenario_name, sentences, times in test_scenarios:
        print(f"\n--- {scenario_name} Dataset ---")
        print(f"Sentences: {len(sentences)}, Avg length: {sum(len(s) for s in sentences)/len(sentences):.1f}")
        
        for system_name, SystemClass in systems:
            # Measure initialization time
            start_time = time.time()
            system = SystemClass(sentences, times)
            init_time = time.time() - start_time
            
            # Measure query time
            test_queries = ["test", "hello", "python"]
            start_time = time.time()
            
            for query in test_queries:
                for char in query:
                    system.input(char)
                system.input('#')  # Complete query
            
            query_time = (time.time() - start_time) / len(test_queries)
            
            print(f"  {system_name:15}: Init {init_time*1000:.2f}ms, Query {query_time*1000:.2f}ms")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Trie with DFS",
         "Init: O(sum of sentence lengths)",
         "Query: O(prefix_length + total_sentences * max_length)",
         "Space: O(sum of sentence lengths)"),
        
        ("Pre-computed Hot Sentences",
         "Init: O(sum of sentence lengths * log(hot_count))",
         "Query: O(prefix_length)",
         "Space: O(sum of sentence lengths * 3)"),
        
        ("Hash Map Filtering",
         "Init: O(1)",
         "Query: O(total_sentences * prefix_length)",
         "Space: O(sum of sentence lengths)"),
        
        ("Lazy Computation",
         "Init: O(sum of sentence lengths)",
         "Query: O(prefix_length) cached, O(total_sentences) uncached",
         "Space: O(sum of sentence lengths + cache_size)"),
        
        ("Heap-based Top-K",
         "Init: O(1)",
         "Query: O(total_sentences * log(k))",
         "Space: O(sum of sentence lengths + k)"),
        
        ("Inverted Index",
         "Init: O(sum of sentence lengths²)",
         "Query: O(sentences_with_prefix * log(sentences_with_prefix))",
         "Space: O(sum of sentence lengths²)"),
    ]
    
    print("System Analysis:")
    for system, init_complexity, query_complexity, space_complexity in complexity_analysis:
        print(f"\n{system}:")
        print(f"  Initialization: {init_complexity}")
        print(f"  Query: {query_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nRecommendations:")
    print(f"  • Use Pre-computed Hot for frequent queries (best query performance)")
    print(f"  • Use Hash Map for simple implementation with few sentences")
    print(f"  • Use Lazy Computation for balanced performance with caching")
    print(f"  • Use Inverted Index for complex prefix pattern queries")


if __name__ == "__main__":
    test_autocomplete_systems()
    demonstrate_system_usage()
    demonstrate_real_world_scenarios()
    benchmark_systems()
    analyze_complexity()

"""
642. Design Search Autocomplete System demonstrates multiple system architectures:

1. Trie with DFS - Build trie and collect suggestions using depth-first search
2. Pre-computed Hot Sentences - Store top-k suggestions at each trie node
3. Hash Map Filtering - Simple approach filtering all sentences by prefix
4. Lazy Computation - Compute suggestions on-demand with caching
5. Heap-based Top-K - Use min-heap for efficient top-k selection
6. Inverted Index - Build prefix index for fast retrieval

Each approach offers different trade-offs between initialization cost,
query performance, memory usage, and implementation complexity.
"""

