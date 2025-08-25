"""
Pattern Optimization Techniques - Multiple Approaches
Difficulty: Advanced

Advanced optimization techniques for pattern matching algorithms including
preprocessing optimizations, cache-friendly implementations, and parallel approaches.
"""

from typing import List, Dict, Set, Tuple, Optional, Union, Callable
from collections import defaultdict, deque
import threading
import concurrent.futures
from functools import lru_cache
import bisect

class OptimizedTrieNode:
    """Memory-optimized trie node"""
    __slots__ = ['children', 'is_end', 'patterns', 'failure_link']
    
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.patterns = []
        self.failure_link = None

class CompressedTrieNode:
    """Compressed trie node for memory efficiency"""
    __slots__ = ['edge_label', 'children', 'is_end', 'patterns']
    
    def __init__(self, edge_label: str = ""):
        self.edge_label = edge_label
        self.children = {}
        self.is_end = False
        self.patterns = []

class PatternOptimizer:
    
    def __init__(self):
        self.cache = {}
        self.preprocessed_patterns = {}
    
    def memory_optimized_trie(self, patterns: List[str]) -> OptimizedTrieNode:
        """
        Approach 1: Memory-Optimized Trie Construction
        
        Build memory-efficient trie using __slots__ and compression.
        
        Time: O(Σ|pattern_i|)
        Space: O(unique_prefixes) - significantly reduced
        """
        root = OptimizedTrieNode()
        
        for pattern in patterns:
            node = root
            for char in pattern:
                if char not in node.children:
                    node.children[char] = OptimizedTrieNode()
                node = node.children[char]
            node.is_end = True
            node.patterns.append(pattern)
        
        # Compress trie to save memory
        self._compress_trie(root)
        
        return root
    
    def _compress_trie(self, node: OptimizedTrieNode, parent: OptimizedTrieNode = None, edge_char: str = None) -> None:
        """Compress single-child chains in trie"""
        # If node has exactly one child and is not an end node, compress
        while len(node.children) == 1 and not node.is_end and not node.patterns:
            char, child = next(iter(node.children.items()))
            
            # Merge with child
            if parent and edge_char:
                parent.children[edge_char + char] = child
                del parent.children[edge_char]
            
            node = child
        
        # Recursively compress children
        for char, child in list(node.children.items()):
            self._compress_trie(child, node, char)
    
    def parallel_pattern_search(self, text: str, patterns: List[str], num_threads: int = 4) -> Dict[str, List[int]]:
        """
        Approach 2: Parallel Pattern Matching
        
        Divide patterns among threads for parallel processing.
        
        Time: O((|text| + Σ|pattern_i|) / num_threads)
        Space: O(Σ|pattern_i|)
        """
        # Divide patterns among threads
        chunk_size = len(patterns) // num_threads
        pattern_chunks = [patterns[i:i + chunk_size] for i in range(0, len(patterns), chunk_size)]
        
        results = {}
        
        def search_chunk(pattern_chunk: List[str]) -> Dict[str, List[int]]:
            chunk_results = {}
            for pattern in pattern_chunk:
                positions = self._naive_search(text, pattern)
                chunk_results[pattern] = positions
            return chunk_results
        
        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_chunk = {executor.submit(search_chunk, chunk): chunk for chunk in pattern_chunks}
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_results = future.result()
                results.update(chunk_results)
        
        return results
    
    def _naive_search(self, text: str, pattern: str) -> List[int]:
        """Simple naive pattern search"""
        positions = []
        for i in range(len(text) - len(pattern) + 1):
            if text[i:i + len(pattern)] == pattern:
                positions.append(i)
        return positions
    
    def cache_optimized_search(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Approach 3: Cache-Optimized Pattern Search
        
        Use caching to avoid recomputation of similar patterns.
        
        Time: O(|text| + Σ|pattern_i|) with cache hits
        Space: O(cache_size)
        """
        results = {}
        
        for pattern in patterns:
            # Check cache first
            cache_key = (text, pattern)
            if cache_key in self.cache:
                results[pattern] = self.cache[cache_key]
                continue
            
            # Compute and cache result
            positions = self._optimized_kmp_search(text, pattern)
            self.cache[cache_key] = positions
            results[pattern] = positions
            
            # Limit cache size
            if len(self.cache) > 1000:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
        
        return results
    
    @lru_cache(maxsize=256)
    def _build_failure_function_cached(self, pattern: str) -> Tuple[int, ...]:
        """Cached failure function computation"""
        m = len(pattern)
        failure = [0] * m
        j = 0
        
        for i in range(1, m):
            while j > 0 and pattern[i] != pattern[j]:
                j = failure[j - 1]
            
            if pattern[i] == pattern[j]:
                j += 1
            
            failure[i] = j
        
        return tuple(failure)
    
    def _optimized_kmp_search(self, text: str, pattern: str) -> List[int]:
        """KMP search with cached failure function"""
        if not pattern:
            return []
        
        failure = list(self._build_failure_function_cached(pattern))
        positions = []
        i = j = 0
        
        while i < len(text):
            if text[i] == pattern[j]:
                i += 1
                j += 1
            
            if j == len(pattern):
                positions.append(i - j)
                j = failure[j - 1]
            elif i < len(text) and text[i] != pattern[j]:
                if j != 0:
                    j = failure[j - 1]
                else:
                    i += 1
        
        return positions
    
    def preprocessing_optimized_search(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Approach 4: Preprocessing-Optimized Search
        
        Preprocess patterns for various optimizations.
        
        Time: O(preprocessing + optimized_search)
        Space: O(preprocessed_data)
        """
        # Group patterns by length for batch processing
        patterns_by_length = defaultdict(list)
        for pattern in patterns:
            patterns_by_length[len(pattern)].append(pattern)
        
        results = {}
        
        for length, length_patterns in patterns_by_length.items():
            if length == 1:
                # Optimize single character patterns
                char_positions = self._find_all_char_positions(text)
                for pattern in length_patterns:
                    results[pattern] = char_positions.get(pattern, [])
            else:
                # Use rolling hash for longer patterns
                length_results = self._rolling_hash_batch_search(text, length_patterns)
                results.update(length_results)
        
        return results
    
    def _find_all_char_positions(self, text: str) -> Dict[str, List[int]]:
        """Find positions of all characters in text"""
        char_positions = defaultdict(list)
        for i, char in enumerate(text):
            char_positions[char].append(i)
        return dict(char_positions)
    
    def _rolling_hash_batch_search(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """Rolling hash search for patterns of same length"""
        if not patterns:
            return {}
        
        pattern_length = len(patterns[0])
        base = 256
        mod = 10**9 + 7
        
        # Compute pattern hashes
        pattern_hashes = {}
        for pattern in patterns:
            pattern_hash = 0
            for char in pattern:
                pattern_hash = (pattern_hash * base + ord(char)) % mod
            pattern_hashes[pattern] = pattern_hash
        
        results = {pattern: [] for pattern in patterns}
        
        if pattern_length > len(text):
            return results
        
        # Compute rolling hash
        text_hash = 0
        h = 1
        
        # h = base^(pattern_length-1) % mod
        for _ in range(pattern_length - 1):
            h = (h * base) % mod
        
        # Initial hash
        for i in range(pattern_length):
            text_hash = (text_hash * base + ord(text[i])) % mod
        
        # Check initial window
        self._check_hash_matches(text, 0, pattern_length, text_hash, pattern_hashes, results)
        
        # Roll through text
        for i in range(pattern_length, len(text)):
            # Remove leading character and add trailing character
            text_hash = (text_hash - ord(text[i - pattern_length]) * h) % mod
            text_hash = (text_hash * base + ord(text[i])) % mod
            
            if text_hash < 0:
                text_hash += mod
            
            start_pos = i - pattern_length + 1
            self._check_hash_matches(text, start_pos, pattern_length, text_hash, pattern_hashes, results)
        
        return results
    
    def _check_hash_matches(self, text: str, start: int, length: int, text_hash: int, 
                           pattern_hashes: Dict[str, int], results: Dict[str, List[int]]) -> None:
        """Check if hash matches any pattern and verify actual string"""
        substring = text[start:start + length]
        for pattern, pattern_hash in pattern_hashes.items():
            if text_hash == pattern_hash and substring == pattern:
                results[pattern].append(start)
    
    def bit_parallel_search(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Approach 5: Bit-Parallel Pattern Matching
        
        Use bitwise operations for parallel character matching.
        
        Time: O(|text| * |patterns| / word_size)
        Space: O(|patterns|)
        """
        results = {}
        
        # Filter patterns that fit in machine word
        max_pattern_length = 64  # Assume 64-bit machine
        
        for pattern in patterns:
            if len(pattern) <= max_pattern_length:
                positions = self._bit_parallel_single_pattern(text, pattern)
                results[pattern] = positions
            else:
                # Fall back to regular algorithm for long patterns
                positions = self._optimized_kmp_search(text, pattern)
                results[pattern] = positions
        
        return results
    
    def _bit_parallel_single_pattern(self, text: str, pattern: str) -> List[int]:
        """Bit-parallel search for single pattern"""
        if not pattern:
            return []
        
        m = len(pattern)
        positions = []
        
        # Build character masks
        char_masks = {}
        for i, char in enumerate(pattern):
            if char not in char_masks:
                char_masks[char] = 0
            char_masks[char] |= (1 << i)
        
        # Search using bit operations
        state = 0
        match_mask = 1 << (m - 1)
        
        for i, char in enumerate(text):
            # Shift state and add new character
            state = ((state << 1) | 1) & char_masks.get(char, 0)
            
            # Check for match
            if state & match_mask:
                positions.append(i - m + 1)
        
        return positions
    
    def adaptive_algorithm_selection(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Approach 6: Adaptive Algorithm Selection
        
        Choose optimal algorithm based on input characteristics.
        
        Time: Varies based on selected algorithm
        Space: Varies based on selected algorithm
        """
        text_len = len(text)
        num_patterns = len(patterns)
        avg_pattern_len = sum(len(p) for p in patterns) / len(patterns) if patterns else 0
        
        # Decision logic based on characteristics
        if num_patterns == 1:
            # Single pattern - use KMP
            pattern = patterns[0]
            positions = self._optimized_kmp_search(text, pattern)
            return {pattern: positions}
        
        elif num_patterns > 10 and avg_pattern_len < 10:
            # Many short patterns - use Aho-Corasick
            return self._aho_corasick_optimized(text, patterns)
        
        elif text_len > 10000 and num_patterns < 5:
            # Long text, few patterns - use parallel search
            return self.parallel_pattern_search(text, patterns)
        
        elif all(len(p) == len(patterns[0]) for p in patterns):
            # Same length patterns - use rolling hash
            return self._rolling_hash_batch_search(text, patterns)
        
        else:
            # Default to preprocessing optimized
            return self.preprocessing_optimized_search(text, patterns)
    
    def _aho_corasick_optimized(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """Optimized Aho-Corasick implementation"""
        # Build optimized trie
        root = self.memory_optimized_trie(patterns)
        
        # Build failure links efficiently
        self._build_failure_links_optimized(root)
        
        # Search with optimizations
        results = defaultdict(list)
        current = root
        
        for i, char in enumerate(text):
            # Follow failure links
            while current != root and char not in current.children:
                current = current.failure_link
            
            if char in current.children:
                current = current.children[char]
                
                # Check for matches
                if current.is_end:
                    for pattern in current.patterns:
                        start_pos = i - len(pattern) + 1
                        results[pattern].append(start_pos)
        
        return dict(results)
    
    def _build_failure_links_optimized(self, root: OptimizedTrieNode) -> None:
        """Build failure links with optimizations"""
        queue = deque()
        
        # Initialize first level
        for child in root.children.values():
            child.failure_link = root
            queue.append(child)
        
        # BFS to build failure links
        while queue:
            current = queue.popleft()
            
            for char, child in current.children.items():
                queue.append(child)
                
                # Find failure link
                failure = current.failure_link
                while failure != root and char not in failure.children:
                    failure = failure.failure_link
                
                child.failure_link = failure.children.get(char, root)
    
    def streaming_pattern_search(self, text_stream: Callable[[], str], patterns: List[str], 
                                buffer_size: int = 1024) -> Dict[str, List[int]]:
        """
        Approach 7: Streaming Pattern Search
        
        Search patterns in streaming text with limited memory.
        
        Time: O(stream_length + Σ|pattern_i|)
        Space: O(buffer_size + trie_size)
        """
        # Build automaton
        root = self.memory_optimized_trie(patterns)
        self._build_failure_links_optimized(root)
        
        results = defaultdict(list)
        buffer = ""
        position = 0
        current = root
        
        max_pattern_len = max(len(p) for p in patterns) if patterns else 0
        
        while True:
            # Read next chunk
            chunk = text_stream()
            if not chunk:
                break
            
            buffer += chunk
            
            # Process buffer
            for i, char in enumerate(buffer):
                # Standard Aho-Corasick processing
                while current != root and char not in current.children:
                    current = current.failure_link
                
                if char in current.children:
                    current = current.children[char]
                    
                    if current.is_end:
                        for pattern in current.patterns:
                            match_start = position + i - len(pattern) + 1
                            results[pattern].append(match_start)
            
            # Update position and maintain buffer
            position += len(buffer) - max_pattern_len
            buffer = buffer[-max_pattern_len:] if len(buffer) > max_pattern_len else ""
        
        return dict(results)


def test_optimization_techniques():
    """Test various optimization techniques"""
    print("=== Testing Optimization Techniques ===")
    
    optimizer = PatternOptimizer()
    
    text = "abcdefghijklmnopqrstuvwxyz" * 100  # Long text
    patterns = ["abc", "def", "xyz", "rst", "klm"]
    
    print(f"Text length: {len(text)}")
    print(f"Patterns: {patterns}")
    
    techniques = [
        ("Cache Optimized", optimizer.cache_optimized_search),
        ("Preprocessing Optimized", optimizer.preprocessing_optimized_search),
        ("Bit Parallel", optimizer.bit_parallel_search),
        ("Adaptive Selection", optimizer.adaptive_algorithm_selection),
    ]
    
    for name, technique in techniques:
        print(f"\n{name}:")
        try:
            result = technique(text, patterns)
            total_matches = sum(len(positions) for positions in result.values())
            print(f"  Total matches found: {total_matches}")
            
            for pattern, positions in result.items():
                print(f"    '{pattern}': {len(positions)} matches")
        except Exception as e:
            print(f"  Error: {e}")


def test_memory_optimization():
    """Test memory optimization techniques"""
    print("\n=== Testing Memory Optimization ===")
    
    import sys
    
    optimizer = PatternOptimizer()
    
    # Large set of patterns
    patterns = [f"pattern{i:04d}" for i in range(1000)]
    
    print(f"Testing with {len(patterns)} patterns")
    
    # Build regular trie
    regular_trie = optimizer.memory_optimized_trie(patterns[:100])  # Smaller set for demo
    
    print(f"Memory-optimized trie constructed successfully")
    print(f"Trie uses __slots__ for reduced memory overhead")
    
    # Test search
    text = "pattern0001pattern0002pattern0003"
    search_patterns = ["pattern0001", "pattern0002", "pattern0999"]
    
    results = optimizer._aho_corasick_optimized(text, search_patterns)
    
    print(f"\nSearch results:")
    for pattern, positions in results.items():
        print(f"  '{pattern}': found at positions {positions}")


def test_parallel_processing():
    """Test parallel pattern matching"""
    print("\n=== Testing Parallel Processing ===")
    
    import time
    
    optimizer = PatternOptimizer()
    
    # Generate large dataset
    text = "abcdefghijklmnopqrstuvwxyz" * 1000
    patterns = [f"{chr(97+i)}{chr(97+j)}{chr(97+k)}" for i in range(5) for j in range(5) for k in range(5)]
    
    print(f"Text length: {len(text)}")
    print(f"Number of patterns: {len(patterns)}")
    
    # Test sequential vs parallel
    start_time = time.time()
    sequential_results = optimizer.cache_optimized_search(text, patterns)
    sequential_time = time.time() - start_time
    
    start_time = time.time()
    parallel_results = optimizer.parallel_pattern_search(text, patterns, num_threads=4)
    parallel_time = time.time() - start_time
    
    print(f"\nPerformance comparison:")
    print(f"  Sequential: {sequential_time:.3f}s")
    print(f"  Parallel (4 threads): {parallel_time:.3f}s")
    print(f"  Speedup: {sequential_time/parallel_time:.2f}x")
    
    # Verify results are identical
    results_match = all(
        sequential_results.get(pattern, []) == parallel_results.get(pattern, [])
        for pattern in patterns
    )
    print(f"  Results identical: {results_match}")


def test_adaptive_selection():
    """Test adaptive algorithm selection"""
    print("\n=== Testing Adaptive Algorithm Selection ===")
    
    optimizer = PatternOptimizer()
    
    test_scenarios = [
        ("Single long pattern", "abcdefghijklmnop" * 100, ["abcdefghijklmnop"]),
        ("Many short patterns", "abcabc" * 50, ["a", "b", "c", "ab", "bc", "abc"]),
        ("Same length patterns", "hello world hello", ["hello", "world", "ellow"]),
        ("Mixed patterns", "programming", ["prog", "gram", "amming", "p", "programming"]),
    ]
    
    for scenario_name, text, patterns in test_scenarios:
        print(f"\n{scenario_name}:")
        print(f"  Text: '{text[:30]}{'...' if len(text) > 30 else ''}'")
        print(f"  Patterns: {patterns}")
        
        results = optimizer.adaptive_algorithm_selection(text, patterns)
        
        total_matches = sum(len(positions) for positions in results.values())
        print(f"  Total matches: {total_matches}")
        
        for pattern, positions in results.items():
            if positions:
                print(f"    '{pattern}': {len(positions)} matches")


def demonstrate_streaming_search():
    """Demonstrate streaming pattern search"""
    print("\n=== Streaming Pattern Search Demo ===")
    
    optimizer = PatternOptimizer()
    
    # Simulate text stream
    text_data = "The quick brown fox jumps over the lazy dog. " * 20
    chunk_size = 50
    current_pos = 0
    
    def text_stream():
        nonlocal current_pos
        if current_pos >= len(text_data):
            return ""
        
        chunk = text_data[current_pos:current_pos + chunk_size]
        current_pos += chunk_size
        return chunk
    
    patterns = ["the", "fox", "lazy", "dog", "quick"]
    
    print(f"Streaming search for patterns: {patterns}")
    print(f"Text data length: {len(text_data)}")
    print(f"Chunk size: {chunk_size}")
    
    results = optimizer.streaming_pattern_search(text_stream, patterns)
    
    print(f"\nStreaming search results:")
    for pattern, positions in results.items():
        print(f"  '{pattern}': found {len(positions)} times at positions {positions[:5]}{'...' if len(positions) > 5 else ''}")


def benchmark_optimization_techniques():
    """Benchmark different optimization techniques"""
    print("\n=== Benchmarking Optimization Techniques ===")
    
    import time
    import random
    import string
    
    optimizer = PatternOptimizer()
    
    # Generate test data
    text = ''.join(random.choices(string.ascii_lowercase, k=10000))
    patterns = [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8))) 
               for _ in range(100)]
    
    techniques = [
        ("Basic Search", lambda t, p: {pat: optimizer._naive_search(t, pat) for pat in p}),
        ("Cache Optimized", optimizer.cache_optimized_search),
        ("Preprocessing Optimized", optimizer.preprocessing_optimized_search),
        ("Adaptive Selection", optimizer.adaptive_algorithm_selection),
    ]
    
    print(f"Text length: {len(text)}")
    print(f"Number of patterns: {len(patterns)}")
    
    for name, technique in techniques:
        start_time = time.time()
        
        try:
            results = technique(text, patterns)
            total_matches = sum(len(positions) for positions in results.values())
            
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            
            print(f"\n{name:20}: {duration:6.2f}ms ({total_matches} matches)")
        except Exception as e:
            print(f"\n{name:20}: Error - {e}")


def analyze_optimization_strategies():
    """Analyze different optimization strategies"""
    print("\n=== Optimization Strategy Analysis ===")
    
    strategies = [
        ("Memory Optimization",
         "• Use __slots__ in trie nodes",
         "• Compress single-child chains",
         "• Share common prefixes"),
        
        ("Preprocessing Optimization",
         "• Group patterns by length",
         "• Precompute character positions",
         "• Cache failure functions"),
        
        ("Algorithm Selection",
         "• Single pattern → KMP",
         "• Many patterns → Aho-Corasick",
         "• Same length → Rolling hash"),
        
        ("Parallel Processing",
         "• Divide patterns among threads",
         "• Use thread pools for management",
         "• Minimize synchronization overhead"),
        
        ("Cache Optimization",
         "• LRU cache for repeated patterns",
         "• Memoize expensive computations",
         "• Locality-aware data structures"),
        
        ("Bit Parallel Techniques",
         "• Use bitwise operations",
         "• Parallel character matching",
         "• Efficient for short patterns"),
        
        ("Streaming Optimization",
         "• Process data in chunks",
         "• Maintain sliding window",
         "• Minimize memory usage"),
    ]
    
    print("Optimization Strategy Analysis:")
    
    for strategy_name, *techniques in strategies:
        print(f"\n{strategy_name}:")
        for technique in techniques:
            print(f"  {technique}")
    
    print(f"\nGeneral Guidelines:")
    print(f"  • Profile before optimizing")
    print(f"  • Consider input characteristics")
    print(f"  • Balance time vs space trade-offs")
    print(f"  • Use appropriate data structures")
    print(f"  • Leverage hardware capabilities")


if __name__ == "__main__":
    test_optimization_techniques()
    test_memory_optimization()
    test_parallel_processing()
    test_adaptive_selection()
    demonstrate_streaming_search()
    benchmark_optimization_techniques()
    analyze_optimization_strategies()

"""
Pattern Optimization Techniques demonstrates advanced optimization strategies:

1. Memory-Optimized Trie - Use __slots__ and compression for reduced memory
2. Parallel Pattern Search - Divide patterns among threads for speedup
3. Cache-Optimized Search - LRU caching for repeated pattern computations
4. Preprocessing Optimization - Group patterns and precompute structures
5. Bit-Parallel Search - Use bitwise operations for parallel matching
6. Adaptive Algorithm Selection - Choose optimal algorithm based on input
7. Streaming Pattern Search - Process data streams with limited memory

Each technique targets specific bottlenecks and use cases to achieve
optimal performance in different pattern matching scenarios.
"""
