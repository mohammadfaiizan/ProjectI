"""
Fast I/O String Processing - Multiple Approaches
Difficulty: Hard

High-performance string processing techniques for competitive programming
with focus on fast input/output and efficient string operations.

Techniques:
1. Fast Input/Output Methods
2. Bulk String Processing
3. Memory-Mapped File Processing
4. Streaming String Operations
5. Parallel String Processing
6. Cache-Optimized Algorithms
"""

import sys
import io
from typing import List, Dict, Iterator, Generator
import mmap
import threading
from collections import deque
import time

class FastIO:
    """Fast I/O class for competitive programming"""
    
    def __init__(self):
        self.input_buffer = []
        self.input_index = 0
    
    def fast_input(self) -> str:
        """Fast input reading"""
        if self.input_index >= len(self.input_buffer):
            # Read entire input at once
            self.input_buffer = sys.stdin.read().strip().split()
            self.input_index = 0
        
        if self.input_index < len(self.input_buffer):
            result = self.input_buffer[self.input_index]
            self.input_index += 1
            return result
        return ""
    
    def fast_int(self) -> int:
        """Fast integer input"""
        return int(self.fast_input())
    
    def fast_ints(self, n: int) -> List[int]:
        """Fast multiple integer input"""
        return [self.fast_int() for _ in range(n)]
    
    def fast_output(self, *args) -> None:
        """Fast output writing"""
        print(*args)

class TrieNode:
    __slots__ = ['children', 'is_end', 'count', 'depth']
    
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0
        self.depth = 0

class FastStringProcessor:
    
    def __init__(self):
        self.root = TrieNode()
        self.fast_io = FastIO()
        self.buffer_size = 8192
    
    def bulk_string_insertion(self, strings: List[str]) -> Dict[str, int]:
        """
        Approach 1: Bulk String Processing
        
        Process multiple strings efficiently in batches.
        
        Time: O(total_length)
        Space: O(trie_size)
        """
        # Sort strings for better cache locality
        sorted_strings = sorted(strings)
        
        # Batch processing
        batch_size = 1000
        statistics = {'total_nodes': 0, 'max_depth': 0, 'duplicates': 0}
        
        for i in range(0, len(sorted_strings), batch_size):
            batch = sorted_strings[i:i + batch_size]
            self._process_string_batch(batch, statistics)
        
        return statistics
    
    def _process_string_batch(self, batch: List[str], stats: Dict[str, int]) -> None:
        """Process a batch of strings"""
        for string in batch:
            node = self.root
            depth = 0
            
            for char in string:
                if char not in node.children:
                    node.children[char] = TrieNode()
                    stats['total_nodes'] += 1
                
                node = node.children[char]
                depth += 1
                node.depth = depth
            
            if node.is_end:
                stats['duplicates'] += 1
            else:
                node.is_end = True
            
            node.count += 1
            stats['max_depth'] = max(stats['max_depth'], depth)
    
    def memory_mapped_processing(self, filename: str) -> Dict[str, int]:
        """
        Approach 2: Memory-Mapped File Processing
        
        Process large files using memory mapping for efficiency.
        
        Time: O(file_size)
        Space: O(unique_strings)
        """
        word_count = {}
        
        try:
            with open(filename, 'r+b') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    # Read in chunks for better performance
                    chunk_size = self.buffer_size
                    buffer = ""
                    
                    for i in range(0, len(mmapped_file), chunk_size):
                        chunk = mmapped_file[i:i + chunk_size].decode('utf-8', errors='ignore')
                        buffer += chunk
                        
                        # Process complete words
                        lines = buffer.split('\n')
                        buffer = lines[-1]  # Keep incomplete line
                        
                        for line in lines[:-1]:
                            words = line.split()
                            for word in words:
                                word = word.strip().lower()
                                if word:
                                    word_count[word] = word_count.get(word, 0) + 1
                    
                    # Process remaining buffer
                    if buffer:
                        words = buffer.split()
                        for word in words:
                            word = word.strip().lower()
                            if word:
                                word_count[word] = word_count.get(word, 0) + 1
        
        except FileNotFoundError:
            print(f"File {filename} not found. Using sample data.")
            # Use sample data for demonstration
            sample_words = ["hello", "world", "fast", "processing", "trie", "algorithm"]
            for word in sample_words:
                word_count[word] = word_count.get(word, 0) + 1
        
        return word_count
    
    def streaming_string_operations(self, string_generator: Generator[str, None, None]) -> Dict[str, any]:
        """
        Approach 3: Streaming String Operations
        
        Process strings as they arrive without storing all in memory.
        
        Time: O(stream_length)
        Space: O(unique_prefixes)
        """
        streaming_stats = {
            'processed_count': 0,
            'unique_prefixes': 0,
            'longest_string': 0,
            'avg_length': 0,
            'total_length': 0
        }
        
        prefix_trie = TrieNode()
        
        for string in string_generator:
            streaming_stats['processed_count'] += 1
            streaming_stats['total_length'] += len(string)
            streaming_stats['longest_string'] = max(streaming_stats['longest_string'], len(string))
            
            # Update trie with all prefixes
            node = prefix_trie
            for i, char in enumerate(string):
                if char not in node.children:
                    node.children[char] = TrieNode()
                    streaming_stats['unique_prefixes'] += 1
                
                node = node.children[char]
            
            # Update running average
            if streaming_stats['processed_count'] > 0:
                streaming_stats['avg_length'] = (
                    streaming_stats['total_length'] / streaming_stats['processed_count']
                )
        
        return streaming_stats
    
    def parallel_string_processing(self, strings: List[str], num_threads: int = 4) -> Dict[str, any]:
        """
        Approach 4: Parallel String Processing
        
        Process strings using multiple threads for better performance.
        
        Time: O(total_length / num_threads)
        Space: O(trie_size * num_threads)
        """
        chunk_size = len(strings) // num_threads
        results = [None] * num_threads
        threads = []
        
        def process_chunk(chunk: List[str], thread_id: int) -> None:
            """Process a chunk of strings in a separate thread"""
            local_trie = TrieNode()
            local_stats = {
                'thread_id': thread_id,
                'processed': len(chunk),
                'unique_words': 0,
                'total_chars': sum(len(s) for s in chunk)
            }
            
            for string in chunk:
                node = local_trie
                is_new_word = True
                
                for char in string:
                    if char not in node.children:
                        node.children[char] = TrieNode()
                    node = node.children[char]
                
                if node.is_end:
                    is_new_word = False
                else:
                    node.is_end = True
                
                if is_new_word:
                    local_stats['unique_words'] += 1
            
            results[thread_id] = local_stats
        
        # Create and start threads
        for i in range(num_threads):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_threads - 1 else len(strings)
            chunk = strings[start_idx:end_idx]
            
            thread = threading.Thread(target=process_chunk, args=(chunk, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Merge results
        merged_stats = {
            'total_processed': sum(r['processed'] for r in results if r),
            'total_unique': sum(r['unique_words'] for r in results if r),
            'total_chars': sum(r['total_chars'] for r in results if r),
            'thread_results': results
        }
        
        return merged_stats
    
    def cache_optimized_search(self, trie_root: TrieNode, queries: List[str]) -> List[bool]:
        """
        Approach 5: Cache-Optimized Search
        
        Optimize search operations for better cache performance.
        
        Time: O(queries * avg_query_length)
        Space: O(1) additional
        """
        # Sort queries by length for better cache locality
        sorted_queries = sorted(enumerate(queries), key=lambda x: len(x[1]))
        results = [False] * len(queries)
        
        # Process queries in batches of similar length
        batch_size = 100
        
        for i in range(0, len(sorted_queries), batch_size):
            batch = sorted_queries[i:i + batch_size]
            
            for original_idx, query in batch:
                node = trie_root
                found = True
                
                for char in query:
                    if char not in node.children:
                        found = False
                        break
                    node = node.children[char]
                
                results[original_idx] = found and node.is_end
        
        return results
    
    def fast_prefix_counting(self, strings: List[str], prefixes: List[str]) -> Dict[str, int]:
        """
        Approach 6: Fast Prefix Counting
        
        Count strings with given prefixes efficiently.
        
        Time: O(total_string_length + total_prefix_length)
        Space: O(trie_size)
        """
        # Build trie with count tracking
        for string in strings:
            node = self.root
            for char in string:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                node.count += 1  # Count strings passing through this node
            node.is_end = True
        
        # Count prefixes
        prefix_counts = {}
        
        for prefix in prefixes:
            node = self.root
            count = 0
            
            # Navigate to prefix end
            valid_prefix = True
            for char in prefix:
                if char not in node.children:
                    valid_prefix = False
                    break
                node = node.children[char]
            
            if valid_prefix:
                count = node.count
            
            prefix_counts[prefix] = count
        
        return prefix_counts
    
    def optimize_for_contest(self, problem_type: str) -> Dict[str, any]:
        """
        Approach 7: Contest-Specific Optimizations
        
        Apply optimizations based on problem type.
        
        Time: Varies by optimization
        Space: Varies by optimization
        """
        optimizations = {}
        
        if problem_type == "string_matching":
            optimizations.update({
                'use_rolling_hash': True,
                'precompute_powers': True,
                'batch_queries': True,
                'early_termination': True
            })
        
        elif problem_type == "trie_operations":
            optimizations.update({
                'compress_single_child_paths': True,
                'use_arrays_for_children': True,
                'bit_manipulation_for_sets': True,
                'memory_pool_allocation': True
            })
        
        elif problem_type == "large_input":
            optimizations.update({
                'fast_io': True,
                'memory_mapping': True,
                'streaming_processing': True,
                'parallel_processing': True
            })
        
        elif problem_type == "time_critical":
            optimizations.update({
                'inline_functions': True,
                'avoid_function_calls': True,
                'use_bitwise_operations': True,
                'cache_friendly_layout': True
            })
        
        return optimizations


def test_fast_io():
    """Test fast I/O operations"""
    print("=== Testing Fast I/O ===")
    
    # Simulate fast input (normally would read from stdin)
    test_input = "5\n1 2 3 4 5\nhello world test"
    
    # Mock stdin for testing
    import io
    sys.stdin = io.StringIO(test_input)
    
    fast_io = FastIO()
    
    n = fast_io.fast_int()
    numbers = fast_io.fast_ints(n)
    
    print(f"Read n = {n}")
    print(f"Read numbers = {numbers}")
    
    # Read remaining strings
    strings = []
    for _ in range(3):  # Read 3 more strings
        s = fast_io.fast_input()
        if s:
            strings.append(s)
    
    print(f"Read strings = {strings}")
    
    # Restore stdin
    sys.stdin = sys.__stdin__

def test_bulk_processing():
    """Test bulk string processing"""
    print("\n=== Testing Bulk Processing ===")
    
    processor = FastStringProcessor()
    
    # Generate test strings
    import random
    import string
    
    def generate_strings(count: int) -> List[str]:
        strings = []
        for _ in range(count):
            length = random.randint(3, 10)
            s = ''.join(random.choices(string.ascii_lowercase, k=length))
            strings.append(s)
        return strings
    
    test_sizes = [1000, 5000, 10000]
    
    print(f"{'Size':<8} {'Time(ms)':<12} {'Nodes':<8} {'Max Depth':<10} {'Duplicates':<10}")
    print("-" * 55)
    
    for size in test_sizes:
        strings = generate_strings(size)
        
        # Reset trie for each test
        processor.root = TrieNode()
        
        start_time = time.time()
        stats = processor.bulk_string_insertion(strings)
        elapsed = (time.time() - start_time) * 1000
        
        print(f"{size:<8} {elapsed:<12.2f} {stats['total_nodes']:<8} "
              f"{stats['max_depth']:<10} {stats['duplicates']:<10}")

def test_streaming_operations():
    """Test streaming string operations"""
    print("\n=== Testing Streaming Operations ===")
    
    processor = FastStringProcessor()
    
    def string_generator() -> Generator[str, None, None]:
        """Generate strings for streaming test"""
        import random
        import string
        
        for _ in range(1000):
            length = random.randint(2, 15)
            s = ''.join(random.choices(string.ascii_lowercase, k=length))
            yield s
    
    print("Processing streaming strings...")
    
    start_time = time.time()
    stats = processor.streaming_string_operations(string_generator())
    elapsed = (time.time() - start_time) * 1000
    
    print(f"Streaming processing completed in {elapsed:.2f}ms")
    print("Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

def test_parallel_processing():
    """Test parallel string processing"""
    print("\n=== Testing Parallel Processing ===")
    
    processor = FastStringProcessor()
    
    # Generate test data
    import random
    import string
    
    def generate_test_strings(count: int) -> List[str]:
        return [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8))) 
                for _ in range(count)]
    
    strings = generate_test_strings(10000)
    
    # Test different thread counts
    thread_counts = [1, 2, 4, 8]
    
    print(f"{'Threads':<8} {'Time(ms)':<12} {'Processed':<10} {'Unique':<8}")
    print("-" * 45)
    
    for num_threads in thread_counts:
        start_time = time.time()
        results = processor.parallel_string_processing(strings, num_threads)
        elapsed = (time.time() - start_time) * 1000
        
        print(f"{num_threads:<8} {elapsed:<12.2f} {results['total_processed']:<10} "
              f"{results['total_unique']:<8}")

def test_cache_optimization():
    """Test cache-optimized operations"""
    print("\n=== Testing Cache Optimization ===")
    
    processor = FastStringProcessor()
    
    # Build test trie
    test_words = ["apple", "app", "application", "apply", "banana", "band", "bandana"]
    
    for word in test_words:
        node = processor.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    # Test queries
    queries = ["app", "apple", "ban", "banana", "xyz", "application", "band"]
    
    print(f"Test words: {test_words}")
    print(f"Queries: {queries}")
    
    start_time = time.time()
    results = processor.cache_optimized_search(processor.root, queries)
    elapsed = (time.time() - start_time) * 1000
    
    print(f"Search completed in {elapsed:.2f}ms")
    print("Results:")
    for query, found in zip(queries, results):
        print(f"  '{query}': {'Found' if found else 'Not found'}")

def benchmark_optimization_techniques():
    """Benchmark different optimization techniques"""
    print("\n=== Benchmarking Optimization Techniques ===")
    
    processor = FastStringProcessor()
    
    # Test data
    import random
    import string
    
    strings = [''.join(random.choices(string.ascii_lowercase, k=random.randint(5, 12))) 
               for _ in range(5000)]
    
    print("Optimization techniques comparison:")
    
    # Technique 1: Standard processing
    start_time = time.time()
    standard_stats = processor.bulk_string_insertion(strings[:1000])
    standard_time = (time.time() - start_time) * 1000
    
    print(f"1. Standard processing (1000 strings): {standard_time:.2f}ms")
    
    # Technique 2: Parallel processing
    start_time = time.time()
    parallel_stats = processor.parallel_string_processing(strings[:1000], 4)
    parallel_time = (time.time() - start_time) * 1000
    
    speedup = standard_time / parallel_time if parallel_time > 0 else 0
    print(f"2. Parallel processing (4 threads): {parallel_time:.2f}ms (speedup: {speedup:.2f}x)")
    
    # Technique 3: Streaming processing
    def string_stream():
        for s in strings[:1000]:
            yield s
    
    start_time = time.time()
    streaming_stats = processor.streaming_string_operations(string_stream())
    streaming_time = (time.time() - start_time) * 1000
    
    print(f"3. Streaming processing: {streaming_time:.2f}ms")
    
    # Show contest-specific optimizations
    print(f"\n4. Contest-specific optimizations:")
    
    optimizations = processor.optimize_for_contest("trie_operations")
    for opt, enabled in optimizations.items():
        print(f"   {opt}: {'Enabled' if enabled else 'Disabled'}")

def demonstrate_contest_techniques():
    """Demonstrate contest programming techniques"""
    print("\n=== Contest Programming Techniques ===")
    
    techniques = [
        "Fast I/O: Read all input at once, minimize output calls",
        "Memory optimization: Use arrays instead of dictionaries when possible",
        "Preprocessing: Build auxiliary structures before processing queries",
        "Batch processing: Group similar operations together",
        "Early termination: Stop processing when answer is found",
        "Bit manipulation: Use bitwise operations for set operations",
        "Cache optimization: Access memory in predictable patterns",
        "Template code: Prepare reusable code templates"
    ]
    
    print("Key techniques for competitive programming:")
    for i, technique in enumerate(techniques, 1):
        print(f"{i}. {technique}")
    
    print(f"\nTime complexity targets:")
    print(f"  • 10^6 operations: O(n) or O(n log n)")
    print(f"  • 10^5 operations: O(n log n) or O(n log^2 n)")
    print(f"  • 10^4 operations: O(n^2) acceptable")
    print(f"  • 10^3 operations: O(n^3) acceptable")
    
    print(f"\nMemory usage guidelines:")
    print(f"  • Typical limit: 256MB or 512MB")
    print(f"  • Array of 10^6 integers: ~4MB")
    print(f"  • Array of 10^6 strings (avg 10 chars): ~10MB")
    print(f"  • Trie with 10^6 nodes: ~20-40MB")

if __name__ == "__main__":
    test_fast_io()
    test_bulk_processing()
    test_streaming_operations()
    test_parallel_processing()
    test_cache_optimization()
    benchmark_optimization_techniques()
    demonstrate_contest_techniques()

"""
Fast I/O String Processing demonstrates high-performance techniques:

Key Optimization Areas:
1. Fast I/O - Read all input at once, minimize system calls
2. Bulk Processing - Process data in batches for better cache performance
3. Memory Mapping - Handle large files without loading into memory
4. Streaming - Process data as it arrives without storing everything
5. Parallel Processing - Use multiple threads for CPU-intensive operations
6. Cache Optimization - Arrange data access patterns for better cache hits

Performance Considerations:
- I/O is often the bottleneck in competitive programming
- Memory access patterns significantly affect performance
- Preprocessing can trade memory for query speed
- Batch operations reduce function call overhead
- Parallel processing helps with independent computations

Contest-Specific Techniques:
- Template code for common operations
- Fast input/output implementations
- Bit manipulation for set operations
- Early termination strategies
- Memory usage optimization
- Time complexity analysis

These optimizations can provide 2-10x performance improvements
in competitive programming scenarios.
"""
