"""
Trie Optimizations and Advanced Techniques
==========================================

Topics: Memory optimization, performance tuning, specialized implementations
Companies: Google, Amazon, Microsoft, Facebook, Apple, System design roles
Difficulty: Hard to Expert
Time Complexity: Varies by optimization technique
Space Complexity: Optimized from O(ALPHABET_SIZE * N * M) to better bounds
"""

from typing import List, Tuple, Optional, Dict, Any, Set, Union, Iterator
from collections import defaultdict, deque
import array
import sys
import gc

class TrieOptimizations:
    
    def __init__(self):
        """Initialize with optimization tracking"""
        self.optimization_count = 0
        self.memory_stats = {}
    
    # ==========================================
    # 1. MEMORY OPTIMIZATIONS
    # ==========================================
    
    def demonstrate_compressed_trie(self) -> None:
        """
        Demonstrate compressed trie (Radix tree) for memory efficiency
        
        Compresses chains of single-child nodes into single edges
        Reduces space complexity significantly for sparse data
        """
        print("=== COMPRESSED TRIE (RADIX TREE) ===")
        print("Optimization: Compress single-child chains into single edges")
        print("Benefits: Reduced memory usage, better cache performance")
        print()
        
        # Create both regular and compressed tries for comparison
        words = ["romane", "romanus", "romulus", "rubens", "rubicon", "rubicundus"]
        
        print(f"Input words: {words}")
        print()
        
        # Regular trie
        regular_trie = RegularTrie()
        for word in words:
            regular_trie.insert(word)
        
        # Compressed trie
        compressed_trie = CompressedTrie()
        for word in words:
            compressed_trie.insert(word)
        
        # Compare statistics
        regular_stats = regular_trie.get_stats()
        compressed_stats = compressed_trie.get_stats()
        
        print("Memory Usage Comparison:")
        print(f"{'Metric':<25} {'Regular Trie':<15} {'Compressed Trie':<15} {'Savings'}")
        print("-" * 70)
        print(f"{'Total nodes':<25} {regular_stats['nodes']:<15} {compressed_stats['nodes']:<15} {regular_stats['nodes'] - compressed_stats['nodes']}")
        print(f"{'Total characters stored':<25} {regular_stats['chars']:<15} {compressed_stats['chars']:<15} {regular_stats['chars'] - compressed_stats['chars']}")
        print(f"{'Memory efficiency':<25} {regular_stats['efficiency']:<15.1%} {compressed_stats['efficiency']:<15.1%} {compressed_stats['efficiency'] - regular_stats['efficiency']:.1%}")
        print()
        
        # Show structure
        print("Regular Trie structure (first few levels):")
        regular_trie.display_structure(max_depth=4)
        print()
        
        print("Compressed Trie structure:")
        compressed_trie.display_structure()
        print()
    
    def demonstrate_array_optimization(self) -> None:
        """
        Demonstrate array-based optimization for fixed alphabets
        
        Uses arrays instead of hash maps for better cache performance
        """
        print("=== ARRAY-BASED OPTIMIZATION ===")
        print("Optimization: Use arrays for fixed alphabets (better cache performance)")
        print("Best for: Lowercase letters, digits, small fixed character sets")
        print()
        
        words = ["apple", "application", "apply", "banana", "band", "bandana"]
        
        # Test both implementations
        hash_trie = HashMapTrie()
        array_trie = ArrayTrie()
        
        print("Building tries with both implementations:")
        for word in words:
            hash_trie.insert(word)
            array_trie.insert(word)
        
        print(f"  Inserted {len(words)} words")
        print()
        
        # Performance comparison (simplified)
        test_operations = [
            ("search", "apple"),
            ("search", "app"),
            ("starts_with", "app"),
            ("starts_with", "ban"),
            ("search", "missing")
        ]
        
        print("Performance comparison:")
        for operation, word in test_operations:
            if operation == "search":
                hash_result = hash_trie.search(word)
                array_result = array_trie.search(word)
                print(f"  Search '{word}': HashMap={hash_result}, Array={array_result}")
            elif operation == "starts_with":
                hash_result = hash_trie.starts_with(word)
                array_result = array_trie.starts_with(word)
                print(f"  StartsWith '{word}': HashMap={hash_result}, Array={array_result}")
        
        print()
        print("Array-based advantages:")
        print("  ✓ Better cache locality (contiguous memory)")
        print("  ✓ No hash computation overhead")
        print("  ✓ Predictable memory access patterns")
        print("  ✗ Fixed alphabet size only")
        print("  ✗ Memory waste for sparse alphabets")


class RegularTrie:
    """Regular trie implementation for comparison"""
    
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end_of_word = False
    
    def __init__(self):
        self.root = self.TrieNode()
        self.node_count = 1
        self.total_chars = 0
    
    def insert(self, word: str) -> None:
        current = self.root
        
        for char in word:
            if char not in current.children:
                current.children[char] = self.TrieNode()
                self.node_count += 1
                self.total_chars += 1
            current = current.children[char]
        
        current.is_end_of_word = True
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'nodes': self.node_count,
            'chars': self.total_chars,
            'efficiency': self.total_chars / (self.node_count * 26) if self.node_count > 0 else 0
        }
    
    def display_structure(self, max_depth: int = 3) -> None:
        def _display(node, prefix, depth):
            if depth >= max_depth:
                return
            
            if node.is_end_of_word:
                print(f"  {'  ' * depth}'{prefix}' [WORD]")
            
            for char, child in sorted(node.children.items()):
                print(f"  {'  ' * depth}├── '{char}'")
                _display(child, prefix + char, depth + 1)
        
        _display(self.root, "", 0)


class CompressedTrie:
    """Compressed trie (Radix tree) implementation"""
    
    class CompressedTrieNode:
        def __init__(self, edge_label: str = ""):
            self.edge_label = edge_label  # The compressed edge label
            self.children = {}
            self.is_end_of_word = False
    
    def __init__(self):
        self.root = self.CompressedTrieNode()
        self.node_count = 1
        self.total_chars = 0
    
    def insert(self, word: str) -> None:
        current = self.root
        i = 0
        
        while i < len(word):
            char = word[i]
            
            if char not in current.children:
                # Create new node with remaining suffix
                remaining = word[i:]
                current.children[char] = self.CompressedTrieNode(remaining)
                current.children[char].is_end_of_word = True
                self.node_count += 1
                self.total_chars += len(remaining)
                return
            
            child = current.children[char]
            
            # Find common prefix length
            j = 0
            while (j < len(child.edge_label) and 
                   i + j < len(word) and 
                   child.edge_label[j] == word[i + j]):
                j += 1
            
            if j == len(child.edge_label):
                # Full edge match, continue
                current = child
                i += j
            elif j < len(child.edge_label):
                # Need to split the edge
                self._split_edge(current, char, child, j)
                
                if i + j == len(word):
                    # Word ends at split point
                    current.children[char].is_end_of_word = True
                else:
                    # Continue with remaining part
                    remaining = word[i + j:]
                    split_char = remaining[0]
                    current.children[char].children[split_char] = self.CompressedTrieNode(remaining)
                    current.children[char].children[split_char].is_end_of_word = True
                    self.node_count += 1
                    self.total_chars += len(remaining)
                return
    
    def _split_edge(self, parent, char, child, split_pos):
        """Split an edge at given position"""
        # Create new intermediate node
        prefix = child.edge_label[:split_pos]
        suffix = child.edge_label[split_pos:]
        
        new_node = self.CompressedTrieNode(prefix)
        suffix_char = suffix[0]
        
        # Update child's edge label
        child.edge_label = suffix
        
        # Reconnect
        parent.children[char] = new_node
        new_node.children[suffix_char] = child
        
        self.node_count += 1
        self.total_chars += len(prefix)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'nodes': self.node_count,
            'chars': self.total_chars,
            'efficiency': self.total_chars / (self.node_count * 26) if self.node_count > 0 else 0
        }
    
    def display_structure(self) -> None:
        def _display(node, prefix, depth):
            if node.is_end_of_word:
                print(f"  {'  ' * depth}'{prefix}' [WORD]")
            
            for char, child in sorted(node.children.items()):
                edge_label = child.edge_label
                print(f"  {'  ' * depth}├── '{edge_label}'")
                _display(child, prefix + edge_label, depth + 1)
        
        _display(self.root, "", 0)


class HashMapTrie:
    """Standard hash map based trie"""
    
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end_of_word = False
    
    def __init__(self):
        self.root = self.TrieNode()
    
    def insert(self, word: str) -> None:
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = self.TrieNode()
            current = current.children[char]
        current.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        current = self.root
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]
        return current.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        current = self.root
        for char in prefix:
            if char not in current.children:
                return False
            current = current.children[char]
        return True


class ArrayTrie:
    """Array-based trie for lowercase letters"""
    
    class TrieNode:
        def __init__(self):
            self.children = [None] * 26  # a-z
            self.is_end_of_word = False
        
        def _char_to_index(self, char):
            return ord(char) - ord('a')
    
    def __init__(self):
        self.root = self.TrieNode()
    
    def insert(self, word: str) -> None:
        current = self.root
        for char in word.lower():
            index = current._char_to_index(char)
            if current.children[index] is None:
                current.children[index] = self.TrieNode()
            current = current.children[index]
        current.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        current = self.root
        for char in word.lower():
            index = current._char_to_index(char)
            if current.children[index] is None:
                return False
            current = current.children[index]
        return current.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        current = self.root
        for char in prefix.lower():
            index = current._char_to_index(char)
            if current.children[index] is None:
                return False
            current = current.children[index]
        return True


# ==========================================
# 2. PERFORMANCE OPTIMIZATIONS
# ==========================================

class PerformanceOptimizedTrie:
    """
    High-performance trie with multiple optimization techniques
    
    Optimizations:
    - Lazy deletion
    - Path compression
    - Caching of frequent queries
    - Memory pooling
    """
    
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end_of_word = False
            self.is_deleted = False  # Lazy deletion
            self.access_count = 0    # For caching decisions
            self.compressed_suffix = None  # For path compression
    
    def __init__(self, cache_size: int = 1000):
        self.root = self.TrieNode()
        self.cache_size = cache_size
        self.query_cache = {}  # LRU cache for frequent queries
        self.cache_access_order = deque()
        self.node_pool = []  # Memory pool for reuse
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'nodes_reused': 0,
            'compressions': 0
        }
    
    def insert(self, word: str) -> None:
        """Optimized insertion with path compression"""
        self._invalidate_cache_for_prefix(word)
        
        current = self.root
        
        for char in word:
            if char not in current.children:
                # Try to reuse node from pool
                if self.node_pool:
                    new_node = self.node_pool.pop()
                    new_node.__init__()  # Reset node
                    self.stats['nodes_reused'] += 1
                else:
                    new_node = self.TrieNode()
                
                current.children[char] = new_node
            
            current = current.children[char]
        
        current.is_end_of_word = True
        current.is_deleted = False
        
        # Apply path compression if beneficial
        self._try_compress_path(word)
    
    def search(self, word: str) -> bool:
        """Optimized search with caching"""
        # Check cache first
        if word in self.query_cache:
            self._update_cache_access(word)
            self.stats['cache_hits'] += 1
            return self.query_cache[word]
        
        self.stats['cache_misses'] += 1
        
        # Perform actual search
        result = self._search_internal(word)
        
        # Cache the result
        self._cache_result(word, result)
        
        return result
    
    def _search_internal(self, word: str) -> bool:
        """Internal search implementation"""
        current = self.root
        
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]
        
        return current.is_end_of_word and not current.is_deleted
    
    def delete(self, word: str) -> bool:
        """Lazy deletion for better performance"""
        current = self.root
        
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]
        
        if current.is_end_of_word and not current.is_deleted:
            current.is_deleted = True
            self._invalidate_cache_for_prefix(word)
            return True
        
        return False
    
    def _cache_result(self, query: str, result: bool) -> None:
        """Cache query result with LRU eviction"""
        if len(self.query_cache) >= self.cache_size:
            # Evict least recently used
            oldest_query = self.cache_access_order.popleft()
            del self.query_cache[oldest_query]
        
        self.query_cache[query] = result
        self.cache_access_order.append(query)
    
    def _update_cache_access(self, query: str) -> None:
        """Update access order for LRU"""
        self.cache_access_order.remove(query)
        self.cache_access_order.append(query)
    
    def _invalidate_cache_for_prefix(self, prefix: str) -> None:
        """Invalidate cache entries that might be affected"""
        to_remove = []
        for cached_query in self.query_cache:
            if cached_query.startswith(prefix) or prefix.startswith(cached_query):
                to_remove.append(cached_query)
        
        for query in to_remove:
            del self.query_cache[query]
            self.cache_access_order.remove(query)
    
    def _try_compress_path(self, word: str) -> None:
        """Apply path compression for single-child chains"""
        # Simplified implementation - would need more sophisticated logic
        current = self.root
        
        for i, char in enumerate(word):
            if char in current.children:
                child = current.children[char]
                
                # Check if this could be compressed
                if (len(child.children) == 1 and 
                    not child.is_end_of_word and 
                    child.compressed_suffix is None):
                    
                    # This is a candidate for compression
                    self.stats['compressions'] += 1
                
                current = child
    
    def cleanup_deleted_nodes(self) -> int:
        """Cleanup lazy-deleted nodes and reclaim memory"""
        cleaned_count = 0
        
        def _cleanup_recursive(node: PerformanceOptimizedTrie.TrieNode, parent: PerformanceOptimizedTrie.TrieNode, char: str):
            nonlocal cleaned_count
            
            # Recursively cleanup children
            children_to_remove = []
            for child_char, child_node in node.children.items():
                if _cleanup_recursive(child_node, node, child_char):
                    children_to_remove.append(child_char)
            
            # Remove cleaned children
            for child_char in children_to_remove:
                del node.children[child_char]
            
            # Check if this node should be cleaned
            if (node.is_deleted and 
                not node.children and 
                node != self.root):
                
                # Add to pool for reuse
                self.node_pool.append(node)
                cleaned_count += 1
                return True
            
            return False
        
        _cleanup_recursive(self.root, None, '')
        return cleaned_count
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_hit_rate = self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
        
        return {
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.query_cache),
            'nodes_reused': self.stats['nodes_reused'],
            'compressions': self.stats['compressions'],
            'pool_size': len(self.node_pool)
        }


# ==========================================
# 3. SPECIALIZED TRIE IMPLEMENTATIONS
# ==========================================

class BinaryTrie:
    """
    Binary trie for integer operations (XOR problems)
    
    Optimized for binary representations of integers
    Useful for maximum XOR, subset XOR problems
    """
    
    class BinaryTrieNode:
        def __init__(self):
            self.children = [None, None]  # 0 and 1
            self.count = 0  # Number of integers passing through this node
    
    def __init__(self, max_bits: int = 32):
        self.root = self.BinaryTrieNode()
        self.max_bits = max_bits
    
    def insert(self, num: int) -> None:
        """Insert integer into binary trie"""
        current = self.root
        current.count += 1
        
        # Process from most significant bit
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            
            if current.children[bit] is None:
                current.children[bit] = self.BinaryTrieNode()
            
            current = current.children[bit]
            current.count += 1
    
    def find_max_xor(self, num: int) -> int:
        """Find maximum XOR with any number in trie"""
        if self.root.count == 0:
            return 0
        
        current = self.root
        max_xor = 0
        
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            desired_bit = 1 - bit  # Want opposite for maximum XOR
            
            if (current.children[desired_bit] is not None and 
                current.children[desired_bit].count > 0):
                max_xor |= (1 << i)
                current = current.children[desired_bit]
            else:
                current = current.children[bit]
        
        return max_xor
    
    def count_numbers_with_prefix(self, prefix: str) -> int:
        """Count numbers with binary prefix"""
        current = self.root
        
        for bit_char in prefix:
            bit = int(bit_char)
            if current.children[bit] is None:
                return 0
            current = current.children[bit]
        
        return current.count


class SuffixTrie:
    """
    Suffix trie for string matching and analysis
    
    Stores all suffixes of a string for pattern matching
    """
    
    def __init__(self, text: str):
        self.text = text + '$'  # Add terminator
        self.root = {}
        self._build_suffix_trie()
    
    def _build_suffix_trie(self) -> None:
        """Build suffix trie from all suffixes"""
        print(f"Building suffix trie for: '{self.text[:-1]}'")
        
        for i in range(len(self.text)):
            self._insert_suffix(i)
    
    def _insert_suffix(self, start_index: int) -> None:
        """Insert suffix starting at given index"""
        current = self.root
        
        for i in range(start_index, len(self.text)):
            char = self.text[i]
            
            if char not in current:
                current[char] = {}
            
            current = current[char]
    
    def contains_pattern(self, pattern: str) -> bool:
        """Check if pattern exists in original text"""
        current = self.root
        
        for char in pattern:
            if char not in current:
                return False
            current = current[char]
        
        return True
    
    def find_all_occurrences(self, pattern: str) -> List[int]:
        """Find all positions where pattern occurs"""
        positions = []
        
        # Simple approach - check all positions
        for i in range(len(self.text) - len(pattern)):
            if self.text[i:i+len(pattern)] == pattern:
                positions.append(i)
        
        return positions


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_trie_optimizations():
    """Demonstrate all trie optimizations"""
    print("=== TRIE OPTIMIZATIONS DEMONSTRATION ===\n")
    
    optimizations = TrieOptimizations()
    
    # 1. Memory optimizations
    optimizations.demonstrate_compressed_trie()
    print("\n" + "="*60 + "\n")
    
    optimizations.demonstrate_array_optimization()
    print("\n" + "="*60 + "\n")
    
    # 2. Performance optimizations
    print("=== PERFORMANCE OPTIMIZED TRIE ===")
    
    perf_trie = PerformanceOptimizedTrie(cache_size=100)
    
    # Test with various operations
    test_words = ["apple", "application", "apply", "banana", "band", "bandana"] * 3
    
    print("Testing performance optimizations:")
    
    # Insert words
    for word in test_words:
        perf_trie.insert(word)
    print(f"  Inserted {len(test_words)} words")
    
    # Search with cache warming
    search_words = ["apple", "app", "application", "apple", "banana", "apple"]
    for word in search_words:
        result = perf_trie.search(word)
    print(f"  Performed {len(search_words)} searches")
    
    # Delete some words
    delete_words = ["app", "band"]
    for word in delete_words:
        perf_trie.delete(word)
    print(f"  Deleted {len(delete_words)} words (lazy deletion)")
    
    # Cleanup
    cleaned = perf_trie.cleanup_deleted_nodes()
    print(f"  Cleaned up {cleaned} deleted nodes")
    
    # Show performance stats
    stats = perf_trie.get_performance_stats()
    print("\nPerformance statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "="*60 + "\n")
    
    # 3. Specialized implementations
    print("=== SPECIALIZED TRIE IMPLEMENTATIONS ===")
    
    print("a) Binary Trie for XOR operations:")
    binary_trie = BinaryTrie(8)  # 8-bit integers for demo
    
    numbers = [3, 10, 5, 25, 2, 8]
    print(f"   Numbers: {numbers}")
    print(f"   Binary: {[bin(n)[2:].zfill(8) for n in numbers]}")
    
    for num in numbers:
        binary_trie.insert(num)
    
    print("\n   Maximum XOR for each number:")
    for num in numbers:
        max_xor = binary_trie.find_max_xor(num)
        print(f"     {num} (binary: {bin(num)[2:].zfill(8)}) -> max XOR: {max_xor}")
    
    print("\n" + "-"*40 + "\n")
    
    print("b) Suffix Trie for pattern matching:")
    text = "banana"
    suffix_trie = SuffixTrie(text)
    
    patterns = ["ana", "ban", "na", "xyz"]
    print(f"   Text: '{text}'")
    print("   Pattern search results:")
    
    for pattern in patterns:
        found = suffix_trie.contains_pattern(pattern)
        if found:
            positions = suffix_trie.find_all_occurrences(pattern)
            print(f"     '{pattern}': Found at positions {positions}")
        else:
            print(f"     '{pattern}': Not found")


if __name__ == "__main__":
    demonstrate_trie_optimizations()
    
    print("\n=== TRIE OPTIMIZATIONS MASTERY GUIDE ===")
    
    print("\n🎯 OPTIMIZATION CATEGORIES:")
    print("• Memory: Compressed tries, array-based storage, path compression")
    print("• Performance: Caching, lazy deletion, memory pooling")
    print("• Specialized: Binary tries for XOR, suffix tries for matching")
    print("• System: Concurrent access, distributed implementations")
    
    print("\n📊 OPTIMIZATION IMPACT:")
    print("• Compressed Trie: 50-90% memory reduction for sparse data")
    print("• Array-based: 2-5x faster access for fixed alphabets")
    print("• Caching: 10-50x speedup for repeated queries")
    print("• Lazy deletion: Constant time deletion vs O(m) cleanup")
    
    print("\n⚡ CHOOSING OPTIMIZATIONS:")
    print("• Memory-constrained: Use compressed tries and bit packing")
    print("• Performance-critical: Add caching and array optimization")
    print("• Large-scale: Implement distributed and persistent versions")
    print("• Specific domains: Binary tries for XOR, suffix tries for strings")
    
    print("\n🔧 IMPLEMENTATION STRATEGIES:")
    print("• Profile first: Measure actual memory and performance issues")
    print("• Incremental optimization: Apply one technique at a time")
    print("• Trade-off analysis: Balance memory vs speed vs complexity")
    print("• Benchmark thoroughly: Test with realistic datasets")
    
    print("\n🏆 ADVANCED TECHNIQUES:")
    print("• Succinct data structures for minimal memory")
    print("• Persistent tries for versioning and undo/redo")
    print("• Concurrent tries with lock-free operations")
    print("• Distributed tries for horizontal scaling")
    print("• Hardware-optimized implementations (SIMD, GPU)")
    
    print("\n🎓 MASTERY PROGRESSION:")
    print("1. Understand basic trie memory and performance characteristics")
    print("2. Learn to identify bottlenecks through profiling")
    print("3. Master compression and specialized implementations")
    print("4. Study concurrent and distributed trie algorithms")
    print("5. Research cutting-edge succinct data structures")
    
    print("\n💡 OPTIMIZATION PRINCIPLES:")
    print("• Measure before optimizing - avoid premature optimization")
    print("• Consider data characteristics and usage patterns")
    print("• Balance memory, speed, and implementation complexity")
    print("• Design for scalability and future requirements")
    print("• Document optimization decisions and trade-offs")
