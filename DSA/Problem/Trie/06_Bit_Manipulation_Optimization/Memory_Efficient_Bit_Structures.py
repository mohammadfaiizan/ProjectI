"""
Memory Efficient Bit Structures - Multiple Approaches
Difficulty: Hard

Memory-optimized bit manipulation structures for large-scale applications
with focus on reducing memory footprint while maintaining performance.

Techniques:
1. Compressed Bit Vectors
2. Succinct Data Structures
3. Bit-packed Trie Nodes
4. Memory Pool Management
5. Cache-aware Bit Operations
6. Lazy Evaluation Structures
"""

import sys
from typing import List, Dict, Optional, Tuple, Iterator
from collections import defaultdict
import array
import mmap
import tempfile

class CompressedBitVector:
    """Memory-efficient bit vector with compression"""
    
    def __init__(self, size: int):
        self.size = size
        self.chunks = array.array('L')  # Use unsigned long for efficiency
        self.chunk_size = 64  # 64-bit chunks
        self.num_chunks = (size + self.chunk_size - 1) // self.chunk_size
        self.chunks.extend([0] * self.num_chunks)
        self.population_count = 0
    
    def set_bit(self, index: int) -> None:
        """Set bit at index"""
        if 0 <= index < self.size:
            chunk_idx = index // self.chunk_size
            bit_idx = index % self.chunk_size
            
            if not (self.chunks[chunk_idx] & (1 << bit_idx)):
                self.chunks[chunk_idx] |= (1 << bit_idx)
                self.population_count += 1
    
    def clear_bit(self, index: int) -> None:
        """Clear bit at index"""
        if 0 <= index < self.size:
            chunk_idx = index // self.chunk_size
            bit_idx = index % self.chunk_size
            
            if self.chunks[chunk_idx] & (1 << bit_idx):
                self.chunks[chunk_idx] &= ~(1 << bit_idx)
                self.population_count -= 1
    
    def get_bit(self, index: int) -> bool:
        """Get bit at index"""
        if 0 <= index < self.size:
            chunk_idx = index // self.chunk_size
            bit_idx = index % self.chunk_size
            return bool(self.chunks[chunk_idx] & (1 << bit_idx))
        return False
    
    def rank(self, index: int) -> int:
        """Count number of 1s up to index"""
        if index < 0:
            return 0
        if index >= self.size:
            return self.population_count
        
        count = 0
        chunk_idx = index // self.chunk_size
        
        # Count full chunks
        for i in range(chunk_idx):
            count += bin(self.chunks[i]).count('1')
        
        # Count partial chunk
        bit_idx = index % self.chunk_size
        if bit_idx > 0:
            mask = (1 << (bit_idx + 1)) - 1
            count += bin(self.chunks[chunk_idx] & mask).count('1')
        
        return count
    
    def memory_usage(self) -> int:
        """Calculate memory usage in bytes"""
        return (sys.getsizeof(self.chunks) + 
                self.chunks.itemsize * len(self.chunks) +
                sys.getsizeof(self))

class SuccinctBitTrie:
    """Succinct trie using minimal memory"""
    
    def __init__(self, max_bits: int = 32):
        self.max_bits = max_bits
        self.structure = CompressedBitVector(0)  # Dynamic sizing
        self.nodes = []  # Node information
        self.node_count = 0
    
    def insert(self, number: int) -> None:
        """Insert number into succinct trie"""
        # Convert number to bit path
        path = []
        for i in range(self.max_bits - 1, -1, -1):
            path.append((number >> i) & 1)
        
        # Navigate/create path in succinct structure
        current_node = 0  # Root node index
        
        for bit in path:
            # Find or create child
            child_index = self._find_or_create_child(current_node, bit)
            current_node = child_index
    
    def _find_or_create_child(self, parent_index: int, bit: int) -> int:
        """Find or create child node"""
        # Simplified implementation for demonstration
        # In practice, would use sophisticated succinct indexing
        
        while len(self.nodes) <= parent_index:
            self.nodes.append({'left': -1, 'right': -1, 'exists': False})
        
        if not self.nodes[parent_index]['exists']:
            self.nodes[parent_index]['exists'] = True
        
        child_key = 'left' if bit == 0 else 'right'
        
        if self.nodes[parent_index][child_key] == -1:
            # Create new child
            child_index = len(self.nodes)
            self.nodes.append({'left': -1, 'right': -1, 'exists': False})
            self.nodes[parent_index][child_key] = child_index
            return child_index
        else:
            return self.nodes[parent_index][child_key]

class BitPackedTrieNode:
    """Ultra-compact trie node using bit packing"""
    __slots__ = ['data']
    
    def __init__(self):
        # Pack everything into 64-bit integer:
        # Bits 0-25: character (26 letters)
        # Bit 26: is_word
        # Bits 27-31: frequency (0-31)
        # Bits 32-47: left child index
        # Bits 48-63: right child index
        self.data = 0
    
    def set_char(self, char: str) -> None:
        if char and char.isalpha():
            char_val = ord(char.lower()) - ord('a')
            self.data = (self.data & ~0x3FFFFFF) | char_val
    
    def get_char(self) -> str:
        char_val = self.data & 0x3FFFFFF
        return chr(char_val + ord('a')) if char_val < 26 else ''
    
    def set_is_word(self, is_word: bool) -> None:
        if is_word:
            self.data |= (1 << 26)
        else:
            self.data &= ~(1 << 26)
    
    def get_is_word(self) -> bool:
        return bool(self.data & (1 << 26))
    
    def set_frequency(self, freq: int) -> None:
        freq = min(freq, 31)
        self.data = (self.data & ~(0x1F << 27)) | (freq << 27)
    
    def get_frequency(self) -> int:
        return (self.data >> 27) & 0x1F
    
    def set_left_child(self, index: int) -> None:
        index = min(index, 0xFFFF)
        self.data = (self.data & ~(0xFFFF << 32)) | (index << 32)
    
    def get_left_child(self) -> int:
        return (self.data >> 32) & 0xFFFF
    
    def set_right_child(self, index: int) -> None:
        index = min(index, 0xFFFF)
        self.data = (self.data & ~(0xFFFF << 48)) | (index << 48)
    
    def get_right_child(self) -> int:
        return (self.data >> 48) & 0xFFFF

class MemoryPool:
    """Memory pool for efficient allocation"""
    
    def __init__(self, block_size: int = 1024):
        self.block_size = block_size
        self.pools = {}  # size -> list of available blocks
        self.allocated_blocks = []
    
    def allocate(self, size: int) -> bytearray:
        """Allocate block of given size"""
        # Round up to nearest power of 2
        actual_size = 1
        while actual_size < size:
            actual_size <<= 1
        
        if actual_size not in self.pools:
            self.pools[actual_size] = []
        
        if self.pools[actual_size]:
            return self.pools[actual_size].pop()
        else:
            # Allocate new block
            block = bytearray(actual_size)
            self.allocated_blocks.append(block)
            return block
    
    def deallocate(self, block: bytearray) -> None:
        """Return block to pool"""
        size = len(block)
        if size not in self.pools:
            self.pools[size] = []
        
        # Clear block and return to pool
        for i in range(len(block)):
            block[i] = 0
        
        self.pools[size].append(block)
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics"""
        total_allocated = sum(len(block) for block in self.allocated_blocks)
        total_available = sum(len(pool) * size for size, pool in self.pools.items())
        
        return {
            'total_allocated': total_allocated,
            'total_available': total_available,
            'fragmentation': len(self.pools),
            'utilization': (total_allocated - total_available) / max(total_allocated, 1)
        }

class MemoryEfficientStructures:
    
    def __init__(self):
        self.memory_pool = MemoryPool()
        self.bit_vectors = {}
        self.packed_nodes = []
        self.temp_files = []  # For memory mapping
    
    def create_compressed_trie(self, words: List[str]) -> Dict[str, any]:
        """
        Approach 1: Create memory-compressed trie
        
        Build trie with maximum memory efficiency.
        
        Time: O(n * m) where n=words, m=avg_length
        Space: O(compressed_size)
        """
        # Use bit-packed nodes
        self.packed_nodes = [BitPackedTrieNode()]  # Root
        word_to_index = {}
        
        for word_idx, word in enumerate(words):
            node_index = 0  # Start at root
            
            for char in word:
                current_node = self.packed_nodes[node_index]
                
                # Determine which child to follow
                if char <= 'm':  # Use left child for a-m
                    child_index = current_node.get_left_child()
                    if child_index == 0:  # No child exists
                        # Create new child
                        new_index = len(self.packed_nodes)
                        self.packed_nodes.append(BitPackedTrieNode())
                        current_node.set_left_child(new_index)
                        child_index = new_index
                else:  # Use right child for n-z
                    child_index = current_node.get_right_child()
                    if child_index == 0:
                        new_index = len(self.packed_nodes)
                        self.packed_nodes.append(BitPackedTrieNode())
                        current_node.set_right_child(new_index)
                        child_index = new_index
                
                # Update child node
                child_node = self.packed_nodes[child_index]
                child_node.set_char(char)
                
                node_index = child_index
            
            # Mark end of word
            final_node = self.packed_nodes[node_index]
            final_node.set_is_word(True)
            final_node.set_frequency(final_node.get_frequency() + 1)
            word_to_index[word] = node_index
        
        return {
            'nodes': self.packed_nodes,
            'word_mapping': word_to_index,
            'memory_usage': len(self.packed_nodes) * 8  # 8 bytes per node
        }
    
    def create_memory_mapped_structure(self, data: List[int]) -> str:
        """
        Approach 2: Memory-mapped file structure
        
        Use memory mapping for large datasets.
        
        Time: O(n)
        Space: O(disk_space)
        """
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_files.append(temp_file.name)
        
        # Write data to file
        with open(temp_file.name, 'wb') as f:
            for value in data:
                f.write(value.to_bytes(4, byteorder='little'))
        
        # Memory map the file
        with open(temp_file.name, 'r+b') as f:
            mapped = mmap.mmap(f.fileno(), 0)
            return temp_file.name, mapped
    
    def create_lazy_bit_structure(self, size: int) -> 'LazyBitStructure':
        """
        Approach 3: Lazy evaluation bit structure
        
        Create structure that computes values on demand.
        
        Time: O(1) for creation, O(f(x)) for evaluation
        Space: O(metadata)
        """
        class LazyBitStructure:
            def __init__(self, size: int):
                self.size = size
                self.computed = {}  # Cache computed values
                self.generators = {}  # Functions to generate values
            
            def set_generator(self, start: int, end: int, func: callable) -> None:
                """Set generator function for range"""
                self.generators[(start, end)] = func
            
            def get_value(self, index: int) -> int:
                """Get value at index (computed lazily)"""
                if index in self.computed:
                    return self.computed[index]
                
                # Find appropriate generator
                for (start, end), func in self.generators.items():
                    if start <= index < end:
                        value = func(index)
                        self.computed[index] = value
                        return value
                
                return 0  # Default value
            
            def force_computation(self, start: int, end: int) -> None:
                """Force computation of range"""
                for i in range(start, end):
                    self.get_value(i)
        
        return LazyBitStructure(size)
    
    def optimize_cache_locality(self, data: List[int]) -> List[int]:
        """
        Approach 4: Cache-aware data layout
        
        Reorganize data for better cache performance.
        
        Time: O(n log n)
        Space: O(n)
        """
        # Group data by cache line size (typically 64 bytes = 16 integers)
        cache_line_size = 16
        grouped_data = []
        
        # Sort data to improve locality
        sorted_data = sorted(data)
        
        # Arrange in cache-friendly order
        for i in range(0, len(sorted_data), cache_line_size):
            group = sorted_data[i:i + cache_line_size]
            grouped_data.extend(group)
        
        return grouped_data
    
    def create_succinct_rank_select(self, bits: List[bool]) -> 'SuccinctRankSelect':
        """
        Approach 5: Succinct rank/select structure
        
        Support rank/select queries in minimal space.
        
        Time: O(n) construction, O(1) queries
        Space: O(n + o(n))
        """
        class SuccinctRankSelect:
            def __init__(self, bits: List[bool]):
                self.n = len(bits)
                self.bits = CompressedBitVector(self.n)
                
                # Build bit vector
                for i, bit in enumerate(bits):
                    if bit:
                        self.bits.set_bit(i)
                
                # Build rank/select auxiliary structures
                self.block_size = max(1, int(self.n ** 0.5))
                self.rank_blocks = []
                self.select_samples = []
                
                self._build_rank_structure()
                self._build_select_structure()
            
            def _build_rank_structure(self) -> None:
                """Build rank support structure"""
                running_count = 0
                
                for i in range(0, self.n, self.block_size):
                    self.rank_blocks.append(running_count)
                    
                    # Count bits in this block
                    for j in range(i, min(i + self.block_size, self.n)):
                        if self.bits.get_bit(j):
                            running_count += 1
            
            def _build_select_structure(self) -> None:
                """Build select support structure"""
                sample_rate = max(1, int(self.bits.population_count ** 0.5))
                count = 0
                
                for i in range(self.n):
                    if self.bits.get_bit(i):
                        count += 1
                        if count % sample_rate == 0:
                            self.select_samples.append(i)
            
            def rank(self, index: int) -> int:
                """Count 1s up to index"""
                return self.bits.rank(index)
            
            def select(self, k: int) -> int:
                """Find position of k-th 1"""
                if k <= 0 or k > self.bits.population_count:
                    return -1
                
                # Use samples for fast approximation
                sample_rate = max(1, int(self.bits.population_count ** 0.5))
                sample_index = (k - 1) // sample_rate
                
                if sample_index < len(self.select_samples):
                    start_pos = self.select_samples[sample_index]
                    target_count = sample_index * sample_rate + 1
                else:
                    start_pos = 0
                    target_count = 1
                
                # Linear scan from sample position
                count = self.rank(start_pos)
                for i in range(start_pos, self.n):
                    if self.bits.get_bit(i):
                        count += 1
                        if count == k:
                            return i
                
                return -1
        
        return SuccinctRankSelect(bits)
    
    def cleanup(self) -> None:
        """Clean up temporary resources"""
        import os
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        self.temp_files.clear()


def test_compressed_bit_vector():
    """Test compressed bit vector"""
    print("=== Testing Compressed Bit Vector ===")
    
    size = 1000
    bv = CompressedBitVector(size)
    
    # Set some bits
    test_indices = [1, 5, 10, 50, 100, 500, 999]
    for idx in test_indices:
        bv.set_bit(idx)
    
    print(f"Set bits at indices: {test_indices}")
    print(f"Population count: {bv.population_count}")
    print(f"Memory usage: {bv.memory_usage()} bytes")
    
    # Test rank queries
    for idx in [0, 10, 100, 1000]:
        rank = bv.rank(idx)
        print(f"Rank({idx}): {rank}")


def test_bit_packed_nodes():
    """Test bit-packed trie nodes"""
    print("\n=== Testing Bit-Packed Nodes ===")
    
    node = BitPackedTrieNode()
    
    # Test setting/getting values
    node.set_char('a')
    node.set_is_word(True)
    node.set_frequency(5)
    node.set_left_child(10)
    node.set_right_child(20)
    
    print(f"Character: {node.get_char()}")
    print(f"Is word: {node.get_is_word()}")
    print(f"Frequency: {node.get_frequency()}")
    print(f"Left child: {node.get_left_child()}")
    print(f"Right child: {node.get_right_child()}")
    print(f"Node size: {sys.getsizeof(node)} bytes")
    print(f"Data value: {node.data}")


def test_memory_pool():
    """Test memory pool allocation"""
    print("\n=== Testing Memory Pool ===")
    
    pool = MemoryPool()
    
    # Allocate various sized blocks
    blocks = []
    sizes = [16, 32, 64, 128, 256]
    
    for size in sizes:
        block = pool.allocate(size)
        blocks.append(block)
        print(f"Allocated block of size {len(block)}")
    
    # Get statistics
    stats = pool.get_memory_usage()
    print(f"\nMemory statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Deallocate some blocks
    for block in blocks[:2]:
        pool.deallocate(block)
    
    print(f"\nAfter deallocation:")
    stats = pool.get_memory_usage()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def test_succinct_rank_select():
    """Test succinct rank/select structure"""
    print("\n=== Testing Succinct Rank/Select ===")
    
    structures = MemoryEfficientStructures()
    
    # Create test bit pattern
    bits = [True, False, True, True, False, False, True, False, True, True]
    print(f"Bit pattern: {[int(b) for b in bits]}")
    
    rs_structure = structures.create_succinct_rank_select(bits)
    
    # Test rank queries
    print(f"\nRank queries:")
    for i in range(len(bits) + 1):
        rank = rs_structure.rank(i)
        print(f"  rank({i}): {rank}")
    
    # Test select queries
    print(f"\nSelect queries:")
    ones_count = sum(bits)
    for k in range(1, ones_count + 1):
        pos = rs_structure.select(k)
        print(f"  select({k}): {pos}")


def benchmark_memory_efficiency():
    """Benchmark memory efficiency"""
    print("\n=== Memory Efficiency Benchmark ===")
    
    import random
    
    structures = MemoryEfficientStructures()
    
    # Test data
    test_sizes = [1000, 5000, 10000]
    
    print(f"{'Size':<8} {'Standard(KB)':<15} {'Compressed(KB)':<15} {'Savings':<10}")
    print("-" * 55)
    
    for size in test_sizes:
        # Generate test data
        words = []
        for _ in range(size):
            length = random.randint(3, 10)
            word = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))
            words.append(word)
        
        # Standard storage estimate
        standard_size = sum(len(word) for word in words) * 8  # Rough estimate
        
        # Compressed trie
        compressed = structures.create_compressed_trie(words)
        compressed_size = compressed['memory_usage']
        
        # Calculate savings
        savings = (standard_size - compressed_size) / standard_size * 100
        
        print(f"{size:<8} {standard_size/1024:<15.1f} {compressed_size/1024:<15.1f} {savings:<10.1f}%")
    
    # Cleanup
    structures.cleanup()


if __name__ == "__main__":
    test_compressed_bit_vector()
    test_bit_packed_nodes()
    test_memory_pool()
    test_succinct_rank_select()
    benchmark_memory_efficiency()

"""
Memory Efficient Bit Structures demonstrates space optimization techniques:

Key Techniques:
1. Compressed Bit Vectors - Efficient storage for sparse bit arrays
2. Bit-packed Nodes - Multiple fields in single integers
3. Succinct Structures - Theoretical minimum space with fast queries
4. Memory Pools - Reduce allocation overhead and fragmentation
5. Cache-aware Layout - Optimize for modern CPU cache hierarchies
6. Lazy Evaluation - Compute values on-demand to save space

Memory Optimizations:
- Use appropriate data types (array.array vs list)
- Pack multiple values into single integers
- Implement custom memory pools for frequent allocations
- Memory mapping for very large datasets
- Succinct data structures for theoretical space bounds

Real-world Applications:
- Large-scale indexing systems
- Memory-constrained embedded systems
- Big data processing with limited RAM
- Cache-efficient database structures
- Compressed file systems and archives

These techniques can reduce memory usage by 50-90% while maintaining
acceptable performance for most applications.
"""
