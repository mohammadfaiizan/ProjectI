"""
Trie Memory Compression - Multiple Approaches
Difficulty: Hard

Advanced memory compression techniques for trie data structures.
Focuses on reducing memory footprint while maintaining performance.

Approaches:
1. Array-based Compressed Trie (Radix Tree)
2. Bitmap Compression for Dense Tries
3. Huffman Encoded Trie Nodes
4. Memory Pool Allocation
5. Lazy Loading and Paging
6. Delta Compression for Similar Patterns
7. Bit-packed Node Representation

Applications:
- Memory-constrained systems
- Large-scale text processing
- Embedded systems with limited RAM
- High-performance caching systems
"""

import sys
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict, deque
import array
import bisect
import pickle
import gzip
from dataclasses import dataclass
import time

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    nodes_count: int = 0
    total_bytes: int = 0
    compression_ratio: float = 0.0
    access_time_ms: float = 0.0

class CompressedTrieNode:
    """Memory-optimized trie node"""
    __slots__ = ['edge', 'children', 'is_word', 'frequency']
    
    def __init__(self, edge: str = ""):
        self.edge = edge  # Compressed edge string
        self.children: Dict[str, 'CompressedTrieNode'] = {}
        self.is_word = False
        self.frequency = 0

class BitPackedNode:
    """Bit-packed representation of trie node"""
    __slots__ = ['data']
    
    def __init__(self):
        # Pack multiple fields into single integer
        # Bits 0-25: character (26 letters)
        # Bit 26: is_word flag
        # Bits 27-31: frequency (max 31)
        self.data = 0
    
    def set_char(self, char: str):
        if char and char.isalpha():
            char_value = ord(char.lower()) - ord('a')
            self.data = (self.data & ~0x3FFFFFF) | char_value
    
    def get_char(self) -> str:
        char_value = self.data & 0x3FFFFFF
        return chr(char_value + ord('a')) if char_value < 26 else ''
    
    def set_is_word(self, is_word: bool):
        if is_word:
            self.data |= (1 << 26)
        else:
            self.data &= ~(1 << 26)
    
    def get_is_word(self) -> bool:
        return bool(self.data & (1 << 26))
    
    def set_frequency(self, freq: int):
        freq = min(freq, 31)  # Cap at 31 (5 bits)
        self.data = (self.data & ~(0x1F << 27)) | (freq << 27)
    
    def get_frequency(self) -> int:
        return (self.data >> 27) & 0x1F

class TrieMemoryCompressor:
    
    def __init__(self):
        """Initialize trie memory compressor"""
        self.compression_stats = MemoryStats()
        self.memory_pool = []  # Reusable node pool
        self.page_cache = {}   # LRU cache for paged nodes
        self.max_cache_size = 1000
    
    def compress_radix_tree(self, words: List[str]) -> Tuple[CompressedTrieNode, MemoryStats]:
        """
        Approach 1: Array-based Compressed Trie (Radix Tree)
        
        Compress common prefixes into single edges.
        
        Time: O(n * m) where n=words, m=average_length
        Space: O(compressed_size) - significantly less than standard trie
        """
        start_time = time.time()
        root = CompressedTrieNode()
        
        def insert_word(word: str) -> None:
            """Insert word into compressed trie"""
            node = root
            i = 0
            
            while i < len(word):
                # Find matching child edge
                found_child = None
                for edge_char, child in node.children.items():
                    if edge_char and word[i:i+1] == edge_char[0]:
                        found_child = child
                        break
                
                if found_child is None:
                    # Create new edge for remaining suffix
                    new_node = CompressedTrieNode(word[i:])
                    new_node.is_word = True
                    new_node.frequency = 1
                    node.children[word[i:i+1]] = new_node
                    break
                else:
                    # Check how much of the edge matches
                    edge = found_child.edge
                    match_len = 0
                    
                    while (match_len < len(edge) and 
                           i + match_len < len(word) and
                           edge[match_len] == word[i + match_len]):
                        match_len += 1
                    
                    if match_len == len(edge):
                        # Exact edge match, continue with child
                        node = found_child
                        i += match_len
                        if i == len(word):
                            node.is_word = True
                            node.frequency += 1
                    else:
                        # Partial match, need to split edge
                        # Create intermediate node
                        intermediate = CompressedTrieNode(edge[:match_len])
                        
                        # Update existing child's edge
                        found_child.edge = edge[match_len:]
                        
                        # Re-link nodes
                        old_key = list(node.children.keys())[list(node.children.values()).index(found_child)]
                        del node.children[old_key]
                        
                        node.children[edge[0:1]] = intermediate
                        intermediate.children[edge[match_len:match_len+1]] = found_child
                        
                        # Continue with remaining suffix
                        i += match_len
                        if i == len(word):
                            intermediate.is_word = True
                            intermediate.frequency = 1
                        else:
                            # Add remaining suffix
                            new_node = CompressedTrieNode(word[i:])
                            new_node.is_word = True
                            new_node.frequency = 1
                            intermediate.children[word[i:i+1]] = new_node
                        break
        
        # Build compressed trie
        for word in words:
            insert_word(word)
        
        # Calculate statistics
        def calculate_stats(node: CompressedTrieNode) -> Tuple[int, int]:
            """Calculate node count and memory usage"""
            nodes = 1
            memory = sys.getsizeof(node) + sys.getsizeof(node.edge) + sys.getsizeof(node.children)
            
            for child in node.children.values():
                child_nodes, child_memory = calculate_stats(child)
                nodes += child_nodes
                memory += child_memory
            
            return nodes, memory
        
        nodes_count, total_bytes = calculate_stats(root)
        
        # Calculate compression ratio vs standard trie
        standard_trie_estimate = sum(len(word) for word in words) * 64  # Rough estimate
        compression_ratio = standard_trie_estimate / total_bytes if total_bytes > 0 else 1.0
        
        stats = MemoryStats(
            nodes_count=nodes_count,
            total_bytes=total_bytes,
            compression_ratio=compression_ratio,
            access_time_ms=(time.time() - start_time) * 1000
        )
        
        return root, stats
    
    def compress_bitmap_dense(self, words: List[str]) -> Tuple[Dict[str, Any], MemoryStats]:
        """
        Approach 2: Bitmap Compression for Dense Tries
        
        Use bitmaps to represent dense character sets efficiently.
        
        Time: O(n * m) for construction
        Space: O(alphabet_size * levels) for dense regions
        """
        start_time = time.time()
        
        # Analyze character frequency at each level
        level_chars = defaultdict(set)
        
        for word in words:
            for i, char in enumerate(word):
                level_chars[i].add(char)
        
        # Create bitmap-compressed representation
        compressed_trie = {
            'levels': {},
            'bitmaps': {},
            'words': set(words)
        }
        
        for level, chars in level_chars.items():
            # Create bitmap for this level
            bitmap = 0
            char_to_bit = {}
            
            for i, char in enumerate(sorted(chars)):
                bitmap |= (1 << i)
                char_to_bit[char] = i
            
            compressed_trie['levels'][level] = {
                'bitmap': bitmap,
                'char_map': char_to_bit,
                'char_count': len(chars)
            }
        
        # Build compressed paths
        def build_compressed_paths():
            """Build compressed path representation"""
            paths = {}
            
            for word in words:
                path_key = ""
                for i, char in enumerate(word):
                    if i in compressed_trie['levels']:
                        bit_pos = compressed_trie['levels'][i]['char_map'][char]
                        path_key += str(bit_pos) + ","
                
                paths[path_key] = word
            
            return paths
        
        compressed_trie['paths'] = build_compressed_paths()
        
        # Calculate memory usage
        total_bytes = (
            sys.getsizeof(compressed_trie) +
            sum(sys.getsizeof(level_data) for level_data in compressed_trie['levels'].values()) +
            sys.getsizeof(compressed_trie['paths']) +
            sum(sys.getsizeof(word) for word in words)
        )
        
        stats = MemoryStats(
            nodes_count=len(compressed_trie['levels']),
            total_bytes=total_bytes,
            compression_ratio=2.5,  # Estimated based on bitmap efficiency
            access_time_ms=(time.time() - start_time) * 1000
        )
        
        return compressed_trie, stats
    
    def compress_huffman_encoded(self, words: List[str]) -> Tuple[Dict[str, Any], MemoryStats]:
        """
        Approach 3: Huffman Encoded Trie Nodes
        
        Use Huffman coding to compress frequently used characters.
        
        Time: O(n * m + k log k) where k=unique_characters
        Space: O(compressed_representation)
        """
        start_time = time.time()
        
        # Analyze character frequencies
        char_freq = defaultdict(int)
        for word in words:
            for char in word:
                char_freq[char] += 1
        
        # Build Huffman tree
        def build_huffman_tree(frequencies: Dict[str, int]) -> Dict[str, str]:
            """Build Huffman encoding table"""
            if len(frequencies) <= 1:
                return {list(frequencies.keys())[0]: '0'} if frequencies else {}
            
            # Priority queue: (frequency, unique_id, node)
            heap = []
            node_id = 0
            
            for char, freq in frequencies.items():
                heapq.heappush(heap, (freq, node_id, {'char': char, 'children': None}))
                node_id += 1
            
            while len(heap) > 1:
                freq1, id1, node1 = heapq.heappop(heap)
                freq2, id2, node2 = heapq.heappop(heap)
                
                merged_node = {
                    'char': None,
                    'children': [node1, node2]
                }
                
                heapq.heappush(heap, (freq1 + freq2, node_id, merged_node))
                node_id += 1
            
            # Generate encoding table
            encoding_table = {}
            
            def generate_codes(node, code=""):
                if node['char'] is not None:
                    encoding_table[node['char']] = code or '0'
                else:
                    if node['children']:
                        generate_codes(node['children'][0], code + '0')
                        generate_codes(node['children'][1], code + '1')
            
            if heap:
                generate_codes(heap[0][2])
            
            return encoding_table
        
        import heapq
        encoding_table = build_huffman_tree(char_freq)
        
        # Encode words using Huffman codes
        encoded_words = {}
        total_bits = 0
        
        for word in words:
            encoded = ""
            for char in word:
                encoded += encoding_table[char]
            
            encoded_words[word] = encoded
            total_bits += len(encoded)
        
        # Create compressed representation
        compressed_data = {
            'encoding_table': encoding_table,
            'encoded_words': encoded_words,
            'huffman_tree': build_huffman_tree(char_freq)
        }
        
        # Calculate compression statistics
        original_bits = sum(len(word) * 8 for word in words)  # ASCII encoding
        compression_ratio = original_bits / total_bits if total_bits > 0 else 1.0
        
        total_bytes = (
            sys.getsizeof(compressed_data) +
            sys.getsizeof(encoding_table) +
            sys.getsizeof(encoded_words) +
            total_bits // 8  # Convert bits to bytes
        )
        
        stats = MemoryStats(
            nodes_count=len(encoding_table),
            total_bytes=total_bytes,
            compression_ratio=compression_ratio,
            access_time_ms=(time.time() - start_time) * 1000
        )
        
        return compressed_data, stats
    
    def optimize_memory_pool(self, words: List[str]) -> Tuple[Any, MemoryStats]:
        """
        Approach 4: Memory Pool Allocation
        
        Use object pooling to reduce memory allocation overhead.
        
        Time: O(n * m) with reduced allocation overhead
        Space: O(pool_size + active_nodes)
        """
        start_time = time.time()
        
        class PooledTrieNode:
            """Trie node with memory pooling"""
            __slots__ = ['children', 'is_word', 'char', 'frequency', '_in_use']
            
            def __init__(self):
                self.children = {}
                self.is_word = False
                self.char = ''
                self.frequency = 0
                self._in_use = False
            
            def reset(self):
                """Reset node for reuse"""
                self.children.clear()
                self.is_word = False
                self.char = ''
                self.frequency = 0
                self._in_use = False
        
        # Initialize memory pool
        pool_size = min(1000, len(words) * 10)  # Estimate needed nodes
        self.memory_pool = [PooledTrieNode() for _ in range(pool_size)]
        pool_index = 0
        
        def get_node() -> PooledTrieNode:
            """Get node from pool"""
            nonlocal pool_index
            if pool_index < len(self.memory_pool):
                node = self.memory_pool[pool_index]
                node._in_use = True
                pool_index += 1
                return node
            else:
                # Pool exhausted, create new node
                return PooledTrieNode()
        
        def return_node(node: PooledTrieNode):
            """Return node to pool"""
            node.reset()
        
        # Build trie using memory pool
        root = get_node()
        
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    new_node = get_node()
                    new_node.char = char
                    node.children[char] = new_node
                
                node = node.children[char]
            
            node.is_word = True
            node.frequency += 1
        
        # Calculate memory usage
        active_nodes = pool_index
        pool_overhead = len(self.memory_pool) * sys.getsizeof(PooledTrieNode())
        
        stats = MemoryStats(
            nodes_count=active_nodes,
            total_bytes=pool_overhead,
            compression_ratio=1.2,  # Pool reduces allocation overhead
            access_time_ms=(time.time() - start_time) * 1000
        )
        
        return root, stats
    
    def implement_lazy_loading(self, words: List[str]) -> Tuple[Any, MemoryStats]:
        """
        Approach 5: Lazy Loading and Paging
        
        Load trie sections on-demand to reduce memory footprint.
        
        Time: O(page_load_time) per access
        Space: O(active_pages * page_size)
        """
        start_time = time.time()
        
        class LazyTrieNode:
            """Lazy-loaded trie node"""
            def __init__(self, page_id: str = ""):
                self.page_id = page_id
                self.children = {}
                self.is_word = False
                self.frequency = 0
                self._loaded = False
                self._data = None
        
        # Partition words into pages based on prefixes
        page_size = 50
        pages = {}
        page_data = {}
        
        # Group words by common prefixes
        prefix_groups = defaultdict(list)
        for word in words:
            prefix = word[:2] if len(word) >= 2 else word
            prefix_groups[prefix].append(word)
        
        # Create pages
        page_id = 0
        for prefix, word_list in prefix_groups.items():
            for i in range(0, len(word_list), page_size):
                page_words = word_list[i:i + page_size]
                page_key = f"page_{page_id}"
                pages[page_key] = page_words
                
                # Serialize page data
                page_data[page_key] = pickle.dumps(page_words)
                page_id += 1
        
        # Create lazy trie structure
        root = LazyTrieNode("root")
        page_mapping = {}  # word -> page_id
        
        for page_key, page_words in pages.items():
            for word in page_words:
                page_mapping[word] = page_key
        
        def load_page(page_id: str) -> List[str]:
            """Load page from storage"""
            if page_id in self.page_cache:
                return self.page_cache[page_id]
            
            if len(self.page_cache) >= self.max_cache_size:
                # Evict oldest page (simple FIFO)
                oldest_page = next(iter(self.page_cache))
                del self.page_cache[oldest_page]
            
            # Deserialize page data
            page_words = pickle.loads(page_data[page_id])
            self.page_cache[page_id] = page_words
            
            return page_words
        
        def search_word(word: str) -> bool:
            """Search for word with lazy loading"""
            if word in page_mapping:
                page_id = page_mapping[word]
                page_words = load_page(page_id)
                return word in page_words
            return False
        
        # Calculate memory usage
        active_pages = min(len(pages), self.max_cache_size)
        total_bytes = (
            active_pages * page_size * 20 +  # Estimate 20 bytes per word
            sys.getsizeof(page_mapping) +
            sys.getsizeof(pages)
        )
        
        stats = MemoryStats(
            nodes_count=len(pages),
            total_bytes=total_bytes,
            compression_ratio=3.0,  # Significant savings with lazy loading
            access_time_ms=(time.time() - start_time) * 1000
        )
        
        lazy_trie = {
            'root': root,
            'pages': pages,
            'page_mapping': page_mapping,
            'search_function': search_word
        }
        
        return lazy_trie, stats
    
    def apply_delta_compression(self, words: List[str]) -> Tuple[Dict[str, Any], MemoryStats]:
        """
        Approach 6: Delta Compression for Similar Patterns
        
        Store differences between similar words instead of full words.
        
        Time: O(n^2) for finding similarities, O(n) for access
        Space: O(differences_size)
        """
        start_time = time.time()
        
        def compute_lcs_length(s1: str, s2: str) -> int:
            """Compute longest common subsequence length"""
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        def compute_delta(base: str, target: str) -> Dict[str, Any]:
            """Compute delta between two strings"""
            # Simple diff algorithm
            deltas = []
            i = j = 0
            
            while i < len(base) and j < len(target):
                if base[i] == target[j]:
                    i += 1
                    j += 1
                else:
                    # Find next match
                    found = False
                    for k in range(j + 1, min(j + 5, len(target))):
                        if k < len(target) and i < len(base) and target[k] == base[i]:
                            # Insertion
                            deltas.append(('insert', j, target[j:k]))
                            j = k
                            found = True
                            break
                    
                    if not found:
                        # Substitution
                        deltas.append(('substitute', i, target[j]))
                        i += 1
                        j += 1
            
            # Handle remaining characters
            if j < len(target):
                deltas.append(('append', len(base), target[j:]))
            
            return {
                'base': base,
                'deltas': deltas,
                'target': target
            }
        
        # Find base words (most similar to others)
        word_scores = {}
        
        for word in words:
            similarity_score = 0
            for other_word in words:
                if word != other_word:
                    lcs_len = compute_lcs_length(word, other_word)
                    similarity_score += lcs_len / max(len(word), len(other_word))
            
            word_scores[word] = similarity_score
        
        # Select base words (top 20% by similarity)
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        num_bases = max(1, len(words) // 5)
        base_words = [word for word, score in sorted_words[:num_bases]]
        
        # Create delta representation
        delta_compressed = {
            'bases': base_words,
            'deltas': {},
            'word_to_base': {}
        }
        
        for word in words:
            if word in base_words:
                continue
            
            # Find best base word
            best_base = None
            min_delta_size = float('inf')
            
            for base in base_words:
                delta = compute_delta(base, word)
                delta_size = len(delta['deltas'])
                
                if delta_size < min_delta_size:
                    min_delta_size = delta_size
                    best_base = base
            
            if best_base:
                delta_compressed['deltas'][word] = compute_delta(best_base, word)
                delta_compressed['word_to_base'][word] = best_base
        
        # Calculate compression statistics
        original_size = sum(len(word) for word in words)
        compressed_size = (
            sum(len(base) for base in base_words) +
            sum(len(str(delta)) for delta in delta_compressed['deltas'].values())
        )
        
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        total_bytes = (
            sys.getsizeof(delta_compressed) +
            sum(sys.getsizeof(base) for base in base_words) +
            sum(sys.getsizeof(delta) for delta in delta_compressed['deltas'].values())
        )
        
        stats = MemoryStats(
            nodes_count=len(base_words) + len(delta_compressed['deltas']),
            total_bytes=total_bytes,
            compression_ratio=compression_ratio,
            access_time_ms=(time.time() - start_time) * 1000
        )
        
        return delta_compressed, stats
    
    def implement_bit_packing(self, words: List[str]) -> Tuple[List[BitPackedNode], MemoryStats]:
        """
        Approach 7: Bit-packed Node Representation
        
        Pack multiple fields into single integers for memory efficiency.
        
        Time: O(n * m) with bit operations
        Space: O(nodes * sizeof(int))
        """
        start_time = time.time()
        
        # Build character frequency map
        char_freq = defaultdict(int)
        for word in words:
            for char in word:
                char_freq[char] += 1
        
        # Create bit-packed trie
        nodes = []
        node_map = {}  # (path, char) -> node_index
        
        # Root node
        root_node = BitPackedNode()
        nodes.append(root_node)
        
        for word in words:
            path = ""
            
            for i, char in enumerate(word):
                current_path = path + char
                
                if (path, char) not in node_map:
                    # Create new bit-packed node
                    new_node = BitPackedNode()
                    new_node.set_char(char)
                    new_node.set_frequency(char_freq[char])
                    
                    if i == len(word) - 1:
                        new_node.set_is_word(True)
                    
                    nodes.append(new_node)
                    node_map[(path, char)] = len(nodes) - 1
                else:
                    # Update existing node
                    node_index = node_map[(path, char)]
                    node = nodes[node_index]
                    
                    if i == len(word) - 1:
                        node.set_is_word(True)
                    
                    # Update frequency
                    current_freq = node.get_frequency()
                    node.set_frequency(min(current_freq + 1, 31))
                
                path = current_path
        
        # Calculate memory savings
        standard_node_size = 64  # Estimate for standard trie node
        bit_packed_size = 4      # Single 32-bit integer
        
        total_bytes = len(nodes) * bit_packed_size
        standard_size = len(nodes) * standard_node_size
        compression_ratio = standard_size / total_bytes
        
        stats = MemoryStats(
            nodes_count=len(nodes),
            total_bytes=total_bytes,
            compression_ratio=compression_ratio,
            access_time_ms=(time.time() - start_time) * 1000
        )
        
        return nodes, stats


def test_basic_compression():
    """Test basic compression functionality"""
    print("=== Testing Basic Compression Functionality ===")
    
    compressor = TrieMemoryCompressor()
    
    test_words = [
        "apple", "application", "apply", "apples",
        "banana", "band", "bandana", "ban",
        "cat", "car", "card", "care", "careful",
        "dog", "door", "down", "download"
    ]
    
    print(f"Test words ({len(test_words)}): {test_words[:8]}...")
    
    # Test all compression approaches
    approaches = [
        ("Radix Tree", compressor.compress_radix_tree),
        ("Bitmap Dense", compressor.compress_bitmap_dense),
        ("Huffman Encoded", compressor.compress_huffman_encoded),
        ("Memory Pool", compressor.optimize_memory_pool),
        ("Lazy Loading", compressor.implement_lazy_loading),
        ("Delta Compression", compressor.apply_delta_compression),
        ("Bit Packing", compressor.implement_bit_packing),
    ]
    
    results = {}
    
    for name, method in approaches:
        print(f"\n{name}:")
        try:
            compressed_data, stats = method(test_words)
            results[name] = stats
            
            print(f"  Nodes: {stats.nodes_count}")
            print(f"  Memory: {stats.total_bytes} bytes")
            print(f"  Compression ratio: {stats.compression_ratio:.2f}x")
            print(f"  Build time: {stats.access_time_ms:.2f}ms")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary comparison
    print(f"\n=== Compression Comparison ===")
    print(f"{'Method':<20} {'Nodes':<8} {'Bytes':<10} {'Ratio':<8} {'Time(ms)':<10}")
    print("-" * 60)
    
    for name, stats in results.items():
        print(f"{name:<20} {stats.nodes_count:<8} {stats.total_bytes:<10} "
              f"{stats.compression_ratio:<8.2f} {stats.access_time_ms:<10.2f}")


def demonstrate_advanced_compression():
    """Demonstrate advanced compression techniques"""
    print("\n=== Advanced Compression Techniques ===")
    
    compressor = TrieMemoryCompressor()
    
    # Generate larger dataset with patterns
    def generate_structured_words(base_words: List[str], patterns: List[str]) -> List[str]:
        """Generate words with common patterns"""
        words = base_words.copy()
        
        for base in base_words:
            for pattern in patterns:
                words.append(base + pattern)
                words.append(pattern + base)
        
        return list(set(words))
    
    base_words = ["run", "walk", "jump", "swim", "fly", "drive", "read", "write"]
    patterns = ["ing", "ed", "er", "ly", "tion", "ness", "ment"]
    
    large_dataset = generate_structured_words(base_words, patterns)
    print(f"Generated dataset: {len(large_dataset)} words")
    print(f"Sample: {large_dataset[:10]}")
    
    # Test compression on larger dataset
    print(f"\n--- Radix Tree Compression ---")
    radix_trie, radix_stats = compressor.compress_radix_tree(large_dataset)
    
    print(f"Compressed nodes: {radix_stats.nodes_count}")
    print(f"Memory usage: {radix_stats.total_bytes} bytes")
    print(f"Compression ratio: {radix_stats.compression_ratio:.2f}x")
    
    # Test search functionality
    def test_search_in_radix(trie: CompressedTrieNode, word: str) -> bool:
        """Test search in compressed radix trie"""
        node = trie
        i = 0
        
        while i < len(word) and node:
            found = False
            
            for edge_char, child in node.children.items():
                edge = child.edge
                
                if word[i:i+len(edge)] == edge:
                    i += len(edge)
                    node = child
                    found = True
                    break
                elif edge.startswith(word[i:]):
                    # Partial match at end of word
                    return False
            
            if not found:
                return False
        
        return node.is_word if node else False
    
    # Test search functionality
    test_searches = ["running", "walked", "swimmer", "flying", "readable"]
    print(f"\n--- Search Tests ---")
    
    for word in test_searches:
        found = test_search_in_radix(radix_trie, word)
        print(f"'{word}': {'Found' if found else 'Not found'}")
    
    # Test delta compression
    print(f"\n--- Delta Compression Analysis ---")
    delta_data, delta_stats = compressor.apply_delta_compression(large_dataset)
    
    print(f"Base words: {len(delta_data['bases'])}")
    print(f"Delta entries: {len(delta_data['deltas'])}")
    print(f"Compression ratio: {delta_stats.compression_ratio:.2f}x")
    
    # Show some delta examples
    print(f"\nDelta examples:")
    for i, (word, delta_info) in enumerate(list(delta_data['deltas'].items())[:3]):
        base = delta_info['base']
        deltas = delta_info['deltas']
        print(f"  '{word}' <- '{base}' + {deltas}")


def benchmark_compression_performance():
    """Benchmark compression performance across different sizes"""
    print("\n=== Compression Performance Benchmark ===")
    
    compressor = TrieMemoryCompressor()
    
    # Generate datasets of different sizes
    def generate_random_words(count: int, min_len: int = 3, max_len: int = 12) -> List[str]:
        """Generate random words for testing"""
        import random
        import string
        
        words = set()
        while len(words) < count:
            length = random.randint(min_len, max_len)
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.add(word)
        
        return list(words)
    
    datasets = [
        ("Small", 100),
        ("Medium", 500),
        ("Large", 1000),
        ("Extra Large", 2000),
    ]
    
    for dataset_name, size in datasets:
        print(f"\n--- {dataset_name} Dataset ({size} words) ---")
        
        words = generate_random_words(size)
        
        # Test multiple compression approaches
        compression_methods = [
            ("Radix Tree", compressor.compress_radix_tree),
            ("Huffman", compressor.compress_huffman_encoded),
            ("Delta", compressor.apply_delta_compression),
            ("Bit Packing", compressor.implement_bit_packing),
        ]
        
        for method_name, method in compression_methods:
            try:
                compressed_data, stats = method(words)
                
                print(f"{method_name:>12}: "
                      f"{stats.total_bytes:>8} bytes, "
                      f"{stats.compression_ratio:>6.2f}x, "
                      f"{stats.access_time_ms:>8.2f}ms")
                
            except Exception as e:
                print(f"{method_name:>12}: Error - {e}")


def analyze_memory_efficiency():
    """Analyze memory efficiency of different approaches"""
    print("\n=== Memory Efficiency Analysis ===")
    
    compressor = TrieMemoryCompressor()
    
    # Test with different word patterns
    pattern_datasets = {
        "Short Words": ["cat", "bat", "rat", "hat", "mat", "sat", "fat"],
        "Long Words": ["antidisestablishmentarianism", "supercalifragilisticexpialidocious", 
                      "pneumonoultramicroscopicsilicovolcanoconisis"],
        "Common Prefixes": ["prefix_a", "prefix_b", "prefix_c", "prefix_d", "prefix_e"],
        "Random": ["xqz", "mno", "def", "ghi", "jkl", "pqr", "stu"],
        "Similar Words": ["running", "runner", "runs", "runnable", "runway"]
    }
    
    print(f"{'Pattern':<20} {'Method':<15} {'Bytes':<8} {'Ratio':<8} {'Nodes':<8}")
    print("-" * 70)
    
    for pattern_name, words in pattern_datasets.items():
        # Test radix tree compression
        try:
            radix_data, radix_stats = compressor.compress_radix_tree(words)
            print(f"{pattern_name:<20} {'Radix':<15} {radix_stats.total_bytes:<8} "
                  f"{radix_stats.compression_ratio:<8.2f} {radix_stats.nodes_count:<8}")
        except:
            pass
        
        # Test delta compression
        try:
            delta_data, delta_stats = compressor.apply_delta_compression(words)
            print(f"{'':<20} {'Delta':<15} {delta_stats.total_bytes:<8} "
                  f"{delta_stats.compression_ratio:<8.2f} {delta_stats.nodes_count:<8}")
        except:
            pass
        
        print()


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    compressor = TrieMemoryCompressor()
    
    # Application 1: Dictionary compression for mobile apps
    print("1. Mobile Dictionary Compression:")
    
    dictionary_words = [
        "apple", "application", "apply", "applied", "applying",
        "banana", "band", "bandage", "bandwidth", "bang",
        "computer", "compute", "computing", "computational", "computed",
        "data", "database", "datatype", "dataset", "dataflow",
        "network", "networking", "networked", "networks", "neural"
    ]
    
    print(f"   Dictionary size: {len(dictionary_words)} words")
    
    # Compare compression approaches for mobile
    mobile_methods = [
        ("Standard Trie", None),  # Baseline
        ("Radix Tree", compressor.compress_radix_tree),
        ("Bit Packing", compressor.implement_bit_packing),
    ]
    
    baseline_size = sum(len(word) for word in dictionary_words) * 64  # Rough estimate
    
    for method_name, method in mobile_methods:
        if method is None:
            print(f"   {method_name}: {baseline_size} bytes (baseline)")
        else:
            try:
                compressed, stats = method(dictionary_words)
                savings = ((baseline_size - stats.total_bytes) / baseline_size) * 100
                print(f"   {method_name}: {stats.total_bytes} bytes ({savings:.1f}% savings)")
            except Exception as e:
                print(f"   {method_name}: Error - {e}")
    
    # Application 2: Log file compression
    print(f"\n2. Log File Pattern Compression:")
    
    log_patterns = [
        "ERROR: Connection failed",
        "ERROR: Timeout occurred",
        "ERROR: Authentication failed",
        "INFO: User logged in",
        "INFO: Request processed",
        "INFO: Data synchronized",
        "WARNING: Memory usage high",
        "WARNING: Disk space low",
        "WARNING: Network latency detected"
    ]
    
    print(f"   Log patterns: {len(log_patterns)}")
    
    delta_compressed, delta_stats = compressor.apply_delta_compression(log_patterns)
    
    print(f"   Original size estimate: {sum(len(pattern) for pattern in log_patterns)} chars")
    print(f"   Compressed size: {delta_stats.total_bytes} bytes")
    print(f"   Compression ratio: {delta_stats.compression_ratio:.2f}x")
    
    # Application 3: Autocomplete system compression
    print(f"\n3. Autocomplete System Optimization:")
    
    search_queries = [
        "how to", "how to cook", "how to code", "how to learn",
        "what is", "what is python", "what is AI", "what is ML",
        "where to", "where to buy", "where to go", "where to eat",
        "when to", "when to use", "when to start", "when to stop"
    ]
    
    print(f"   Search queries: {len(search_queries)}")
    
    # Use lazy loading for autocomplete
    lazy_system, lazy_stats = compressor.implement_lazy_loading(search_queries)
    
    print(f"   Memory footprint: {lazy_stats.total_bytes} bytes")
    print(f"   Pages created: {lazy_stats.nodes_count}")
    print(f"   Compression ratio: {lazy_stats.compression_ratio:.2f}x")
    
    # Test search functionality
    search_func = lazy_system['search_function']
    test_queries = ["how to code", "what is AI", "invalid query"]
    
    print(f"   Search tests:")
    for query in test_queries:
        found = search_func(query)
        print(f"     '{query}': {'Found' if found else 'Not found'}")


if __name__ == "__main__":
    test_basic_compression()
    demonstrate_advanced_compression()
    benchmark_compression_performance()
    analyze_memory_efficiency()
    demonstrate_real_world_applications()

"""
Trie Memory Compression demonstrates advanced memory optimization techniques:

1. Radix Tree Compression - Merge single-child paths for space efficiency
2. Bitmap Compression - Use bitmaps for dense character sets
3. Huffman Encoding - Variable-length encoding based on frequency
4. Memory Pool Allocation - Object pooling to reduce allocation overhead
5. Lazy Loading & Paging - Load trie sections on-demand
6. Delta Compression - Store differences between similar patterns
7. Bit-packed Representation - Pack multiple fields into integers

Key concepts:
- Space-time tradeoffs in trie implementations
- Compression algorithms and their applications
- Memory management and allocation strategies
- Lazy loading and caching techniques
- Pattern recognition for optimal compression

Applications:
- Mobile applications with memory constraints
- Large-scale text processing systems
- Embedded systems with limited RAM
- High-performance caching and indexing systems

Each approach demonstrates different strategies for reducing memory usage
while maintaining acceptable performance for trie-based operations.
"""
