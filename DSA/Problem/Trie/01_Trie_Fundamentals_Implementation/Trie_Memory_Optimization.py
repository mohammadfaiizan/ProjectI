"""
Trie Memory Optimization - Multiple Approaches
Difficulty: Hard

Advanced memory optimization techniques for Trie data structures.
Explore various methods to reduce memory usage while maintaining performance.
"""

from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
import array
import sys
import gc

class CompactTrieNode:
    """Memory-optimized trie node using __slots__"""
    __slots__ = ['children', 'is_end', 'value']
    
    def __init__(self):
        self.children = None  # Lazy initialization
        self.is_end = False
        self.value = None

class ArrayTrieNode:
    """Array-based trie node for lowercase letters"""
    __slots__ = ['children', 'is_end', 'value']
    
    def __init__(self):
        self.children = None  # array.array('l', [-1] * 26) when needed
        self.is_end = False
        self.value = None

class MemoryOptimizedTrie1:
    """
    Approach 1: Lazy Initialization with __slots__
    
    Use __slots__ to reduce memory overhead and lazy initialization.
    
    Memory: Reduced by ~40% compared to standard dict-based nodes
    Time: Slightly slower due to lazy initialization overhead
    """
    
    def __init__(self):
        self.root = CompactTrieNode()
        self.node_count = 1
        self.total_memory = 0
    
    def insert(self, word: str, value: Any = None) -> None:
        """Insert word with lazy node creation"""
        node = self.root
        
        for char in word:
            if node.children is None:
                node.children = {}
                self.node_count += 1
            
            if char not in node.children:
                node.children[char] = CompactTrieNode()
                self.node_count += 1
            
            node = node.children[char]
        
        node.is_end = True
        node.value = value
    
    def search(self, word: str) -> bool:
        """Search for word in trie"""
        node = self.root
        
        for char in word:
            if node.children is None or char not in node.children:
                return False
            node = node.children[char]
        
        return node.is_end
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Estimate memory usage"""
        return {
            "node_count": self.node_count,
            "estimated_bytes": self.node_count * 64,  # Rough estimate
            "python_object_overhead": self.node_count * 28
        }

class MemoryOptimizedTrie2:
    """
    Approach 2: Array-based for Limited Alphabet
    
    Use arrays instead of dictionaries for fixed alphabets.
    
    Memory: Fixed 26 * 4 bytes per node for lowercase letters
    Time: O(1) character lookup
    """
    
    def __init__(self):
        self.nodes = [ArrayTrieNode()]  # Store nodes in array
        self.free_indices = []  # Reuse freed node indices
        self.node_count = 1
    
    def _char_to_index(self, char: str) -> int:
        """Convert character to array index"""
        return ord(char) - ord('a')
    
    def _allocate_node(self) -> int:
        """Allocate new node, reusing freed indices if available"""
        if self.free_indices:
            index = self.free_indices.pop()
            # Reset the reused node
            self.nodes[index] = ArrayTrieNode()
            return index
        else:
            self.nodes.append(ArrayTrieNode())
            self.node_count += 1
            return len(self.nodes) - 1
    
    def insert(self, word: str, value: Any = None) -> None:
        """Insert word using array-based nodes"""
        current_index = 0
        
        for char in word.lower():
            if not char.isalpha():
                continue
            
            node = self.nodes[current_index]
            char_index = self._char_to_index(char)
            
            # Lazy initialization of children array
            if node.children is None:
                node.children = array.array('l', [-1] * 26)
            
            # Create new node if needed
            if node.children[char_index] == -1:
                new_node_index = self._allocate_node()
                node.children[char_index] = new_node_index
            
            current_index = node.children[char_index]
        
        self.nodes[current_index].is_end = True
        self.nodes[current_index].value = value
    
    def search(self, word: str) -> bool:
        """Search for word"""
        current_index = 0
        
        for char in word.lower():
            if not char.isalpha():
                continue
            
            node = self.nodes[current_index]
            if node.children is None:
                return False
            
            char_index = self._char_to_index(char)
            if node.children[char_index] == -1:
                return False
            
            current_index = node.children[char_index]
        
        return self.nodes[current_index].is_end
    
    def delete(self, word: str) -> bool:
        """Delete word and free unused nodes"""
        # Mark node as not end
        current_index = 0
        path = [0]  # Track path for cleanup
        
        for char in word.lower():
            if not char.isalpha():
                continue
            
            node = self.nodes[current_index]
            if node.children is None:
                return False
            
            char_index = self._char_to_index(char)
            if node.children[char_index] == -1:
                return False
            
            current_index = node.children[char_index]
            path.append(current_index)
        
        if not self.nodes[current_index].is_end:
            return False
        
        self.nodes[current_index].is_end = False
        self.nodes[current_index].value = None
        
        # Clean up unused nodes from leaf to root
        self._cleanup_path(path, word)
        return True
    
    def _cleanup_path(self, path: List[int], word: str) -> None:
        """Clean up unused nodes in path"""
        word = word.lower()
        
        for i in range(len(path) - 1, 0, -1):
            node_index = path[i]
            parent_index = path[i - 1]
            node = self.nodes[node_index]
            
            # Check if node can be freed
            if (not node.is_end and 
                (node.children is None or all(child == -1 for child in node.children))):
                
                # Free this node
                char = word[i - 1]
                char_index = self._char_to_index(char)
                parent_node = self.nodes[parent_index]
                
                if parent_node.children is not None:
                    parent_node.children[char_index] = -1
                
                # Add to free list
                self.free_indices.append(node_index)
            else:
                break  # Stop cleanup if node is still needed
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Calculate actual memory usage"""
        active_nodes = self.node_count - len(self.free_indices)
        
        return {
            "total_nodes": self.node_count,
            "active_nodes": active_nodes,
            "free_nodes": len(self.free_indices),
            "array_memory": active_nodes * 26 * 4,  # 4 bytes per int
            "estimated_total": active_nodes * 150  # Including object overhead
        }

class MemoryOptimizedTrie3:
    """
    Approach 3: Compressed Path Trie (Radix Tree)
    
    Compress single-child paths to reduce node count.
    
    Memory: Significantly reduced for sparse tries
    Time: May be slower due to path compression/decompression
    """
    
    def __init__(self):
        self.root = {"path": "", "children": {}, "is_end": False, "value": None}
        self.node_count = 1
    
    def insert(self, word: str, value: Any = None) -> None:
        """Insert with path compression"""
        self._insert_recursive(self.root, word, value, 0)
    
    def _insert_recursive(self, node: dict, word: str, value: Any, index: int) -> None:
        """Recursive insertion with compression"""
        if index >= len(word):
            node["is_end"] = True
            node["value"] = value
            return
        
        path = node["path"]
        remaining = word[index:]
        
        # Handle compressed path
        if path:
            common_len = self._find_common_prefix(path, remaining)
            
            if common_len == len(path):
                # Path fully matches, continue
                if len(remaining) == len(path):
                    node["is_end"] = True
                    node["value"] = value
                else:
                    next_char = remaining[len(path)]
                    if next_char not in node["children"]:
                        # Create new compressed child
                        remaining_suffix = remaining[len(path) + 1:]
                        node["children"][next_char] = {
                            "path": remaining_suffix,
                            "children": {},
                            "is_end": True,
                            "value": value
                        }
                        self.node_count += 1
                    else:
                        self._insert_recursive(node["children"][next_char], word, value, 
                                             index + len(path) + 1)
            else:
                # Need to split the path
                self._split_path(node, word, value, index, common_len)
        else:
            # No compressed path
            next_char = remaining[0]
            if next_char not in node["children"]:
                # Create new compressed child
                remaining_suffix = remaining[1:]
                node["children"][next_char] = {
                    "path": remaining_suffix,
                    "children": {},
                    "is_end": True,
                    "value": value
                }
                self.node_count += 1
            else:
                self._insert_recursive(node["children"][next_char], word, value, index + 1)
    
    def _find_common_prefix(self, str1: str, str2: str) -> int:
        """Find length of common prefix"""
        i = 0
        while i < len(str1) and i < len(str2) and str1[i] == str2[i]:
            i += 1
        return i
    
    def _split_path(self, node: dict, word: str, value: Any, index: int, common_len: int) -> None:
        """Split compressed path when inserting divergent word"""
        old_path = node["path"]
        old_children = node["children"]
        old_is_end = node["is_end"]
        old_value = node["value"]
        
        # Update current node
        node["path"] = old_path[:common_len]
        node["children"] = {}
        node["is_end"] = False
        node["value"] = None
        
        # Create child for old path continuation
        if common_len < len(old_path):
            old_remaining = old_path[common_len:]
            first_char = old_remaining[0]
            node["children"][first_char] = {
                "path": old_remaining[1:],
                "children": old_children,
                "is_end": old_is_end,
                "value": old_value
            }
            self.node_count += 1
        
        # Create child for new word continuation
        remaining = word[index + common_len:]
        if remaining:
            first_char = remaining[0]
            node["children"][first_char] = {
                "path": remaining[1:],
                "children": {},
                "is_end": True,
                "value": value
            }
            self.node_count += 1
        else:
            node["is_end"] = True
            node["value"] = value
    
    def search(self, word: str) -> bool:
        """Search in compressed trie"""
        return self._search_recursive(self.root, word, 0)
    
    def _search_recursive(self, node: dict, word: str, index: int) -> bool:
        """Recursive search"""
        if index >= len(word):
            return node["is_end"]
        
        path = node["path"]
        remaining = word[index:]
        
        if path:
            if remaining.startswith(path):
                if len(remaining) == len(path):
                    return node["is_end"]
                else:
                    next_char = remaining[len(path)]
                    if next_char in node["children"]:
                        return self._search_recursive(node["children"][next_char], 
                                                    word, index + len(path) + 1)
                    return False
            else:
                return False
        else:
            next_char = remaining[0]
            if next_char in node["children"]:
                return self._search_recursive(node["children"][next_char], word, index + 1)
            return False
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Calculate compressed trie memory usage"""
        return {
            "total_nodes": self.node_count,
            "estimated_savings": "Varies based on compression ratio",
            "compression_ratio": self._calculate_compression_ratio()
        }
    
    def _calculate_compression_ratio(self) -> float:
        """Estimate compression ratio"""
        total_path_length = self._calculate_total_path_length(self.root)
        if total_path_length == 0:
            return 1.0
        return total_path_length / self.node_count
    
    def _calculate_total_path_length(self, node: dict) -> int:
        """Calculate total compressed path length"""
        total = len(node["path"])
        for child in node["children"].values():
            total += self._calculate_total_path_length(child)
        return total

class MemoryOptimizedTrie4:
    """
    Approach 4: Bit-packed Trie
    
    Use bit manipulation to pack multiple values into single integers.
    
    Memory: Highly optimized for specific use cases
    Time: Bit operations add overhead
    """
    
    def __init__(self):
        # Each node represented as bit-packed integer
        # Bits 0-25: children existence flags
        # Bit 26: is_end flag
        # Bits 27-31: reserved
        self.nodes = [0]  # Root node
        self.node_children = [{}]  # Map from node_id to char -> child_id
        self.node_values = {}  # node_id -> value
        self.next_node_id = 1
    
    def _get_is_end(self, node_bits: int) -> bool:
        """Extract is_end flag from bit-packed node"""
        return bool(node_bits & (1 << 26))
    
    def _set_is_end(self, node_bits: int, is_end: bool) -> int:
        """Set is_end flag in bit-packed node"""
        if is_end:
            return node_bits | (1 << 26)
        else:
            return node_bits & ~(1 << 26)
    
    def _has_child(self, node_bits: int, char: str) -> bool:
        """Check if node has child for character"""
        if not char.islower():
            return False
        char_bit = ord(char) - ord('a')
        return bool(node_bits & (1 << char_bit))
    
    def _set_child_flag(self, node_bits: int, char: str) -> int:
        """Set child existence flag"""
        if not char.islower():
            return node_bits
        char_bit = ord(char) - ord('a')
        return node_bits | (1 << char_bit)
    
    def insert(self, word: str, value: Any = None) -> None:
        """Insert word using bit-packed nodes"""
        current_id = 0
        
        for char in word.lower():
            if not char.islower():
                continue
            
            current_bits = self.nodes[current_id]
            
            if not self._has_child(current_bits, char):
                # Create new child
                child_id = self.next_node_id
                self.next_node_id += 1
                self.nodes.append(0)
                self.node_children.append({})
                
                # Update parent's child flag
                self.nodes[current_id] = self._set_child_flag(current_bits, char)
                self.node_children[current_id][char] = child_id
            
            current_id = self.node_children[current_id][char]
        
        # Mark as end node
        self.nodes[current_id] = self._set_is_end(self.nodes[current_id], True)
        if value is not None:
            self.node_values[current_id] = value
    
    def search(self, word: str) -> bool:
        """Search in bit-packed trie"""
        current_id = 0
        
        for char in word.lower():
            if not char.islower():
                continue
            
            if not self._has_child(self.nodes[current_id], char):
                return False
            
            current_id = self.node_children[current_id][char]
        
        return self._get_is_end(self.nodes[current_id])
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Calculate bit-packed trie memory usage"""
        return {
            "total_nodes": len(self.nodes),
            "bits_per_node": 32,
            "total_bits": len(self.nodes) * 32,
            "bytes_for_nodes": len(self.nodes) * 4,
            "children_dict_overhead": sum(len(children) * 16 for children in self.node_children),
            "estimated_total": len(self.nodes) * 20  # Optimistic estimate
        }

class MemoryOptimizedTrie5:
    """
    Approach 5: Memory Pool with Object Reuse
    
    Pre-allocate node pool and reuse objects.
    
    Memory: Predictable memory usage
    Time: Fast allocation/deallocation
    """
    
    def __init__(self, pool_size: int = 10000):
        self.pool_size = pool_size
        self.node_pool = [self._create_node() for _ in range(pool_size)]
        self.free_nodes = list(range(pool_size))
        self.used_nodes = set()
        self.root_id = self._allocate_node()
        
    def _create_node(self) -> dict:
        """Create a new node structure"""
        return {
            "children": {},
            "is_end": False,
            "value": None,
            "in_use": False
        }
    
    def _allocate_node(self) -> Optional[int]:
        """Allocate node from pool"""
        if not self.free_nodes:
            return None  # Pool exhausted
        
        node_id = self.free_nodes.pop()
        node = self.node_pool[node_id]
        node["children"].clear()
        node["is_end"] = False
        node["value"] = None
        node["in_use"] = True
        self.used_nodes.add(node_id)
        
        return node_id
    
    def _free_node(self, node_id: int) -> None:
        """Return node to pool"""
        if node_id in self.used_nodes:
            self.node_pool[node_id]["in_use"] = False
            self.used_nodes.remove(node_id)
            self.free_nodes.append(node_id)
    
    def insert(self, word: str, value: Any = None) -> bool:
        """Insert word (returns False if pool exhausted)"""
        current_id = self.root_id
        
        for char in word:
            current_node = self.node_pool[current_id]
            
            if char not in current_node["children"]:
                child_id = self._allocate_node()
                if child_id is None:
                    return False  # Pool exhausted
                current_node["children"][char] = child_id
            
            current_id = current_node["children"][char]
        
        end_node = self.node_pool[current_id]
        end_node["is_end"] = True
        end_node["value"] = value
        return True
    
    def search(self, word: str) -> bool:
        """Search for word"""
        current_id = self.root_id
        
        for char in word:
            current_node = self.node_pool[current_id]
            
            if char not in current_node["children"]:
                return False
            
            current_id = current_node["children"][char]
        
        return self.node_pool[current_id]["is_end"]
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory pool statistics"""
        return {
            "pool_size": self.pool_size,
            "used_nodes": len(self.used_nodes),
            "free_nodes": len(self.free_nodes),
            "memory_efficiency": len(self.used_nodes) / self.pool_size * 100,
            "estimated_total_bytes": self.pool_size * 100  # Rough estimate
        }


def test_memory_optimizations():
    """Test all memory optimization approaches"""
    print("=== Testing Memory Optimizations ===")
    
    implementations = [
        ("Lazy Init + Slots", MemoryOptimizedTrie1),
        ("Array-based", MemoryOptimizedTrie2),
        ("Compressed Path", MemoryOptimizedTrie3),
        ("Bit-packed", MemoryOptimizedTrie4),
        ("Memory Pool", lambda: MemoryOptimizedTrie5(1000)),
    ]
    
    test_words = ["cat", "cats", "car", "card", "care", "careful", "dog", "dogs", "doggy"]
    
    for name, TrieClass in implementations:
        print(f"\n--- Testing {name} ---")
        
        trie = TrieClass()
        
        # Insert words
        success_count = 0
        for word in test_words:
            if hasattr(trie, 'insert'):
                result = trie.insert(word, len(word))
                if result is not False:  # Memory pool returns False on failure
                    success_count += 1
            else:
                trie.insert(word, len(word))
                success_count += 1
        
        print(f"  Inserted {success_count}/{len(test_words)} words")
        
        # Test search
        found_count = 0
        for word in test_words:
            if trie.search(word):
                found_count += 1
        
        print(f"  Found {found_count}/{len(test_words)} words")
        
        # Memory usage
        memory_info = trie.get_memory_usage()
        print(f"  Memory usage: {memory_info}")


def benchmark_memory_usage():
    """Benchmark actual memory usage"""
    print("\n=== Memory Usage Benchmark ===")
    
    import tracemalloc
    import random
    import string
    
    # Generate test data
    words = []
    for _ in range(1000):
        length = random.randint(3, 10)
        word = ''.join(random.choices(string.ascii_lowercase, k=length))
        words.append(word)
    
    words = list(set(words))  # Remove duplicates
    
    implementations = [
        ("Standard Dict Trie", dict),  # Placeholder for standard implementation
        ("Lazy + Slots", MemoryOptimizedTrie1),
        ("Array-based", MemoryOptimizedTrie2),
        ("Compressed", MemoryOptimizedTrie3),
        ("Bit-packed", MemoryOptimizedTrie4),
    ]
    
    print(f"Testing with {len(words)} unique words")
    
    for name, TrieClass in implementations[1:]:  # Skip standard for now
        print(f"\n--- {name} ---")
        
        # Start memory tracing
        tracemalloc.start()
        
        trie = TrieClass()
        
        # Insert all words
        for word in words:
            trie.insert(word)
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"  Current memory: {current / 1024:.1f} KB")
        print(f"  Peak memory: {peak / 1024:.1f} KB")
        
        # Internal memory statistics
        memory_stats = trie.get_memory_usage()
        print(f"  Internal stats: {memory_stats}")


def demonstrate_compression_benefits():
    """Demonstrate compression benefits"""
    print("\n=== Compression Benefits Demo ===")
    
    # Create test data with common prefixes (good for compression)
    prefixed_words = []
    prefixes = ["pre", "pro", "anti", "super", "micro"]
    suffixes = ["fix", "cess", "gram", "file", "soft", "wave", "data"]
    
    for prefix in prefixes:
        for suffix in suffixes:
            prefixed_words.append(prefix + suffix)
    
    print(f"Testing with {len(prefixed_words)} words with common prefixes:")
    print(f"Sample words: {prefixed_words[:10]}")
    
    # Test regular vs compressed trie
    implementations = [
        ("Regular Trie", MemoryOptimizedTrie1),
        ("Compressed Trie", MemoryOptimizedTrie3),
    ]
    
    for name, TrieClass in implementations:
        trie = TrieClass()
        
        for word in prefixed_words:
            trie.insert(word)
        
        memory_info = trie.get_memory_usage()
        print(f"\n{name}:")
        print(f"  {memory_info}")
        
        # Test search performance
        import time
        start_time = time.time()
        
        for word in prefixed_words:
            trie.search(word)
        
        search_time = time.time() - start_time
        print(f"  Search time: {search_time*1000:.2f}ms")


def analyze_memory_patterns():
    """Analyze memory usage patterns"""
    print("\n=== Memory Pattern Analysis ===")
    
    # Test different data patterns
    patterns = [
        ("Random strings", [f"{''.join(random.choices(string.ascii_lowercase, k=5))}" 
                           for _ in range(100)]),
        ("Common prefixes", [f"prefix{i}" for i in range(100)]),
        ("Common suffixes", [f"{i}suffix" for i in range(100)]),
        ("Single char diff", [f"word{chr(ord('a') + i)}" for i in range(26)]),
    ]
    
    for pattern_name, words in patterns:
        print(f"\n--- {pattern_name} ---")
        
        # Test with array-based trie
        trie = MemoryOptimizedTrie2()
        
        for word in words:
            trie.insert(word)
        
        memory_info = trie.get_memory_usage()
        efficiency = memory_info['active_nodes'] / len(words) * 100
        
        print(f"  Words: {len(words)}")
        print(f"  Active nodes: {memory_info['active_nodes']}")
        print(f"  Node efficiency: {efficiency:.1f}%")
        print(f"  Memory per word: {memory_info['estimated_total'] / len(words):.1f} bytes")


def demonstrate_real_world_scenarios():
    """Demonstrate real-world memory optimization scenarios"""
    print("\n=== Real-World Scenarios ===")
    
    # Scenario 1: Large dictionary with memory constraints
    print("1. Large Dictionary Optimization:")
    
    # Simulate large dictionary
    large_dict = []
    for i in range(1000):
        word = ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 12)))
        large_dict.append(word)
    
    large_dict = list(set(large_dict))  # Remove duplicates
    
    print(f"   Dictionary size: {len(large_dict)} words")
    
    # Compare memory usage
    for name, TrieClass in [("Standard", MemoryOptimizedTrie1), 
                           ("Compressed", MemoryOptimizedTrie3)]:
        trie = TrieClass()
        
        for word in large_dict:
            trie.insert(word)
        
        memory_info = trie.get_memory_usage()
        print(f"   {name}: {memory_info}")
    
    # Scenario 2: Embedded system with limited memory
    print(f"\n2. Embedded System Optimization:")
    
    # Small memory pool
    embedded_trie = MemoryOptimizedTrie5(pool_size=100)
    
    test_words = ["sensor", "data", "temp", "humidity", "pressure", "voltage"]
    
    print(f"   Available memory pool: 100 nodes")
    print(f"   Inserting system vocabulary: {test_words}")
    
    for word in test_words:
        success = embedded_trie.insert(word)
        print(f"     '{word}': {'✓' if success else '✗'}")
    
    memory_stats = embedded_trie.get_memory_usage()
    print(f"   Memory utilization: {memory_stats['memory_efficiency']:.1f}%")
    
    # Scenario 3: Cache-friendly autocomplete
    print(f"\n3. Cache-friendly Autocomplete:")
    
    # Array-based for better cache locality
    autocomplete_trie = MemoryOptimizedTrie2()
    
    # Common search terms
    search_terms = [
        "search", "settings", "profile", "account", "password",
        "email", "phone", "address", "payment", "history"
    ]
    
    for term in search_terms:
        autocomplete_trie.insert(term)
    
    print(f"   Search terms: {len(search_terms)}")
    print(f"   Memory layout: Array-based for cache efficiency")
    
    # Simulate autocomplete queries
    prefixes = ["s", "se", "p", "pa", "e"]
    for prefix in prefixes:
        # This would normally return suggestions
        found = autocomplete_trie.search(prefix)
        print(f"   Prefix '{prefix}': cache-friendly lookup")


if __name__ == "__main__":
    test_memory_optimizations()
    benchmark_memory_usage()
    demonstrate_compression_benefits()
    analyze_memory_patterns()
    demonstrate_real_world_scenarios()

"""
Trie Memory Optimization demonstrates advanced memory management techniques:

1. Lazy Initialization + __slots__ - Reduces object overhead by 40%
2. Array-based Nodes - Fixed memory layout for limited alphabets
3. Compressed Path Trie - Radix tree approach for sparse data
4. Bit-packed Trie - Ultra-compact representation using bit manipulation
5. Memory Pool - Pre-allocated object reuse for predictable memory usage

Each approach addresses different memory constraints from embedded systems
to large-scale dictionary applications with specific optimization strategies.
"""
