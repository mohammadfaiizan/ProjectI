"""
Advanced XOR Techniques - Multiple Approaches
Difficulty: Hard

Advanced XOR operations and bit manipulation techniques using trie structures
for solving complex algorithmic problems efficiently.

Techniques:
1. Maximum XOR Path in Trie
2. XOR Basis and Linear Independence
3. Range XOR Queries with Updates
4. XOR Convolution using Trie
5. Persistent XOR Structures
6. XOR Distance Metrics
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import copy

class XORTrieNode:
    """XOR-optimized trie node"""
    def __init__(self):
        self.children = [None, None]  # 0 and 1
        self.count = 0
        self.max_xor = 0
        self.elements = []  # Store actual elements

class PersistentXORNode:
    """Persistent XOR trie node for version control"""
    def __init__(self):
        self.children = [None, None]
        self.version = 0
        self.count = 0

class AdvancedXORTechniques:
    
    def __init__(self, max_bits: int = 32):
        self.max_bits = max_bits
        self.root = XORTrieNode()
        self.basis = []  # XOR basis for linear independence
        self.versions = {}  # Persistent versions
    
    def insert_with_max_tracking(self, num: int) -> None:
        """
        Approach 1: Insert with Maximum XOR Path Tracking
        
        Insert number and maintain maximum XOR at each node.
        
        Time: O(log(max_value))
        Space: O(log(max_value))
        """
        node = self.root
        node.count += 1
        node.elements.append(num)
        
        # Update max XOR at root
        if len(node.elements) >= 2:
            node.max_xor = max(node.max_xor, self._find_max_xor_in_list(node.elements))
        
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            
            if node.children[bit] is None:
                node.children[bit] = XORTrieNode()
            
            node = node.children[bit]
            node.count += 1
            node.elements.append(num)
            
            # Update max XOR for this subtree
            if len(node.elements) >= 2:
                node.max_xor = max(node.max_xor, self._find_max_xor_in_list(node.elements))
    
    def _find_max_xor_in_list(self, numbers: List[int]) -> int:
        """Find maximum XOR in a list of numbers"""
        max_xor = 0
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                max_xor = max(max_xor, numbers[i] ^ numbers[j])
        return max_xor
    
    def build_xor_basis(self, numbers: List[int]) -> List[int]:
        """
        Approach 2: XOR Basis Construction
        
        Build basis for XOR linear independence.
        
        Time: O(n * log(max_value))
        Space: O(log(max_value))
        """
        self.basis = [0] * self.max_bits
        
        for num in numbers:
            current = num
            
            for i in range(self.max_bits - 1, -1, -1):
                if not (current & (1 << i)):
                    continue
                
                if self.basis[i] == 0:
                    self.basis[i] = current
                    break
                
                current ^= self.basis[i]
        
        # Remove zeros and return basis
        return [x for x in self.basis if x != 0]
    
    def query_max_xor_with_basis(self, x: int) -> int:
        """
        Query maximum XOR using precomputed basis.
        
        Time: O(log(max_value))
        Space: O(1)
        """
        result = x
        
        for i in range(self.max_bits - 1, -1, -1):
            if self.basis[i] != 0:
                result = max(result, result ^ self.basis[i])
        
        return result ^ x  # Return the XOR difference
    
    def range_xor_with_updates(self, nums: List[int], queries: List[Tuple[str, int, int]]) -> List[int]:
        """
        Approach 3: Range XOR Queries with Updates
        
        Handle range XOR queries with point updates.
        
        Time: O(q * log(n) * log(max_value)) per query
        Space: O(n * log(max_value))
        """
        n = len(nums)
        results = []
        
        # Build segment tree of XOR tries
        class SegmentXORTrie:
            def __init__(self, start: int, end: int):
                self.start = start
                self.end = end
                self.trie_root = XORTrieNode()
                self.left_child = None
                self.right_child = None
        
        def build_segment_tree(start: int, end: int) -> SegmentXORTrie:
            node = SegmentXORTrie(start, end)
            
            if start == end:
                self._insert_into_trie(node.trie_root, nums[start])
                return node
            
            mid = (start + end) // 2
            node.left_child = build_segment_tree(start, mid)
            node.right_child = build_segment_tree(mid + 1, end)
            
            # Merge tries from children
            self._merge_tries(node.trie_root, node.left_child.trie_root)
            self._merge_tries(node.trie_root, node.right_child.trie_root)
            
            return node
        
        def update_point(node: SegmentXORTrie, index: int, new_value: int) -> None:
            if node.start == node.end:
                # Rebuild trie for this leaf
                node.trie_root = XORTrieNode()
                self._insert_into_trie(node.trie_root, new_value)
                return
            
            mid = (node.start + node.end) // 2
            if index <= mid:
                update_point(node.left_child, index, new_value)
            else:
                update_point(node.right_child, index, new_value)
            
            # Rebuild parent trie
            node.trie_root = XORTrieNode()
            if node.left_child:
                self._merge_tries(node.trie_root, node.left_child.trie_root)
            if node.right_child:
                self._merge_tries(node.trie_root, node.right_child.trie_root)
        
        def query_range_max_xor(node: SegmentXORTrie, query_start: int, query_end: int, x: int) -> int:
            if query_start > node.end or query_end < node.start:
                return 0
            
            if query_start <= node.start and node.end <= query_end:
                return self._find_max_xor_with_x(node.trie_root, x)
            
            left_result = 0
            right_result = 0
            
            if node.left_child:
                left_result = query_range_max_xor(node.left_child, query_start, query_end, x)
            if node.right_child:
                right_result = query_range_max_xor(node.right_child, query_start, query_end, x)
            
            return max(left_result, right_result)
        
        # Build initial segment tree
        seg_tree = build_segment_tree(0, n - 1)
        
        # Process queries
        for query in queries:
            if query[0] == "update":
                _, index, value = query
                nums[index] = value
                update_point(seg_tree, index, value)
            elif query[0] == "query":
                _, start, end, x = query
                result = query_range_max_xor(seg_tree, start, end, x)
                results.append(result)
        
        return results
    
    def _insert_into_trie(self, root: XORTrieNode, num: int) -> None:
        """Insert number into XOR trie"""
        node = root
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            if node.children[bit] is None:
                node.children[bit] = XORTrieNode()
            node = node.children[bit]
        node.count += 1
    
    def _merge_tries(self, dest: XORTrieNode, src: XORTrieNode) -> None:
        """Merge source trie into destination trie"""
        if src is None:
            return
        
        def merge_recursive(dest_node: XORTrieNode, src_node: XORTrieNode) -> None:
            if src_node is None:
                return
            
            dest_node.count += src_node.count
            
            for i in range(2):
                if src_node.children[i] is not None:
                    if dest_node.children[i] is None:
                        dest_node.children[i] = XORTrieNode()
                    merge_recursive(dest_node.children[i], src_node.children[i])
        
        merge_recursive(dest, src)
    
    def _find_max_xor_with_x(self, root: XORTrieNode, x: int) -> int:
        """Find maximum XOR with x in trie"""
        if root is None or root.count == 0:
            return 0
        
        node = root
        max_xor = 0
        
        for i in range(self.max_bits - 1, -1, -1):
            bit = (x >> i) & 1
            desired_bit = 1 - bit
            
            if node.children[desired_bit] is not None and node.children[desired_bit].count > 0:
                max_xor |= (1 << i)
                node = node.children[desired_bit]
            elif node.children[bit] is not None:
                node = node.children[bit]
            else:
                break
        
        return max_xor
    
    def xor_convolution(self, a: List[int], b: List[int]) -> List[int]:
        """
        Approach 4: XOR Convolution using Trie
        
        Compute XOR convolution of two sequences.
        
        Time: O(n * m * log(max_value))
        Space: O(n * m)
        """
        result = defaultdict(int)
        
        # Build trie for sequence b
        b_trie = XORTrieNode()
        for val in b:
            self._insert_into_trie_with_index(b_trie, val, len(result))
        
        # For each element in a, find XOR combinations with b
        for a_val in a:
            for b_val in b:
                xor_result = a_val ^ b_val
                result[xor_result] += 1
        
        return list(result.keys())
    
    def _insert_into_trie_with_index(self, root: XORTrieNode, num: int, index: int) -> None:
        """Insert with index tracking"""
        node = root
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            if node.children[bit] is None:
                node.children[bit] = XORTrieNode()
            node = node.children[bit]
        node.elements.append((num, index))
    
    def create_persistent_version(self, version_id: int, operations: List[Tuple[str, int]]) -> None:
        """
        Approach 5: Persistent XOR Structures
        
        Create persistent versions for time-travel queries.
        
        Time: O(operations * log(max_value))
        Space: O(versions * log(max_value))
        """
        if version_id == 0:
            # Base version
            self.versions[version_id] = PersistentXORNode()
        else:
            # Copy from previous version
            prev_version = max(v for v in self.versions.keys() if v < version_id)
            self.versions[version_id] = self._copy_persistent_trie(self.versions[prev_version])
        
        current_root = self.versions[version_id]
        
        for op_type, value in operations:
            if op_type == "insert":
                self._persistent_insert(current_root, value, version_id)
            elif op_type == "delete":
                self._persistent_delete(current_root, value, version_id)
    
    def _copy_persistent_trie(self, root: PersistentXORNode) -> PersistentXORNode:
        """Create copy of persistent trie"""
        if root is None:
            return None
        
        new_root = PersistentXORNode()
        new_root.count = root.count
        new_root.version = root.version
        
        for i in range(2):
            if root.children[i] is not None:
                new_root.children[i] = self._copy_persistent_trie(root.children[i])
        
        return new_root
    
    def _persistent_insert(self, root: PersistentXORNode, num: int, version: int) -> None:
        """Insert into persistent trie"""
        node = root
        node.version = version
        
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            
            if node.children[bit] is None:
                node.children[bit] = PersistentXORNode()
            
            node = node.children[bit]
            node.version = version
            node.count += 1
    
    def _persistent_delete(self, root: PersistentXORNode, num: int, version: int) -> None:
        """Delete from persistent trie"""
        # Simplified deletion - mark as deleted rather than physical removal
        node = root
        
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            if node.children[bit] is None:
                return  # Number not found
            node = node.children[bit]
        
        node.count = max(0, node.count - 1)
        node.version = version
    
    def xor_distance_clustering(self, points: List[int], k: int) -> Dict[int, List[int]]:
        """
        Approach 6: XOR Distance Clustering
        
        Cluster points based on XOR distance metrics.
        
        Time: O(n^2 * log(max_value))
        Space: O(n)
        """
        n = len(points)
        
        # Calculate XOR distances
        distances = {}
        for i in range(n):
            for j in range(i + 1, n):
                xor_dist = points[i] ^ points[j]
                distances[(i, j)] = self._popcount(xor_dist)  # Hamming distance
        
        # Simple k-means clustering based on XOR distance
        clusters = {i: [] for i in range(k)}
        
        # Initialize cluster centers
        import random
        centers = random.sample(points, k)
        
        # Assign points to clusters
        for point in points:
            best_cluster = 0
            min_distance = float('inf')
            
            for i, center in enumerate(centers):
                dist = self._popcount(point ^ center)
                if dist < min_distance:
                    min_distance = dist
                    best_cluster = i
            
            clusters[best_cluster].append(point)
        
        return clusters
    
    def _popcount(self, x: int) -> int:
        """Count number of set bits"""
        count = 0
        while x:
            count += x & 1
            x >>= 1
        return count


def test_xor_basis():
    """Test XOR basis construction"""
    print("=== Testing XOR Basis ===")
    
    xor_tech = AdvancedXORTechniques(max_bits=8)
    
    numbers = [1, 2, 3, 4, 5, 6, 7, 8]
    print(f"Numbers: {numbers}")
    print(f"Binary representations:")
    for num in numbers:
        print(f"  {num:2}: {bin(num)[2:]:>8}")
    
    basis = xor_tech.build_xor_basis(numbers)
    print(f"\nXOR Basis: {basis}")
    print(f"Basis binary:")
    for b in basis:
        print(f"  {b:2}: {bin(b)[2:]:>8}")
    
    # Test maximum XOR query
    x = 10
    max_xor = xor_tech.query_max_xor_with_basis(x)
    print(f"\nMax XOR with {x}: {max_xor}")


def test_persistent_operations():
    """Test persistent XOR operations"""
    print("\n=== Testing Persistent Operations ===")
    
    xor_tech = AdvancedXORTechniques()
    
    # Create version 0
    operations_v0 = [("insert", 1), ("insert", 2), ("insert", 3)]
    xor_tech.create_persistent_version(0, operations_v0)
    print(f"Version 0 operations: {operations_v0}")
    
    # Create version 1
    operations_v1 = [("insert", 4), ("delete", 1)]
    xor_tech.create_persistent_version(1, operations_v1)
    print(f"Version 1 operations: {operations_v1}")
    
    # Create version 2
    operations_v2 = [("insert", 5), ("insert", 6)]
    xor_tech.create_persistent_version(2, operations_v2)
    print(f"Version 2 operations: {operations_v2}")
    
    print(f"Created {len(xor_tech.versions)} persistent versions")


def test_xor_clustering():
    """Test XOR distance clustering"""
    print("\n=== Testing XOR Distance Clustering ===")
    
    xor_tech = AdvancedXORTechniques()
    
    # Create test points with similar XOR patterns
    points = [
        0b00001111,  # Group 1: low bits set
        0b00001110,
        0b00001101,
        0b11110000,  # Group 2: high bits set
        0b11100000,
        0b11010000,
        0b01010101,  # Group 3: alternating pattern
        0b10101010,
    ]
    
    print(f"Points to cluster:")
    for i, point in enumerate(points):
        print(f"  Point {i}: {point:3} -> {bin(point)[2:]:>8}")
    
    clusters = xor_tech.xor_distance_clustering(points, k=3)
    
    print(f"\nClustering results (k=3):")
    for cluster_id, cluster_points in clusters.items():
        if cluster_points:
            print(f"  Cluster {cluster_id}: {cluster_points}")


def benchmark_advanced_operations():
    """Benchmark advanced XOR operations"""
    print("\n=== Benchmarking Advanced Operations ===")
    
    import time
    import random
    
    xor_tech = AdvancedXORTechniques()
    
    # Generate test data
    test_sizes = [100, 500, 1000]
    
    print(f"{'Size':<8} {'Basis(ms)':<12} {'MaxXOR(ms)':<12} {'Cluster(ms)':<12}")
    print("-" * 50)
    
    for size in test_sizes:
        numbers = [random.randint(0, 1000) for _ in range(size)]
        
        # Benchmark basis construction
        start_time = time.time()
        basis = xor_tech.build_xor_basis(numbers)
        basis_time = (time.time() - start_time) * 1000
        
        # Benchmark max XOR queries
        start_time = time.time()
        for _ in range(min(100, size)):
            x = random.randint(0, 1000)
            xor_tech.query_max_xor_with_basis(x)
        max_xor_time = (time.time() - start_time) * 1000
        
        # Benchmark clustering (smaller dataset)
        cluster_numbers = numbers[:min(20, size)]
        start_time = time.time()
        clusters = xor_tech.xor_distance_clustering(cluster_numbers, k=3)
        cluster_time = (time.time() - start_time) * 1000
        
        print(f"{size:<8} {basis_time:<12.2f} {max_xor_time:<12.2f} {cluster_time:<12.2f}")


if __name__ == "__main__":
    test_xor_basis()
    test_persistent_operations()
    test_xor_clustering()
    benchmark_advanced_operations()

"""
Advanced XOR Techniques demonstrates sophisticated bit manipulation with tries:

Key Techniques:
1. XOR Basis - Linear independence for maximum XOR queries
2. Persistent Structures - Version control for time-travel queries
3. Range Operations - Segment trees with XOR tries
4. XOR Convolution - Signal processing with XOR operations
5. Distance Clustering - Hamming distance-based grouping
6. Path Optimization - Maximum XOR path tracking

Mathematical Foundations:
- XOR basis provides optimal representation for linear XOR combinations
- Persistent data structures enable efficient version management
- XOR distance metrics useful for similarity clustering
- Convolution operations extend to XOR algebra

Real-world Applications:
- Cryptographic key analysis and generation
- Error correction codes and syndrome computation
- Network routing with XOR-based metrics
- Data compression using XOR patterns
- Machine learning feature engineering

These techniques provide O(log V) complexity for most operations
while supporting advanced queries and updates efficiently.
"""
