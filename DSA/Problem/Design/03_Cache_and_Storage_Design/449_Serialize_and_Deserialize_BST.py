"""
449. Serialize and Deserialize BST - Multiple Approaches
Difficulty: Medium

Serialization is converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary search tree. There is no restriction on how your serialization/deserialization algorithm should work. You need to ensure that a binary search tree can be serialized to a string, and this string can be deserialized to the original tree structure.

The encoded string should be as compact as possible.
"""

from typing import Optional, List
import json

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"

class CodecPreorder:
    """
    Approach 1: Preorder Traversal with Bounds
    
    Use preorder traversal and reconstruct using BST property.
    
    Time Complexity: 
    - serialize: O(n)
    - deserialize: O(n)
    
    Space Complexity: O(n)
    """
    
    def serialize(self, root: Optional[TreeNode]) -> str:
        """Serialize BST to string using preorder traversal"""
        if not root:
            return ""
        
        def preorder(node):
            if node:
                values.append(str(node.val))
                preorder(node.left)
                preorder(node.right)
        
        values = []
        preorder(root)
        return ",".join(values)
    
    def deserialize(self, data: str) -> Optional[TreeNode]:
        """Deserialize string to BST using BST property"""
        if not data:
            return None
        
        values = list(map(int, data.split(",")))
        self.index = 0
        
        def build_tree(min_val, max_val):
            if self.index >= len(values):
                return None
            
            val = values[self.index]
            if val < min_val or val > max_val:
                return None
            
            self.index += 1
            node = TreeNode(val)
            node.left = build_tree(min_val, val)
            node.right = build_tree(val, max_val)
            return node
        
        return build_tree(float('-inf'), float('inf'))

class CodecWithNulls:
    """
    Approach 2: Include Null Markers
    
    Traditional approach with explicit null markers.
    
    Time Complexity: 
    - serialize: O(n)
    - deserialize: O(n)
    
    Space Complexity: O(n)
    """
    
    def serialize(self, root: Optional[TreeNode]) -> str:
        """Serialize with null markers"""
        def preorder(node):
            if not node:
                values.append("null")
                return
            
            values.append(str(node.val))
            preorder(node.left)
            preorder(node.right)
        
        values = []
        preorder(root)
        return ",".join(values)
    
    def deserialize(self, data: str) -> Optional[TreeNode]:
        """Deserialize with null markers"""
        if not data:
            return None
        
        values = data.split(",")
        self.index = 0
        
        def build_tree():
            if self.index >= len(values):
                return None
            
            val = values[self.index]
            self.index += 1
            
            if val == "null":
                return None
            
            node = TreeNode(int(val))
            node.left = build_tree()
            node.right = build_tree()
            return node
        
        return build_tree()

class CodecMinimal:
    """
    Approach 3: Minimal Encoding
    
    Most compact representation using only values.
    
    Time Complexity: 
    - serialize: O(n)
    - deserialize: O(n log n) 
    
    Space Complexity: O(n)
    """
    
    def serialize(self, root: Optional[TreeNode]) -> str:
        """Serialize to minimal string"""
        if not root:
            return ""
        
        def inorder(node):
            if node:
                inorder(node.left)
                values.append(node.val)
                inorder(node.right)
        
        values = []
        inorder(root)
        return ",".join(map(str, values))
    
    def deserialize(self, data: str) -> Optional[TreeNode]:
        """Deserialize from sorted values"""
        if not data:
            return None
        
        values = list(map(int, data.split(",")))
        
        def build_balanced_bst(left, right):
            if left > right:
                return None
            
            mid = (left + right) // 2
            node = TreeNode(values[mid])
            node.left = build_balanced_bst(left, mid - 1)
            node.right = build_balanced_bst(mid + 1, right)
            return node
        
        return build_balanced_bst(0, len(values) - 1)

class CodecOptimized:
    """
    Approach 4: Optimized with Structure Info
    
    Include minimal structure information for exact reconstruction.
    
    Time Complexity: 
    - serialize: O(n)
    - deserialize: O(n)
    
    Space Complexity: O(n)
    """
    
    def serialize(self, root: Optional[TreeNode]) -> str:
        """Serialize with structure preservation"""
        if not root:
            return ""
        
        # Use postorder to include structure info
        def postorder(node):
            if not node:
                return
            
            postorder(node.left)
            postorder(node.right)
            values.append(node.val)
            
            # Add structure info: 0=leaf, 1=left only, 2=right only, 3=both
            structure = 0
            if node.left:
                structure += 1
            if node.right:
                structure += 2
            
            structures.append(structure)
        
        values = []
        structures = []
        postorder(root)
        
        # Combine values and structure
        result = []
        for i in range(len(values)):
            result.append(f"{values[i]}:{structures[i]}")
        
        return ",".join(result)
    
    def deserialize(self, data: str) -> Optional[TreeNode]:
        """Deserialize with structure info"""
        if not data:
            return None
        
        parts = data.split(",")
        values = []
        structures = []
        
        for part in parts:
            val_str, struct_str = part.split(":")
            values.append(int(val_str))
            structures.append(int(struct_str))
        
        self.index = len(values) - 1
        
        def build_tree():
            if self.index < 0:
                return None
            
            val = values[self.index]
            structure = structures[self.index]
            self.index -= 1
            
            node = TreeNode(val)
            
            # Build children based on structure
            if structure & 2:  # Has right child
                node.right = build_tree()
            
            if structure & 1:  # Has left child
                node.left = build_tree()
            
            return node
        
        return build_tree()

class CodecAdvanced:
    """
    Approach 5: Advanced with Multiple Formats
    
    Support multiple serialization formats and metadata.
    
    Time Complexity: 
    - serialize: O(n)
    - deserialize: O(n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, format_type="preorder"):
        self.format_type = format_type
        self.metadata = {}
    
    def serialize(self, root: Optional[TreeNode]) -> str:
        """Serialize with chosen format"""
        if not root:
            return ""
        
        # Gather metadata
        self.metadata = {
            "format": self.format_type,
            "node_count": self._count_nodes(root),
            "height": self._calculate_height(root),
            "is_balanced": self._is_balanced(root)
        }
        
        if self.format_type == "preorder":
            data = self._serialize_preorder(root)
        elif self.format_type == "level_order":
            data = self._serialize_level_order(root)
        elif self.format_type == "compact":
            data = self._serialize_compact(root)
        else:
            data = self._serialize_preorder(root)
        
        # Include metadata
        metadata_str = json.dumps(self.metadata)
        return f"{metadata_str}|{data}"
    
    def deserialize(self, data: str) -> Optional[TreeNode]:
        """Deserialize with format detection"""
        if not data:
            return None
        
        # Split metadata and data
        parts = data.split("|", 1)
        if len(parts) != 2:
            return None
        
        metadata_str, tree_data = parts
        metadata = json.loads(metadata_str)
        format_type = metadata.get("format", "preorder")
        
        if format_type == "preorder":
            return self._deserialize_preorder(tree_data)
        elif format_type == "level_order":
            return self._deserialize_level_order(tree_data)
        elif format_type == "compact":
            return self._deserialize_compact(tree_data)
        else:
            return self._deserialize_preorder(tree_data)
    
    def _count_nodes(self, root):
        if not root:
            return 0
        return 1 + self._count_nodes(root.left) + self._count_nodes(root.right)
    
    def _calculate_height(self, root):
        if not root:
            return 0
        return 1 + max(self._calculate_height(root.left), self._calculate_height(root.right))
    
    def _is_balanced(self, root):
        def check_balance(node):
            if not node:
                return True, 0
            
            left_balanced, left_height = check_balance(node.left)
            right_balanced, right_height = check_balance(node.right)
            
            balanced = (left_balanced and right_balanced and 
                       abs(left_height - right_height) <= 1)
            height = 1 + max(left_height, right_height)
            
            return balanced, height
        
        balanced, _ = check_balance(root)
        return balanced
    
    def _serialize_preorder(self, root):
        values = []
        
        def preorder(node):
            if node:
                values.append(str(node.val))
                preorder(node.left)
                preorder(node.right)
        
        preorder(root)
        return ",".join(values)
    
    def _deserialize_preorder(self, data):
        if not data:
            return None
        
        values = list(map(int, data.split(",")))
        self.index = 0
        
        def build_tree(min_val, max_val):
            if self.index >= len(values):
                return None
            
            val = values[self.index]
            if val < min_val or val > max_val:
                return None
            
            self.index += 1
            node = TreeNode(val)
            node.left = build_tree(min_val, val)
            node.right = build_tree(val, max_val)
            return node
        
        return build_tree(float('-inf'), float('inf'))
    
    def _serialize_level_order(self, root):
        if not root:
            return ""
        
        from collections import deque
        
        queue = deque([root])
        values = []
        
        while queue:
            node = queue.popleft()
            if node:
                values.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                values.append("null")
        
        # Remove trailing nulls
        while values and values[-1] == "null":
            values.pop()
        
        return ",".join(values)
    
    def _deserialize_level_order(self, data):
        if not data:
            return None
        
        from collections import deque
        
        values = data.split(",")
        if not values or values[0] == "null":
            return None
        
        root = TreeNode(int(values[0]))
        queue = deque([root])
        i = 1
        
        while queue and i < len(values):
            node = queue.popleft()
            
            # Left child
            if i < len(values) and values[i] != "null":
                node.left = TreeNode(int(values[i]))
                queue.append(node.left)
            i += 1
            
            # Right child
            if i < len(values) and values[i] != "null":
                node.right = TreeNode(int(values[i]))
                queue.append(node.right)
            i += 1
        
        return root
    
    def _serialize_compact(self, root):
        # Use preorder without bounds (most compact for BST)
        return self._serialize_preorder(root)
    
    def _deserialize_compact(self, data):
        return self._deserialize_preorder(data)
    
    def get_metadata(self):
        """Get last serialization metadata"""
        return self.metadata.copy()


def create_test_bst():
    """Create a test BST: [5, 3, 8, 2, 4, 7, 9]"""
    root = TreeNode(5)
    root.left = TreeNode(3)
    root.right = TreeNode(8)
    root.left.left = TreeNode(2)
    root.left.right = TreeNode(4)
    root.right.left = TreeNode(7)
    root.right.right = TreeNode(9)
    return root

def inorder_traversal(root):
    """Get inorder traversal for comparison"""
    result = []
    
    def inorder(node):
        if node:
            inorder(node.left)
            result.append(node.val)
            inorder(node.right)
    
    inorder(root)
    return result

def test_serialization_basic():
    """Test basic serialization functionality"""
    print("=== Testing Basic Serialization Functionality ===")
    
    implementations = [
        ("Preorder with Bounds", CodecPreorder),
        ("With Null Markers", CodecWithNulls),
        ("Minimal Encoding", CodecMinimal),
        ("Optimized Structure", CodecOptimized),
        ("Advanced Multi-format", lambda: CodecAdvanced("preorder"))
    ]
    
    original_tree = create_test_bst()
    original_inorder = inorder_traversal(original_tree)
    
    print(f"Original tree inorder: {original_inorder}")
    
    for name, codec_factory in implementations:
        print(f"\n{name}:")
        
        codec = codec_factory()
        
        # Serialize
        serialized = codec.serialize(original_tree)
        print(f"  Serialized: {serialized[:50]}{'...' if len(serialized) > 50 else ''}")
        print(f"  Length: {len(serialized)} characters")
        
        # Deserialize
        reconstructed = codec.deserialize(serialized)
        reconstructed_inorder = inorder_traversal(reconstructed)
        
        print(f"  Reconstructed inorder: {reconstructed_inorder}")
        print(f"  Correct: {original_inorder == reconstructed_inorder}")

def test_serialization_edge_cases():
    """Test serialization edge cases"""
    print("\n=== Testing Serialization Edge Cases ===")
    
    codec = CodecPreorder()
    
    # Empty tree
    print("Empty tree:")
    empty_serialized = codec.serialize(None)
    empty_reconstructed = codec.deserialize(empty_serialized)
    print(f"  Serialized: '{empty_serialized}'")
    print(f"  Reconstructed: {empty_reconstructed}")
    
    # Single node
    print(f"\nSingle node:")
    single = TreeNode(42)
    single_serialized = codec.serialize(single)
    single_reconstructed = codec.deserialize(single_serialized)
    
    print(f"  Original: {single.val}")
    print(f"  Serialized: '{single_serialized}'")
    print(f"  Reconstructed: {single_reconstructed.val if single_reconstructed else None}")
    
    # Linear tree (only left children)
    print(f"\nLinear tree (left only):")
    linear = TreeNode(5)
    linear.left = TreeNode(3)
    linear.left.left = TreeNode(1)
    
    linear_serialized = codec.serialize(linear)
    linear_reconstructed = codec.deserialize(linear_serialized)
    
    linear_original = inorder_traversal(linear)
    linear_result = inorder_traversal(linear_reconstructed)
    
    print(f"  Original inorder: {linear_original}")
    print(f"  Reconstructed inorder: {linear_result}")
    print(f"  Correct: {linear_original == linear_result}")

def test_format_comparison():
    """Compare different serialization formats"""
    print("\n=== Testing Format Comparison ===")
    
    test_tree = create_test_bst()
    
    formats = [
        ("Preorder", "preorder"),
        ("Level Order", "level_order"),
        ("Compact", "compact")
    ]
    
    for format_name, format_type in formats:
        codec = CodecAdvanced(format_type)
        
        serialized = codec.serialize(test_tree)
        metadata = codec.get_metadata()
        
        print(f"\n{format_name} Format:")
        print(f"  Serialized length: {len(serialized)} characters")
        print(f"  Node count: {metadata['node_count']}")
        print(f"  Tree height: {metadata['height']}")
        print(f"  Is balanced: {metadata['is_balanced']}")
        
        # Test reconstruction
        reconstructed = codec.deserialize(serialized)
        original_inorder = inorder_traversal(test_tree)
        reconstructed_inorder = inorder_traversal(reconstructed)
        
        print(f"  Reconstruction correct: {original_inorder == reconstructed_inorder}")

def test_performance():
    """Test serialization performance"""
    print("\n=== Testing Serialization Performance ===")
    
    import time
    
    # Create larger BST
    def create_large_bst(values):
        if not values:
            return None
        
        mid = len(values) // 2
        root = TreeNode(values[mid])
        root.left = create_large_bst(values[:mid])
        root.right = create_large_bst(values[mid+1:])
        return root
    
    large_values = list(range(1, 1001))  # 1000 nodes
    large_tree = create_large_bst(large_values)
    
    implementations = [
        ("Preorder", CodecPreorder),
        ("With Nulls", CodecWithNulls),
        ("Minimal", CodecMinimal),
        ("Optimized", CodecOptimized)
    ]
    
    for name, CodecClass in implementations:
        codec = CodecClass()
        
        # Time serialization
        start_time = time.time()
        serialized = codec.serialize(large_tree)
        serialize_time = (time.time() - start_time) * 1000
        
        # Time deserialization
        start_time = time.time()
        reconstructed = codec.deserialize(serialized)
        deserialize_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    Serialization: {serialize_time:.2f}ms")
        print(f"    Deserialization: {deserialize_time:.2f}ms")
        print(f"    Output size: {len(serialized)} characters")

def test_advanced_features():
    """Test advanced codec features"""
    print("\n=== Testing Advanced Features ===")
    
    test_tree = create_test_bst()
    codec = CodecAdvanced("preorder")
    
    # Test serialization with metadata
    serialized = codec.serialize(test_tree)
    metadata = codec.get_metadata()
    
    print("Advanced Codec Features:")
    print(f"  Metadata: {metadata}")
    
    # Test format switching
    print(f"\nTesting format switching:")
    
    formats = ["preorder", "level_order", "compact"]
    
    for fmt in formats:
        codec_fmt = CodecAdvanced(fmt)
        serialized_fmt = codec_fmt.serialize(test_tree)
        
        print(f"  {fmt}: {len(serialized_fmt)} chars")
        
        # Verify reconstruction
        reconstructed = codec_fmt.deserialize(serialized_fmt)
        original_inorder = inorder_traversal(test_tree)
        reconstructed_inorder = inorder_traversal(reconstructed)
        
        print(f"    Correct: {original_inorder == reconstructed_inorder}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Database index persistence
    print("Application 1: Database Index Persistence")
    
    # Simulate database index (BST of record IDs)
    index_tree = TreeNode(1000)
    index_tree.left = TreeNode(500)
    index_tree.right = TreeNode(1500)
    index_tree.left.left = TreeNode(250)
    index_tree.left.right = TreeNode(750)
    index_tree.right.left = TreeNode(1250)
    index_tree.right.right = TreeNode(1750)
    
    codec = CodecPreorder()
    
    # Serialize for storage
    serialized_index = codec.serialize(index_tree)
    print(f"  Index serialized: {serialized_index}")
    
    # Simulate loading from storage
    loaded_index = codec.deserialize(serialized_index)
    loaded_inorder = inorder_traversal(loaded_index)
    
    print(f"  Loaded index (sorted record IDs): {loaded_inorder}")
    
    # Application 2: Configuration tree storage
    print(f"\nApplication 2: Configuration Tree Storage")
    
    # Configuration hierarchy as BST
    config_tree = TreeNode(100)  # Main config
    config_tree.left = TreeNode(50)    # User settings
    config_tree.right = TreeNode(150)  # System settings
    config_tree.left.left = TreeNode(25)   # UI preferences
    config_tree.left.right = TreeNode(75)  # Account settings
    
    advanced_codec = CodecAdvanced("compact")
    
    # Serialize configuration
    config_serialized = advanced_codec.serialize(config_tree)
    config_metadata = advanced_codec.get_metadata()
    
    print(f"  Config serialized: {config_serialized}")
    print(f"  Config metadata: {config_metadata}")
    
    # Application 3: Caching tree structures
    print(f"\nApplication 3: Cache Serialization")
    
    # Simulate cached search tree
    cache_tree = create_test_bst()
    
    # Different codecs for different needs
    codecs = [
        ("Fast", CodecPreorder),      # Fast serialize/deserialize
        ("Compact", CodecMinimal),    # Minimal storage
        ("Robust", CodecWithNulls)    # Explicit structure
    ]
    
    for cache_type, CodecClass in codecs:
        codec = CodecClass()
        serialized = codec.serialize(cache_tree)
        
        print(f"  {cache_type} cache: {len(serialized)} bytes")

def test_compression_efficiency():
    """Test compression efficiency of different approaches"""
    print("\n=== Testing Compression Efficiency ===")
    
    # Create different tree shapes
    trees = {
        "Balanced": create_test_bst(),
        "Linear": None,
        "Full": None
    }
    
    # Create linear tree
    linear = TreeNode(1)
    current = linear
    for i in range(2, 8):
        current.right = TreeNode(i)
        current = current.right
    trees["Linear"] = linear
    
    # Create full tree
    full = TreeNode(4)
    full.left = TreeNode(2)
    full.right = TreeNode(6)
    full.left.left = TreeNode(1)
    full.left.right = TreeNode(3)
    full.right.left = TreeNode(5)
    full.right.right = TreeNode(7)
    trees["Full"] = full
    
    codecs = [
        ("Preorder", CodecPreorder),
        ("With Nulls", CodecWithNulls),
        ("Minimal", CodecMinimal),
        ("Optimized", CodecOptimized)
    ]
    
    for tree_name, tree in trees.items():
        print(f"\n{tree_name} Tree:")
        
        for codec_name, CodecClass in codecs:
            codec = CodecClass()
            serialized = codec.serialize(tree)
            
            print(f"  {codec_name}: {len(serialized)} characters")

def test_error_handling():
    """Test error handling in serialization"""
    print("\n=== Testing Error Handling ===")
    
    codec = CodecPreorder()
    
    # Test invalid input
    test_cases = [
        ("Empty string", ""),
        ("Invalid format", "invalid,data,format"),
        ("Single value", "42"),
        ("Mixed types", "1,hello,3")
    ]
    
    for test_name, test_data in test_cases:
        print(f"\n{test_name}: '{test_data}'")
        
        try:
            result = codec.deserialize(test_data)
            if result:
                inorder_result = inorder_traversal(result)
                print(f"  Successfully parsed: {inorder_result}")
            else:
                print(f"  Result: None (empty tree)")
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")

def benchmark_memory_usage():
    """Benchmark memory usage of different approaches"""
    print("\n=== Benchmarking Memory Usage ===")
    
    test_tree = create_test_bst()
    
    implementations = [
        ("Preorder", CodecPreorder),
        ("With Nulls", CodecWithNulls),
        ("Minimal", CodecMinimal),
        ("Advanced", lambda: CodecAdvanced("preorder"))
    ]
    
    for name, codec_factory in implementations:
        codec = codec_factory()
        
        # Serialize
        serialized = codec.serialize(test_tree)
        
        # Estimate memory usage (simplified)
        string_memory = len(serialized)
        
        if hasattr(codec, 'metadata'):
            metadata_memory = len(str(codec.metadata))
        else:
            metadata_memory = 0
        
        total_memory = string_memory + metadata_memory
        
        print(f"  {name}:")
        print(f"    Serialized size: {string_memory} chars")
        print(f"    Metadata size: {metadata_memory} chars")
        print(f"    Total: {total_memory} chars")

def stress_test_serialization():
    """Stress test serialization with large trees"""
    print("\n=== Stress Testing Serialization ===")
    
    import time
    
    # Create increasingly large BSTs
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\nTesting with {size} nodes:")
        
        # Create balanced BST
        values = list(range(1, size + 1))
        
        def build_balanced(vals):
            if not vals:
                return None
            mid = len(vals) // 2
            root = TreeNode(vals[mid])
            root.left = build_balanced(vals[:mid])
            root.right = build_balanced(vals[mid+1:])
            return root
        
        large_tree = build_balanced(values)
        
        # Test with fastest codec
        codec = CodecPreorder()
        
        start_time = time.time()
        serialized = codec.serialize(large_tree)
        serialize_time = time.time() - start_time
        
        start_time = time.time()
        reconstructed = codec.deserialize(serialized)
        deserialize_time = time.time() - start_time
        
        # Verify correctness
        original_inorder = inorder_traversal(large_tree)
        reconstructed_inorder = inorder_traversal(reconstructed)
        correct = original_inorder == reconstructed_inorder
        
        print(f"  Serialize: {serialize_time:.3f}s")
        print(f"  Deserialize: {deserialize_time:.3f}s") 
        print(f"  Output size: {len(serialized):,} characters")
        print(f"  Correct: {correct}")

if __name__ == "__main__":
    test_serialization_basic()
    test_serialization_edge_cases()
    test_format_comparison()
    test_performance()
    test_advanced_features()
    demonstrate_applications()
    test_compression_efficiency()
    test_error_handling()
    benchmark_memory_usage()
    stress_test_serialization()

"""
BST Serialization Design demonstrates key concepts:

Core Approaches:
1. Preorder with Bounds - Use BST property for compact encoding
2. With Null Markers - Traditional approach with explicit structure
3. Minimal Encoding - Inorder only, reconstruct as balanced BST
4. Optimized Structure - Include minimal structure information
5. Advanced Multi-format - Support multiple formats with metadata

Key Design Principles:
- BST property enables compact serialization without nulls
- Preorder traversal with min/max bounds for reconstruction
- Trade-offs between compactness and reconstruction complexity
- Format flexibility for different use cases

Serialization Strategies:
- Preorder only: Most compact for BSTs, O(n) reconstruction
- With nulls: Larger but preserves exact structure
- Inorder only: Minimal but loses original structure
- Level order: Good for breadth-first reconstruction

Performance Characteristics:
- Preorder: O(n) serialize, O(n) deserialize, minimal space
- With nulls: O(n) serialize, O(n) deserialize, larger space
- Minimal: O(n) serialize, O(n log n) deserialize, smallest space
- Advanced: O(n) serialize, O(n) deserialize, flexible formats

Real-world Applications:
- Database index persistence and recovery
- Configuration tree storage and loading
- Cache serialization for distributed systems
- Tree structure transmission over networks
- Backup and restore of search structures
- Inter-process communication with tree data

The preorder with bounds approach is most commonly used
for BST serialization due to its optimal balance of
compactness and reconstruction efficiency.
"""
