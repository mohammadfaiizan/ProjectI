"""
173. Binary Search Tree Iterator - Multiple Approaches
Difficulty: Medium

Implement the BSTIterator class that represents an iterator over the in-order traversal of a binary search tree (BST):

- BSTIterator(TreeNode root) Initializes an object of the BSTIterator class. The root of the BST is given as part of the constructor.
- boolean hasNext() Returns true if there exists a next element in the traversal.
- int next() Returns the next element in the traversal.

Note that by definition, the in-order traversal of a BST is in ascending order.

Follow up: Could you implement next() and hasNext() in average O(1) time? Try to use O(h) memory, where h is the height of the tree.
"""

from typing import List, Optional, Generator

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BSTIteratorPrecompute:
    """
    Approach 1: Precompute All Values
    
    Perform complete in-order traversal during initialization.
    
    Time Complexity: 
    - Constructor: O(n)
    - hasNext(), next(): O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, root: Optional[TreeNode]):
        self.values = []
        self.index = 0
        self._inorder(root)
    
    def _inorder(self, node: Optional[TreeNode]) -> None:
        if node:
            self._inorder(node.left)
            self.values.append(node.val)
            self._inorder(node.right)
    
    def next(self) -> int:
        if self.hasNext():
            val = self.values[self.index]
            self.index += 1
            return val
        return -1
    
    def hasNext(self) -> bool:
        return self.index < len(self.values)

class BSTIteratorStack:
    """
    Approach 2: Stack-based Controlled Recursion
    
    Use stack to simulate recursive in-order traversal.
    
    Time Complexity: 
    - Constructor: O(h) where h is height
    - hasNext(): O(1)
    - next(): O(1) amortized
    
    Space Complexity: O(h)
    """
    
    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        self._push_left(root)
    
    def _push_left(self, node: Optional[TreeNode]) -> None:
        """Push all left children onto stack"""
        while node:
            self.stack.append(node)
            node = node.left
    
    def next(self) -> int:
        if not self.hasNext():
            return -1
        
        # Pop the next smallest element
        node = self.stack.pop()
        
        # If it has right child, push all left children of right subtree
        if node.right:
            self._push_left(node.right)
        
        return node.val
    
    def hasNext(self) -> bool:
        return len(self.stack) > 0

class BSTIteratorGenerator:
    """
    Approach 3: Python Generator
    
    Use generator for lazy evaluation of in-order traversal.
    
    Time Complexity: 
    - Constructor: O(1)
    - hasNext(): O(1)
    - next(): O(1) amortized
    
    Space Complexity: O(h)
    """
    
    def __init__(self, root: Optional[TreeNode]):
        self.generator = self._inorder_generator(root)
        self.current_value = None
        self.has_next_value = True
        self._advance()
    
    def _inorder_generator(self, node: Optional[TreeNode]) -> Generator[int, None, None]:
        if node:
            yield from self._inorder_generator(node.left)
            yield node.val
            yield from self._inorder_generator(node.right)
    
    def _advance(self) -> None:
        try:
            self.current_value = next(self.generator)
        except StopIteration:
            self.has_next_value = False
    
    def next(self) -> int:
        if not self.hasNext():
            return -1
        
        val = self.current_value
        self._advance()
        return val
    
    def hasNext(self) -> bool:
        return self.has_next_value

class BSTIteratorMorris:
    """
    Approach 4: Morris Traversal (Space-Optimized)
    
    Use Morris in-order traversal for O(1) space complexity.
    
    Time Complexity: 
    - Constructor: O(1)
    - hasNext(): O(1)
    - next(): O(1) amortized
    
    Space Complexity: O(1)
    """
    
    def __init__(self, root: Optional[TreeNode]):
        self.current = root
        self.next_val = None
        self._find_next()
    
    def _find_next(self) -> None:
        """Find next value using Morris traversal"""
        while self.current:
            if not self.current.left:
                # No left child, current is next
                self.next_val = self.current.val
                self.current = self.current.right
                return
            else:
                # Find predecessor
                predecessor = self.current.left
                while predecessor.right and predecessor.right != self.current:
                    predecessor = predecessor.right
                
                if not predecessor.right:
                    # Make threading
                    predecessor.right = self.current
                    self.current = self.current.left
                else:
                    # Remove threading
                    predecessor.right = None
                    self.next_val = self.current.val
                    self.current = self.current.right
                    return
        
        self.next_val = None
    
    def next(self) -> int:
        if not self.hasNext():
            return -1
        
        val = self.next_val
        self._find_next()
        return val
    
    def hasNext(self) -> bool:
        return self.next_val is not None

class BSTIteratorWithPeek:
    """
    Approach 5: Enhanced Iterator with Peek
    
    Stack-based iterator with additional peek functionality.
    
    Time Complexity: 
    - Constructor: O(h)
    - hasNext(), peek(): O(1)
    - next(): O(1) amortized
    
    Space Complexity: O(h)
    """
    
    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        self._push_left(root)
    
    def _push_left(self, node: Optional[TreeNode]) -> None:
        while node:
            self.stack.append(node)
            node = node.left
    
    def next(self) -> int:
        if not self.hasNext():
            return -1
        
        node = self.stack.pop()
        
        if node.right:
            self._push_left(node.right)
        
        return node.val
    
    def peek(self) -> int:
        """Peek at next value without consuming it"""
        if not self.hasNext():
            return -1
        return self.stack[-1].val
    
    def hasNext(self) -> bool:
        return len(self.stack) > 0
    
    def skip(self, count: int) -> None:
        """Skip the next 'count' elements"""
        for _ in range(count):
            if self.hasNext():
                self.next()
    
    def remaining_count(self) -> int:
        """Get approximate count of remaining elements"""
        # This is an approximation based on stack size
        return len(self.stack)


def create_test_bst() -> TreeNode:
    """Create a test BST: [7, 3, 15, null, null, 9, 20]"""
    root = TreeNode(7)
    root.left = TreeNode(3)
    root.right = TreeNode(15)
    root.right.left = TreeNode(9)
    root.right.right = TreeNode(20)
    return root

def create_large_bst() -> TreeNode:
    """Create a larger BST for testing"""
    #       10
    #      /  \
    #     5    15
    #    / \   / \
    #   3   7 12  18
    #  /   /     /
    # 1   6     17
    
    root = TreeNode(10)
    root.left = TreeNode(5)
    root.right = TreeNode(15)
    root.left.left = TreeNode(3)
    root.left.right = TreeNode(7)
    root.right.left = TreeNode(12)
    root.right.right = TreeNode(18)
    root.left.left.left = TreeNode(1)
    root.left.right.left = TreeNode(6)
    root.right.right.left = TreeNode(17)
    
    return root

def test_bst_iterator_basic():
    """Test basic BST iterator functionality"""
    print("=== Testing Basic BST Iterator Functionality ===")
    
    implementations = [
        ("Precompute All", BSTIteratorPrecompute),
        ("Stack-based", BSTIteratorStack),
        ("Generator", BSTIteratorGenerator),
        ("Morris Traversal", BSTIteratorMorris),
        ("With Peek", BSTIteratorWithPeek)
    ]
    
    root = create_test_bst()
    
    for name, IteratorClass in implementations:
        print(f"\n{name}:")
        
        iterator = IteratorClass(root)
        result = []
        
        while iterator.hasNext():
            val = iterator.next()
            result.append(val)
        
        print(f"  In-order traversal: {result}")
        print(f"  hasNext() after complete: {iterator.hasNext()}")

def test_bst_iterator_edge_cases():
    """Test BST iterator edge cases"""
    print("\n=== Testing BST Iterator Edge Cases ===")
    
    # Test with empty tree
    print("Empty tree:")
    iterator = BSTIteratorStack(None)
    print(f"  hasNext(): {iterator.hasNext()}")
    print(f"  next(): {iterator.next()}")
    
    # Test with single node
    print(f"\nSingle node tree:")
    single_node = TreeNode(42)
    iterator = BSTIteratorStack(single_node)
    print(f"  hasNext(): {iterator.hasNext()}")
    print(f"  next(): {iterator.next()}")
    print(f"  hasNext(): {iterator.hasNext()}")
    
    # Test with linear tree (all left children)
    print(f"\nLinear tree (left children only):")
    #     5
    #    /
    #   3
    #  /
    # 1
    linear_root = TreeNode(5)
    linear_root.left = TreeNode(3)
    linear_root.left.left = TreeNode(1)
    
    iterator = BSTIteratorStack(linear_root)
    linear_result = []
    while iterator.hasNext():
        linear_result.append(iterator.next())
    
    print(f"  Traversal: {linear_result}")

def test_iterator_performance():
    """Test iterator performance characteristics"""
    print("\n=== Testing Iterator Performance ===")
    
    import time
    
    implementations = [
        ("Precompute All", BSTIteratorPrecompute),
        ("Stack-based", BSTIteratorStack),
        ("Generator", BSTIteratorGenerator)
    ]
    
    root = create_large_bst()
    
    for name, IteratorClass in implementations:
        # Test initialization time
        start_time = time.time()
        iterator = IteratorClass(root)
        init_time = (time.time() - start_time) * 1000
        
        # Test iteration time
        start_time = time.time()
        count = 0
        while iterator.hasNext():
            iterator.next()
            count += 1
        iteration_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    Initialization: {init_time:.3f}ms")
        print(f"    Iteration ({count} elements): {iteration_time:.3f}ms")

def test_enhanced_features():
    """Test enhanced iterator features"""
    print("\n=== Testing Enhanced Features ===")
    
    root = create_test_bst()
    iterator = BSTIteratorWithPeek(root)
    
    # Test peek functionality
    print("Testing peek functionality:")
    while iterator.hasNext():
        peeked = iterator.peek()
        next_val = iterator.next()
        print(f"  peek(): {peeked}, next(): {next_val}")
        
        if peeked != next_val:
            print("  ERROR: Peek and next don't match!")
    
    # Test skip functionality
    print(f"\nTesting skip functionality:")
    iterator2 = BSTIteratorWithPeek(create_test_bst())
    
    print(f"  First element: {iterator2.next()}")
    print(f"  Skipping 2 elements...")
    iterator2.skip(2)
    print(f"  Next element after skip: {iterator2.next()}")

def test_memory_usage():
    """Test memory usage of different approaches"""
    print("\n=== Testing Memory Usage Analysis ===")
    
    # Create trees of different heights
    def create_balanced_tree(height: int) -> TreeNode:
        if height <= 0:
            return None
        
        root = TreeNode(height)
        root.left = create_balanced_tree(height - 1)
        root.right = create_balanced_tree(height - 1)
        return root
    
    heights = [3, 5, 7]
    
    for height in heights:
        print(f"\nTree height {height}:")
        root = create_balanced_tree(height)
        
        # Stack-based iterator
        iterator_stack = BSTIteratorStack(root)
        stack_size = len(iterator_stack.stack)
        
        # Precompute iterator (count all nodes)
        iterator_precompute = BSTIteratorPrecompute(root)
        precompute_size = len(iterator_precompute.values)
        
        print(f"  Stack-based memory: {stack_size} nodes")
        print(f"  Precompute memory: {precompute_size} nodes")
        print(f"  Memory ratio (stack/precompute): {stack_size/precompute_size:.3f}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Database index scanning
    print("Application 1: Database Index Scanning")
    root = create_large_bst()
    db_scanner = BSTIteratorStack(root)
    
    print("  Simulating database range scan:")
    scan_count = 0
    while db_scanner.hasNext() and scan_count < 5:
        record_id = db_scanner.next()
        print(f"    Retrieved record ID: {record_id}")
        scan_count += 1
    
    print(f"    Remaining records: {'available' if db_scanner.hasNext() else 'none'}")
    
    # Application 2: Sorted data streaming
    print(f"\nApplication 2: Sorted Data Streaming")
    stream_iterator = BSTIteratorWithPeek(create_test_bst())
    
    # Simulate streaming with batching
    batch_size = 3
    batch_count = 0
    
    while stream_iterator.hasNext():
        batch_count += 1
        batch = []
        
        for _ in range(batch_size):
            if stream_iterator.hasNext():
                batch.append(stream_iterator.next())
        
        print(f"    Batch {batch_count}: {batch}")
    
    # Application 3: Merge operation with another sorted sequence
    print(f"\nApplication 3: Merge with External Sorted Data")
    tree_iterator = BSTIteratorStack(create_test_bst())
    external_data = [2, 8, 12, 25]  # Another sorted sequence
    
    merged_result = []
    ext_index = 0
    
    while tree_iterator.hasNext() and ext_index < len(external_data):
        tree_val = tree_iterator.peek()
        ext_val = external_data[ext_index]
        
        if tree_val <= ext_val:
            merged_result.append(tree_iterator.next())
        else:
            merged_result.append(ext_val)
            ext_index += 1
    
    # Add remaining elements
    while tree_iterator.hasNext():
        merged_result.append(tree_iterator.next())
    
    while ext_index < len(external_data):
        merged_result.append(external_data[ext_index])
        ext_index += 1
    
    print(f"    Merged result: {merged_result}")

def test_concurrent_iterations():
    """Test multiple concurrent iterations"""
    print("\n=== Testing Concurrent Iterations ===")
    
    root = create_test_bst()
    
    # Create multiple iterators on same tree
    iter1 = BSTIteratorStack(root)
    iter2 = BSTIteratorStack(root)
    
    print("Concurrent iteration with two iterators:")
    step = 0
    while iter1.hasNext() or iter2.hasNext():
        step += 1
        
        val1 = iter1.next() if iter1.hasNext() else "DONE"
        val2 = iter2.next() if iter2.hasNext() else "DONE"
        
        print(f"  Step {step}: Iterator1={val1}, Iterator2={val2}")

def test_large_tree_performance():
    """Test performance on large trees"""
    print("\n=== Testing Large Tree Performance ===")
    
    def create_large_linear_tree(size: int) -> TreeNode:
        """Create a large linear tree (worst case for height)"""
        if size <= 0:
            return None
        
        root = TreeNode(size)
        current = root
        
        for i in range(size - 1, 0, -1):
            current.left = TreeNode(i)
            current = current.left
        
        return root
    
    import time
    
    tree_sizes = [100, 500, 1000]
    
    for size in tree_sizes:
        print(f"\nLinear tree with {size} nodes:")
        root = create_large_linear_tree(size)
        
        # Test stack-based iterator
        start_time = time.time()
        iterator = BSTIteratorStack(root)
        init_time = (time.time() - start_time) * 1000
        
        start_time = time.time()
        count = 0
        while iterator.hasNext():
            iterator.next()
            count += 1
        traverse_time = (time.time() - start_time) * 1000
        
        print(f"  Stack iterator - Init: {init_time:.2f}ms, Traverse: {traverse_time:.2f}ms")
        
        # Test precompute iterator
        start_time = time.time()
        iterator2 = BSTIteratorPrecompute(root)
        init_time2 = (time.time() - start_time) * 1000
        
        start_time = time.time()
        while iterator2.hasNext():
            iterator2.next()
        traverse_time2 = (time.time() - start_time) * 1000
        
        print(f"  Precompute iterator - Init: {init_time2:.2f}ms, Traverse: {traverse_time2:.2f}ms")

def benchmark_hasNext_performance():
    """Benchmark hasNext() call performance"""
    print("\n=== Benchmarking hasNext() Performance ===")
    
    import time
    
    root = create_large_bst()
    implementations = [
        ("Stack-based", BSTIteratorStack),
        ("Precompute", BSTIteratorPrecompute),
        ("Generator", BSTIteratorGenerator)
    ]
    
    for name, IteratorClass in implementations:
        iterator = IteratorClass(root)
        
        # Benchmark hasNext() calls
        start_time = time.time()
        hasNext_calls = 0
        
        while iterator.hasNext():
            hasNext_calls += 1
            iterator.next()
            
            # Extra hasNext() calls to benchmark
            for _ in range(10):
                iterator.hasNext()
                hasNext_calls += 10
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {name}: {hasNext_calls} hasNext() calls in {elapsed:.2f}ms")

if __name__ == "__main__":
    test_bst_iterator_basic()
    test_bst_iterator_edge_cases()
    test_iterator_performance()
    test_enhanced_features()
    test_memory_usage()
    demonstrate_applications()
    test_concurrent_iterations()
    test_large_tree_performance()
    benchmark_hasNext_performance()

"""
BST Iterator Design demonstrates key concepts:

Core Approaches:
1. Precompute All - O(n) space, O(1) operations
2. Stack-based - O(h) space, O(1) amortized operations
3. Generator - Python-specific lazy evaluation approach
4. Morris Traversal - O(1) space, O(1) amortized operations
5. Enhanced - Additional features like peek and skip

Key Design Principles:
- Space-time tradeoffs in tree traversal
- Lazy evaluation vs eager computation
- Iterator pattern implementation
- Memory-efficient tree traversal

Performance Characteristics:
- Stack-based: Optimal balance of space and time
- Precompute: Fast operations but high memory usage
- Morris: Space-optimal but complex implementation
- Generator: Clean code with good performance

Real-world Applications:
- Database index scanning and range queries
- Sorted data streaming and processing
- Merge operations with multiple sorted sources
- Large dataset processing with memory constraints
- Tree-based data structure enumeration

The stack-based approach is most commonly used
due to its optimal O(h) space complexity and 
O(1) amortized time complexity for operations.
"""
