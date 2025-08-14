"""
Segment Tree - Advanced Range Query and Update Data Structure
This module implements various segment tree variants with lazy propagation and applications.
"""

from typing import List, Callable, Any, Optional, Tuple
import math

class SegmentTree:
    """Basic segment tree for range queries and point updates"""
    
    def __init__(self, arr: List[int], operation: str = "sum"):
        """
        Initialize segment tree
        
        Args:
            arr: Initial array
            operation: "sum", "min", "max", "gcd"
        """
        self.n = len(arr)
        self.operation = operation
        self.tree = [0] * (4 * self.n)
        
        # Set operation function and identity
        self._set_operation(operation)
        
        # Build tree
        self._build(arr, 1, 0, self.n - 1)
    
    def _set_operation(self, operation: str):
        """Set operation function and identity value"""
        if operation == "sum":
            self.op = lambda a, b: a + b
            self.identity = 0
        elif operation == "min":
            self.op = min
            self.identity = float('inf')
        elif operation == "max":
            self.op = max
            self.identity = float('-inf')
        elif operation == "gcd":
            self.op = math.gcd
            self.identity = 0
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _build(self, arr: List[int], node: int, start: int, end: int):
        """Build segment tree recursively"""
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = self.op(self.tree[2 * node], self.tree[2 * node + 1])
    
    def update_point(self, idx: int, val: int):
        """
        Update single point
        
        Time Complexity: O(log n)
        
        Args:
            idx: Index to update (0-indexed)
            val: New value
        """
        self._update_point(1, 0, self.n - 1, idx, val)
    
    def _update_point(self, node: int, start: int, end: int, idx: int, val: int):
        """Recursive point update"""
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._update_point(2 * node, start, mid, idx, val)
            else:
                self._update_point(2 * node + 1, mid + 1, end, idx, val)
            self.tree[node] = self.op(self.tree[2 * node], self.tree[2 * node + 1])
    
    def query_range(self, left: int, right: int) -> int:
        """
        Query range [left, right]
        
        Time Complexity: O(log n)
        
        Args:
            left, right: Range boundaries (0-indexed, inclusive)
        
        Returns:
            Result of operation on range
        """
        return self._query_range(1, 0, self.n - 1, left, right)
    
    def _query_range(self, node: int, start: int, end: int, left: int, right: int):
        """Recursive range query"""
        if right < start or end < left:
            return self.identity
        
        if left <= start and end <= right:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_result = self._query_range(2 * node, start, mid, left, right)
        right_result = self._query_range(2 * node + 1, mid + 1, end, left, right)
        
        return self.op(left_result, right_result)

class LazySegmentTree:
    """Segment tree with lazy propagation for range updates"""
    
    def __init__(self, arr: List[int], operation: str = "sum"):
        """
        Initialize lazy segment tree
        
        Args:
            arr: Initial array
            operation: "sum", "min", "max"
        """
        self.n = len(arr)
        self.operation = operation
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        
        self._set_operation(operation)
        self._build(arr, 1, 0, self.n - 1)
    
    def _set_operation(self, operation: str):
        """Set operation function and identity value"""
        if operation == "sum":
            self.op = lambda a, b: a + b
            self.identity = 0
            self.lazy_apply = lambda val, lazy, length: val + lazy * length
            self.lazy_combine = lambda a, b: a + b
        elif operation == "min":
            self.op = min
            self.identity = float('inf')
            self.lazy_apply = lambda val, lazy, length: val + lazy
            self.lazy_combine = lambda a, b: a + b
        elif operation == "max":
            self.op = max
            self.identity = float('-inf')
            self.lazy_apply = lambda val, lazy, length: val + lazy
            self.lazy_combine = lambda a, b: a + b
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _build(self, arr: List[int], node: int, start: int, end: int):
        """Build segment tree recursively"""
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = self.op(self.tree[2 * node], self.tree[2 * node + 1])
    
    def _push(self, node: int, start: int, end: int):
        """Push lazy value down"""
        if self.lazy[node] != 0:
            length = end - start + 1
            self.tree[node] = self.lazy_apply(self.tree[node], self.lazy[node], length)
            
            if start != end:  # Not a leaf
                self.lazy[2 * node] = self.lazy_combine(self.lazy[2 * node], self.lazy[node])
                self.lazy[2 * node + 1] = self.lazy_combine(self.lazy[2 * node + 1], self.lazy[node])
            
            self.lazy[node] = 0
    
    def update_range(self, left: int, right: int, val: int):
        """
        Update range [left, right] with value
        
        Time Complexity: O(log n)
        
        Args:
            left, right: Range boundaries (0-indexed, inclusive)
            val: Value to add to range
        """
        self._update_range(1, 0, self.n - 1, left, right, val)
    
    def _update_range(self, node: int, start: int, end: int, left: int, right: int, val: int):
        """Recursive range update with lazy propagation"""
        self._push(node, start, end)
        
        if start > right or end < left:
            return
        
        if start >= left and end <= right:
            self.lazy[node] = self.lazy_combine(self.lazy[node], val)
            self._push(node, start, end)
            return
        
        mid = (start + end) // 2
        self._update_range(2 * node, start, mid, left, right, val)
        self._update_range(2 * node + 1, mid + 1, end, left, right, val)
        
        self._push(2 * node, start, mid)
        self._push(2 * node + 1, mid + 1, end)
        
        self.tree[node] = self.op(self.tree[2 * node], self.tree[2 * node + 1])
    
    def query_range(self, left: int, right: int):
        """
        Query range [left, right]
        
        Time Complexity: O(log n)
        
        Args:
            left, right: Range boundaries (0-indexed, inclusive)
        
        Returns:
            Result of operation on range
        """
        return self._query_range(1, 0, self.n - 1, left, right)
    
    def _query_range(self, node: int, start: int, end: int, left: int, right: int):
        """Recursive range query with lazy propagation"""
        if start > right or end < left:
            return self.identity
        
        self._push(node, start, end)
        
        if start >= left and end <= right:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_result = self._query_range(2 * node, start, mid, left, right)
        right_result = self._query_range(2 * node + 1, mid + 1, end, left, right)
        
        return self.op(left_result, right_result)

class SegmentTreeRangeSet:
    """Segment tree with range set operations"""
    
    def __init__(self, arr: List[int]):
        """Initialize segment tree for range set operations"""
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy = [None] * (4 * self.n)
        
        self._build(arr, 1, 0, self.n - 1)
    
    def _build(self, arr: List[int], node: int, start: int, end: int):
        """Build segment tree"""
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
    
    def _push(self, node: int, start: int, end: int):
        """Push lazy value down"""
        if self.lazy[node] is not None:
            length = end - start + 1
            self.tree[node] = self.lazy[node] * length
            
            if start != end:  # Not a leaf
                self.lazy[2 * node] = self.lazy[node]
                self.lazy[2 * node + 1] = self.lazy[node]
            
            self.lazy[node] = None
    
    def set_range(self, left: int, right: int, val: int):
        """
        Set range [left, right] to value
        
        Args:
            left, right: Range boundaries (0-indexed, inclusive)
            val: Value to set range to
        """
        self._set_range(1, 0, self.n - 1, left, right, val)
    
    def _set_range(self, node: int, start: int, end: int, left: int, right: int, val: int):
        """Recursive range set with lazy propagation"""
        self._push(node, start, end)
        
        if start > right or end < left:
            return
        
        if start >= left and end <= right:
            self.lazy[node] = val
            self._push(node, start, end)
            return
        
        mid = (start + end) // 2
        self._set_range(2 * node, start, mid, left, right, val)
        self._set_range(2 * node + 1, mid + 1, end, left, right, val)
        
        self._push(2 * node, start, mid)
        self._push(2 * node + 1, mid + 1, end)
        
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
    
    def query_range(self, left: int, right: int) -> int:
        """Query sum of range [left, right]"""
        return self._query_range(1, 0, self.n - 1, left, right)
    
    def _query_range(self, node: int, start: int, end: int, left: int, right: int):
        """Recursive range query"""
        if start > right or end < left:
            return 0
        
        self._push(node, start, end)
        
        if start >= left and end <= right:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_result = self._query_range(2 * node, start, mid, left, right)
        right_result = self._query_range(2 * node + 1, mid + 1, end, left, right)
        
        return left_result + right_result

class PersistentSegmentTree:
    """Persistent segment tree for historical queries"""
    
    def __init__(self, arr: List[int]):
        """Initialize persistent segment tree"""
        self.n = len(arr)
        self.versions = []
        
        # Build initial version
        initial_root = self._build(arr, 0, self.n - 1)
        self.versions.append(initial_root)
    
    def _build(self, arr: List[int], start: int, end: int):
        """Build segment tree and return root node"""
        if start == end:
            return {'val': arr[start], 'left': None, 'right': None}
        
        mid = (start + end) // 2
        left_child = self._build(arr, start, mid)
        right_child = self._build(arr, mid + 1, end)
        
        return {
            'val': left_child['val'] + right_child['val'],
            'left': left_child,
            'right': right_child
        }
    
    def update(self, version: int, idx: int, val: int):
        """
        Create new version with updated value
        
        Args:
            version: Version to update from
            idx: Index to update
            val: New value
        
        Returns:
            int: New version number
        """
        if version >= len(self.versions):
            raise ValueError("Invalid version")
        
        old_root = self.versions[version]
        new_root = self._update(old_root, 0, self.n - 1, idx, val)
        self.versions.append(new_root)
        
        return len(self.versions) - 1
    
    def _update(self, node, start: int, end: int, idx: int, val: int):
        """Create new node with updated value"""
        if start == end:
            return {'val': val, 'left': None, 'right': None}
        
        mid = (start + end) // 2
        
        if idx <= mid:
            new_left = self._update(node['left'], start, mid, idx, val)
            new_right = node['right']  # Reuse old right subtree
        else:
            new_left = node['left']   # Reuse old left subtree
            new_right = self._update(node['right'], mid + 1, end, idx, val)
        
        return {
            'val': new_left['val'] + new_right['val'],
            'left': new_left,
            'right': new_right
        }
    
    def query(self, version: int, left: int, right: int) -> int:
        """
        Query range in specific version
        
        Args:
            version: Version to query
            left, right: Range boundaries
        
        Returns:
            Sum of range in given version
        """
        if version >= len(self.versions):
            raise ValueError("Invalid version")
        
        root = self.versions[version]
        return self._query(root, 0, self.n - 1, left, right)
    
    def _query(self, node, start: int, end: int, left: int, right: int):
        """Query range in given node"""
        if not node or start > right or end < left:
            return 0
        
        if start >= left and end <= right:
            return node['val']
        
        mid = (start + end) // 2
        left_result = self._query(node['left'], start, mid, left, right)
        right_result = self._query(node['right'], mid + 1, end, left, right)
        
        return left_result + right_result

class SegmentTree2D:
    """2D Segment Tree for 2D range queries"""
    
    def __init__(self, matrix: List[List[int]]):
        """Initialize 2D segment tree"""
        self.rows = len(matrix)
        self.cols = len(matrix[0]) if matrix else 0
        self.tree = {}
        
        if self.rows > 0 and self.cols > 0:
            self._build_x(matrix, 1, 0, self.rows - 1)
    
    def _build_x(self, matrix: List[List[int]], node_x: int, start_x: int, end_x: int):
        """Build segment tree for x dimension"""
        if start_x == end_x:
            self._build_y(matrix[start_x], node_x, 1, 0, self.cols - 1)
        else:
            mid_x = (start_x + end_x) // 2
            self._build_x(matrix, 2 * node_x, start_x, mid_x)
            self._build_x(matrix, 2 * node_x + 1, mid_x + 1, end_x)
            self._merge_y(node_x, 2 * node_x, 2 * node_x + 1, 1, 0, self.cols - 1)
    
    def _build_y(self, arr: List[int], node_x: int, node_y: int, start_y: int, end_y: int):
        """Build segment tree for y dimension"""
        if (node_x, node_y) not in self.tree:
            self.tree[(node_x, node_y)] = 0
        
        if start_y == end_y:
            self.tree[(node_x, node_y)] = arr[start_y]
        else:
            mid_y = (start_y + end_y) // 2
            self._build_y(arr, node_x, 2 * node_y, start_y, mid_y)
            self._build_y(arr, node_x, 2 * node_y + 1, mid_y + 1, end_y)
            self.tree[(node_x, node_y)] = (
                self.tree.get((node_x, 2 * node_y), 0) + 
                self.tree.get((node_x, 2 * node_y + 1), 0)
            )
    
    def _merge_y(self, node_x: int, left_x: int, right_x: int, node_y: int, start_y: int, end_y: int):
        """Merge y segments from left and right x children"""
        if (node_x, node_y) not in self.tree:
            self.tree[(node_x, node_y)] = 0
        
        if start_y == end_y:
            self.tree[(node_x, node_y)] = (
                self.tree.get((left_x, node_y), 0) + 
                self.tree.get((right_x, node_y), 0)
            )
        else:
            mid_y = (start_y + end_y) // 2
            self._merge_y(node_x, left_x, right_x, 2 * node_y, start_y, mid_y)
            self._merge_y(node_x, left_x, right_x, 2 * node_y + 1, mid_y + 1, end_y)
            self.tree[(node_x, node_y)] = (
                self.tree.get((node_x, 2 * node_y), 0) + 
                self.tree.get((node_x, 2 * node_y + 1), 0)
            )
    
    def query(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """Query 2D range sum"""
        return self._query_x(1, 0, self.rows - 1, x1, x2, y1, y2)
    
    def _query_x(self, node_x: int, start_x: int, end_x: int, x1: int, x2: int, y1: int, y2: int):
        """Query x dimension"""
        if start_x > x2 or end_x < x1:
            return 0
        
        if start_x >= x1 and end_x <= x2:
            return self._query_y(node_x, 1, 0, self.cols - 1, y1, y2)
        
        mid_x = (start_x + end_x) // 2
        left_result = self._query_x(2 * node_x, start_x, mid_x, x1, x2, y1, y2)
        right_result = self._query_x(2 * node_x + 1, mid_x + 1, end_x, x1, x2, y1, y2)
        
        return left_result + right_result
    
    def _query_y(self, node_x: int, node_y: int, start_y: int, end_y: int, y1: int, y2: int):
        """Query y dimension"""
        if start_y > y2 or end_y < y1:
            return 0
        
        if start_y >= y1 and end_y <= y2:
            return self.tree.get((node_x, node_y), 0)
        
        mid_y = (start_y + end_y) // 2
        left_result = self._query_y(node_x, 2 * node_y, start_y, mid_y, y1, y2)
        right_result = self._query_y(node_x, 2 * node_y + 1, mid_y + 1, end_y, y1, y2)
        
        return left_result + right_result

# ==================== ADVANCED APPLICATIONS ====================

class RangeModularTree:
    """Segment tree for range modular arithmetic"""
    
    def __init__(self, arr: List[int], mod: int):
        """Initialize with modular operations"""
        self.n = len(arr)
        self.mod = mod
        self.tree = [0] * (4 * self.n)
        self._build(arr, 1, 0, self.n - 1)
    
    def _build(self, arr: List[int], node: int, start: int, end: int):
        """Build tree with modular operations"""
        if start == end:
            self.tree[node] = arr[start] % self.mod
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = (self.tree[2 * node] + self.tree[2 * node + 1]) % self.mod
    
    def update(self, idx: int, val: int):
        """Update with modular arithmetic"""
        self._update(1, 0, self.n - 1, idx, val % self.mod)
    
    def _update(self, node: int, start: int, end: int, idx: int, val: int):
        """Recursive update with mod"""
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._update(2 * node, start, mid, idx, val)
            else:
                self._update(2 * node + 1, mid + 1, end, idx, val)
            self.tree[node] = (self.tree[2 * node] + self.tree[2 * node + 1]) % self.mod
    
    def query(self, left: int, right: int) -> int:
        """Query with modular result"""
        return self._query(1, 0, self.n - 1, left, right)
    
    def _query(self, node: int, start: int, end: int, left: int, right: int):
        """Recursive query with mod"""
        if start > right or end < left:
            return 0
        
        if start >= left and end <= right:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_result = self._query(2 * node, start, mid, left, right)
        right_result = self._query(2 * node + 1, mid + 1, end, left, right)
        
        return (left_result + right_result) % self.mod

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Segment Tree Demo ===\n")
    
    # Example 1: Basic Segment Tree Operations
    print("1. Basic Segment Tree Operations:")
    
    arr = [1, 3, 5, 7, 9, 11]
    seg_tree = SegmentTree(arr, "sum")
    
    print(f"Original array: {arr}")
    print(f"Range sum [1, 3]: {seg_tree.query_range(1, 3)}")  # 3 + 5 + 7 = 15
    print(f"Range sum [0, 4]: {seg_tree.query_range(0, 4)}")  # 1 + 3 + 5 + 7 + 9 = 25
    
    # Update operations
    seg_tree.update_point(2, 10)  # Change arr[2] from 5 to 10
    print(f"After updating index 2 to 10:")
    print(f"Range sum [1, 3]: {seg_tree.query_range(1, 3)}")  # 3 + 10 + 7 = 20
    print()
    
    # Example 2: Different Operations
    print("2. Different Operations:")
    
    operations = ["sum", "min", "max"]
    test_arr = [4, 2, 8, 1, 9, 3]
    
    for op in operations:
        tree = SegmentTree(test_arr, op)
        result = tree.query_range(1, 4)
        print(f"Range {op} [1, 4]: {result}")
    print()
    
    # Example 3: Lazy Propagation
    print("3. Lazy Propagation Segment Tree:")
    
    lazy_arr = [1, 2, 3, 4, 5]
    lazy_tree = LazySegmentTree(lazy_arr, "sum")
    
    print(f"Initial array: {lazy_arr}")
    print(f"Initial range sum [1, 3]: {lazy_tree.query_range(1, 3)}")  # 2 + 3 + 4 = 9
    
    # Range update
    lazy_tree.update_range(1, 3, 5)  # Add 5 to range [1, 3]
    print(f"After adding 5 to range [1, 3]:")
    print(f"Range sum [1, 3]: {lazy_tree.query_range(1, 3)}")  # (2+5) + (3+5) + (4+5) = 24
    print(f"Range sum [0, 4]: {lazy_tree.query_range(0, 4)}")  # 1 + 7 + 8 + 9 + 5 = 30
    print()
    
    # Example 4: Range Set Operations
    print("4. Range Set Operations:")
    
    set_arr = [1, 1, 1, 1, 1]
    set_tree = SegmentTreeRangeSet(set_arr)
    
    print(f"Initial array: {set_arr}")
    print(f"Initial sum: {set_tree.query_range(0, 4)}")
    
    # Set range to specific value
    set_tree.set_range(1, 3, 7)  # Set range [1, 3] to 7
    print(f"After setting range [1, 3] to 7:")
    print(f"Range sum [0, 4]: {set_tree.query_range(0, 4)}")  # 1 + 7 + 7 + 7 + 1 = 23
    print(f"Range sum [1, 3]: {set_tree.query_range(1, 3)}")  # 7 + 7 + 7 = 21
    print()
    
    # Example 5: Persistent Segment Tree
    print("5. Persistent Segment Tree:")
    
    pers_arr = [1, 2, 3, 4, 5]
    pers_tree = PersistentSegmentTree(pers_arr)
    
    print(f"Initial array: {pers_arr}")
    print(f"Version 0 - Range sum [1, 3]: {pers_tree.query(0, 1, 3)}")  # 2 + 3 + 4 = 9
    
    # Create new versions
    v1 = pers_tree.update(0, 2, 10)  # Update index 2 to 10
    v2 = pers_tree.update(v1, 1, 20)  # Update index 1 to 20
    
    print(f"Version {v1} - Range sum [1, 3]: {pers_tree.query(v1, 1, 3)}")  # 2 + 10 + 4 = 16
    print(f"Version {v2} - Range sum [1, 3]: {pers_tree.query(v2, 1, 3)}")  # 20 + 10 + 4 = 34
    print(f"Version 0 - Range sum [1, 3]: {pers_tree.query(0, 1, 3)}")  # Still 9
    print()
    
    # Example 6: 2D Segment Tree
    print("6. 2D Segment Tree:")
    
    matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]
    
    seg_2d = SegmentTree2D(matrix)
    
    print("Matrix:")
    for row in matrix:
        print(row)
    
    print(f"Sum of rectangle (0,0) to (1,1): {seg_2d.query(0, 0, 1, 1)}")  # 1+2+5+6 = 14
    print(f"Sum of rectangle (1,1) to (2,3): {seg_2d.query(1, 1, 2, 3)}")  # 6+7+8+10+11+12 = 54
    print()
    
    # Example 7: Advanced Applications
    print("7. Advanced Applications:")
    
    # GCD Segment Tree
    gcd_arr = [12, 18, 24, 30, 36]
    gcd_tree = SegmentTree(gcd_arr, "gcd")
    
    print(f"GCD array: {gcd_arr}")
    print(f"GCD of range [1, 3]: {gcd_tree.query_range(1, 3)}")  # gcd(18, 24, 30) = 6
    
    # Modular Segment Tree
    mod_arr = [100, 200, 300, 400, 500]
    mod_tree = RangeModularTree(mod_arr, 1000)
    
    print(f"Modular tree (mod 1000): {mod_arr}")
    print(f"Sum [1, 3] mod 1000: {mod_tree.query(1, 3)}")  # (200+300+400) % 1000 = 900
    
    # Update and query again
    mod_tree.update(2, 1500)  # 1500 % 1000 = 500
    print(f"After updating index 2 to 1500:")
    print(f"Sum [1, 3] mod 1000: {mod_tree.query(1, 3)}")  # (200+500+400) % 1000 = 100
    print()
    
    # Example 8: Performance Comparison
    print("8. Performance Test:")
    
    # Large array for performance testing
    large_arr = list(range(1, 1001))  # 1000 elements
    large_tree = SegmentTree(large_arr, "sum")
    large_lazy = LazySegmentTree(large_arr, "sum")
    
    # Single point update vs range update
    print(f"Large array size: {len(large_arr)}")
    print(f"Range sum [100, 200]: {large_tree.query_range(100, 200)}")
    
    # Update and query
    large_tree.update_point(150, 999999)
    large_lazy.update_range(100, 200, 1000)
    
    print(f"After point update: {large_tree.query_range(100, 200)}")
    print(f"After range update: {large_lazy.query_range(100, 200)}")
    
    print("\n=== Demo Complete ===") 