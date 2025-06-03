"""
Binary Indexed Tree (Fenwick Tree) - Efficient Range Query Data Structure
This module implements 1D and 2D Binary Indexed Trees with various operations.
"""

from typing import List, Tuple
import bisect

class BinaryIndexedTree:
    """1D Binary Indexed Tree for range sum queries"""
    
    def __init__(self, size: int):
        """
        Initialize BIT with given size
        
        Args:
            size: Size of the array
        """
        self.size = size
        self.tree = [0] * (size + 1)
    
    def update(self, idx: int, delta: int) -> None:
        """
        Update value at index by delta
        
        Time Complexity: O(log n)
        
        Args:
            idx: Index to update (1-indexed)
            delta: Value to add
        """
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & (-idx)  # Add last set bit
    
    def query(self, idx: int) -> int:
        """
        Query prefix sum up to index
        
        Time Complexity: O(log n)
        
        Args:
            idx: Index to query up to (1-indexed)
        
        Returns:
            int: Prefix sum
        """
        result = 0
        while idx > 0:
            result += self.tree[idx]
            idx -= idx & (-idx)  # Remove last set bit
        return result
    
    def range_query(self, left: int, right: int) -> int:
        """
        Query sum in range [left, right]
        
        Time Complexity: O(log n)
        
        Args:
            left, right: Range boundaries (1-indexed, inclusive)
        
        Returns:
            int: Range sum
        """
        return self.query(right) - self.query(left - 1)

class BITPointUpdateRangeQuery:
    """BIT for point updates and range queries"""
    
    def __init__(self, arr: List[int]):
        """
        Initialize BIT from array
        
        Time Complexity: O(n log n)
        
        Args:
            arr: Initial array (0-indexed)
        """
        self.n = len(arr)
        self.bit = BinaryIndexedTree(self.n)
        
        # Build tree from array
        for i, val in enumerate(arr):
            self.bit.update(i + 1, val)
    
    def update_point(self, idx: int, new_val: int) -> None:
        """
        Update single point
        
        Args:
            idx: Index to update (0-indexed)
            new_val: New value
        """
        current = self.bit.range_query(idx + 1, idx + 1)
        delta = new_val - current
        self.bit.update(idx + 1, delta)
    
    def add_point(self, idx: int, delta: int) -> None:
        """
        Add delta to point
        
        Args:
            idx: Index to update (0-indexed)
            delta: Value to add
        """
        self.bit.update(idx + 1, delta)
    
    def query_range(self, left: int, right: int) -> int:
        """
        Query range sum
        
        Args:
            left, right: Range boundaries (0-indexed, inclusive)
        
        Returns:
            int: Range sum
        """
        return self.bit.range_query(left + 1, right + 1)

class BITRangeUpdatePointQuery:
    """BIT for range updates and point queries using difference array"""
    
    def __init__(self, arr: List[int]):
        """
        Initialize BIT for range updates
        
        Args:
            arr: Initial array (0-indexed)
        """
        self.n = len(arr)
        self.original = arr[:]
        
        # Create difference array
        diff = [0] * (self.n + 1)
        diff[0] = arr[0]
        for i in range(1, self.n):
            diff[i] = arr[i] - arr[i - 1]
        
        self.bit = BinaryIndexedTree(self.n)
        for i, val in enumerate(diff[:self.n]):
            self.bit.update(i + 1, val)
    
    def update_range(self, left: int, right: int, delta: int) -> None:
        """
        Add delta to range [left, right]
        
        Time Complexity: O(log n)
        
        Args:
            left, right: Range boundaries (0-indexed, inclusive)
            delta: Value to add to range
        """
        self.bit.update(left + 1, delta)
        if right + 1 < self.n:
            self.bit.update(right + 2, -delta)
    
    def query_point(self, idx: int) -> int:
        """
        Query value at single point
        
        Time Complexity: O(log n)
        
        Args:
            idx: Index to query (0-indexed)
        
        Returns:
            int: Value at index
        """
        return self.bit.query(idx + 1)

class BITRangeUpdateRangeQuery:
    """BIT for both range updates and range queries"""
    
    def __init__(self, arr: List[int]):
        """
        Initialize dual BIT structure
        
        Args:
            arr: Initial array (0-indexed)
        """
        self.n = len(arr)
        self.bit1 = BinaryIndexedTree(self.n)  # For difference array
        self.bit2 = BinaryIndexedTree(self.n)  # For prefix sums
        
        # Initialize with array values
        for i, val in enumerate(arr):
            self.update_range(i, i, val)
    
    def update_range(self, left: int, right: int, delta: int) -> None:
        """
        Add delta to range [left, right]
        
        Time Complexity: O(log n)
        
        Args:
            left, right: Range boundaries (0-indexed, inclusive)
            delta: Value to add to range
        """
        self._update(left, delta)
        self._update(right + 1, -delta)
    
    def _update(self, idx: int, delta: int) -> None:
        """Internal update method"""
        self.bit1.update(idx + 1, delta)
        self.bit2.update(idx + 1, delta * idx)
    
    def query_range(self, left: int, right: int) -> int:
        """
        Query range sum
        
        Time Complexity: O(log n)
        
        Args:
            left, right: Range boundaries (0-indexed, inclusive)
        
        Returns:
            int: Range sum
        """
        return self._prefix_sum(right) - self._prefix_sum(left - 1)
    
    def _prefix_sum(self, idx: int) -> int:
        """Calculate prefix sum up to idx"""
        if idx < 0:
            return 0
        return (idx + 1) * self.bit1.query(idx + 1) - self.bit2.query(idx + 1)

class BIT2D:
    """2D Binary Indexed Tree for 2D range sum queries"""
    
    def __init__(self, rows: int, cols: int):
        """
        Initialize 2D BIT
        
        Args:
            rows, cols: Dimensions of 2D array
        """
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]
    
    def update(self, row: int, col: int, delta: int) -> None:
        """
        Update value at (row, col) by delta
        
        Time Complexity: O(log m * log n)
        
        Args:
            row, col: Position to update (1-indexed)
            delta: Value to add
        """
        r = row
        while r <= self.rows:
            c = col
            while c <= self.cols:
                self.tree[r][c] += delta
                c += c & (-c)
            r += r & (-r)
    
    def query(self, row: int, col: int) -> int:
        """
        Query prefix sum up to (row, col)
        
        Time Complexity: O(log m * log n)
        
        Args:
            row, col: Position to query up to (1-indexed)
        
        Returns:
            int: Prefix sum
        """
        result = 0
        r = row
        while r > 0:
            c = col
            while c > 0:
                result += self.tree[r][c]
                c -= c & (-c)
            r -= r & (-r)
        return result
    
    def range_query(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """
        Query 2D range sum
        
        Time Complexity: O(log m * log n)
        
        Args:
            r1, c1: Top-left corner (1-indexed)
            r2, c2: Bottom-right corner (1-indexed)
        
        Returns:
            int: 2D range sum
        """
        return (self.query(r2, c2) - self.query(r1 - 1, c2) - 
                self.query(r2, c1 - 1) + self.query(r1 - 1, c1 - 1))

class BITWithCoordinateCompression:
    """BIT with coordinate compression for large coordinate values"""
    
    def __init__(self, coordinates: List[int]):
        """
        Initialize BIT with coordinate compression
        
        Args:
            coordinates: List of coordinates that will be used
        """
        # Compress coordinates
        sorted_coords = sorted(set(coordinates))
        self.coord_map = {coord: i + 1 for i, coord in enumerate(sorted_coords)}
        self.reverse_map = {i + 1: coord for i, coord in enumerate(sorted_coords)}
        
        self.bit = BinaryIndexedTree(len(sorted_coords))
    
    def update(self, coord: int, delta: int) -> None:
        """
        Update coordinate by delta
        
        Args:
            coord: Original coordinate
            delta: Value to add
        """
        if coord in self.coord_map:
            self.bit.update(self.coord_map[coord], delta)
    
    def query_less_equal(self, coord: int) -> int:
        """
        Query sum of all coordinates <= coord
        
        Args:
            coord: Coordinate threshold
        
        Returns:
            int: Sum of coordinates <= coord
        """
        # Find largest compressed coordinate <= coord
        idx = bisect.bisect_right(list(self.coord_map.keys()), coord)
        if idx == 0:
            return 0
        
        compressed_coord = list(self.coord_map.keys())[idx - 1]
        return self.bit.query(self.coord_map[compressed_coord])

class BITMinMax:
    """BIT for range minimum/maximum queries (using negative values trick)"""
    
    def __init__(self, arr: List[int], operation: str = "min"):
        """
        Initialize BIT for min/max queries
        
        Args:
            arr: Initial array
            operation: "min" or "max"
        """
        self.n = len(arr)
        self.op = operation
        
        if operation == "min":
            self.identity = float('inf')
            self.combine = min
        else:  # max
            self.identity = float('-inf')
            self.combine = max
        
        self.tree = [self.identity] * (self.n + 1)
        
        # Build tree
        for i, val in enumerate(arr):
            self.update(i + 1, val)
    
    def update(self, idx: int, val: int) -> None:
        """
        Update value at index
        
        Time Complexity: O(log n)
        
        Args:
            idx: Index to update (1-indexed)
            val: New value
        """
        while idx <= self.n:
            self.tree[idx] = self.combine(self.tree[idx], val)
            idx += idx & (-idx)
    
    def query(self, idx: int) -> int:
        """
        Query min/max in prefix [1, idx]
        
        Time Complexity: O(log n)
        
        Args:
            idx: Index to query up to (1-indexed)
        
        Returns:
            int: Min/max value in prefix
        """
        result = self.identity
        while idx > 0:
            result = self.combine(result, self.tree[idx])
            idx -= idx & (-idx)
        return result

class BITMultiDimensional:
    """Multi-dimensional BIT (generalized for any dimension)"""
    
    def __init__(self, dimensions: List[int]):
        """
        Initialize multi-dimensional BIT
        
        Args:
            dimensions: List of sizes for each dimension
        """
        self.dims = dimensions
        self.tree = {}
    
    def _get_key(self, coords: List[int]) -> str:
        """Generate key for coordinate tuple"""
        return ','.join(map(str, coords))
    
    def update(self, coords: List[int], delta: int) -> None:
        """
        Update multi-dimensional coordinate
        
        Args:
            coords: Coordinates (1-indexed)
            delta: Value to add
        """
        def update_recursive(dim: int, current_coords: List[int]):
            if dim == len(self.dims):
                key = self._get_key(current_coords)
                self.tree[key] = self.tree.get(key, 0) + delta
                return
            
            idx = coords[dim]
            while idx <= self.dims[dim]:
                current_coords[dim] = idx
                update_recursive(dim + 1, current_coords[:])
                idx += idx & (-idx)
        
        update_recursive(0, [0] * len(self.dims))
    
    def query(self, coords: List[int]) -> int:
        """
        Query prefix sum up to coordinates
        
        Args:
            coords: Coordinates to query up to (1-indexed)
        
        Returns:
            int: Prefix sum
        """
        def query_recursive(dim: int, current_coords: List[int]) -> int:
            if dim == len(self.dims):
                key = self._get_key(current_coords)
                return self.tree.get(key, 0)
            
            result = 0
            idx = coords[dim]
            while idx > 0:
                current_coords[dim] = idx
                result += query_recursive(dim + 1, current_coords[:])
                idx -= idx & (-idx)
            
            return result
        
        return query_recursive(0, [0] * len(self.dims))

# ==================== ADVANCED BIT APPLICATIONS ====================

class BITInversions:
    """BIT for counting inversions in array"""
    
    def count_inversions(self, arr: List[int]) -> int:
        """
        Count inversions using BIT
        
        Time Complexity: O(n log n)
        
        Args:
            arr: Array to count inversions in
        
        Returns:
            int: Number of inversions
        """
        # Coordinate compression
        sorted_vals = sorted(set(arr))
        coord_map = {val: i + 1 for i, val in enumerate(sorted_vals)}
        
        bit = BinaryIndexedTree(len(sorted_vals))
        inversions = 0
        
        for i in range(len(arr) - 1, -1, -1):
            compressed = coord_map[arr[i]]
            
            # Count elements smaller than current element
            inversions += bit.query(compressed - 1)
            
            # Add current element
            bit.update(compressed, 1)
        
        return inversions

class BITKthElement:
    """BIT for finding kth smallest element"""
    
    def __init__(self, max_val: int):
        """
        Initialize BIT for kth element queries
        
        Args:
            max_val: Maximum possible value
        """
        self.bit = BinaryIndexedTree(max_val)
        self.max_val = max_val
    
    def add_element(self, val: int) -> None:
        """Add element to set"""
        self.bit.update(val, 1)
    
    def remove_element(self, val: int) -> None:
        """Remove element from set"""
        self.bit.update(val, -1)
    
    def find_kth_smallest(self, k: int) -> int:
        """
        Find kth smallest element
        
        Time Complexity: O(log^2 n)
        
        Args:
            k: Position (1-indexed)
        
        Returns:
            int: kth smallest element
        """
        left, right = 1, self.max_val
        
        while left < right:
            mid = (left + right) // 2
            count = self.bit.query(mid)
            
            if count >= k:
                right = mid
            else:
                left = mid + 1
        
        return left

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Binary Indexed Tree Demo ===\n")
    
    # Example 1: Basic 1D BIT operations
    print("1. Basic 1D BIT Operations:")
    
    arr = [1, 3, 5, 7, 9, 11]
    bit = BITPointUpdateRangeQuery(arr)
    
    print(f"Original array: {arr}")
    print(f"Range sum [0, 2]: {bit.query_range(0, 2)}")  # 1 + 3 + 5 = 9
    print(f"Range sum [1, 4]: {bit.query_range(1, 4)}")  # 3 + 5 + 7 + 9 = 24
    
    # Update operations
    bit.update_point(2, 10)  # Change arr[2] from 5 to 10
    print(f"After updating index 2 to 10:")
    print(f"Range sum [0, 2]: {bit.query_range(0, 2)}")  # 1 + 3 + 10 = 14
    
    bit.add_point(3, 5)  # Add 5 to arr[3]
    print(f"After adding 5 to index 3:")
    print(f"Range sum [1, 4]: {bit.query_range(1, 4)}")  # 3 + 10 + 12 + 9 = 34
    print()
    
    # Example 2: Range Update, Point Query
    print("2. Range Update, Point Query:")
    
    arr2 = [0, 0, 0, 0, 0]
    range_bit = BITRangeUpdatePointQuery(arr2)
    
    print(f"Initial array: {arr2}")
    
    # Range updates
    range_bit.update_range(1, 3, 5)  # Add 5 to range [1, 3]
    range_bit.update_range(2, 4, 3)  # Add 3 to range [2, 4]
    
    print("After range updates:")
    for i in range(5):
        print(f"arr[{i}] = {range_bit.query_point(i)}")
    print()
    
    # Example 3: Range Update, Range Query
    print("3. Range Update, Range Query:")
    
    arr3 = [1, 2, 3, 4, 5]
    range_range_bit = BITRangeUpdateRangeQuery(arr3)
    
    print(f"Initial array: {arr3}")
    print(f"Initial range sum [1, 3]: {range_range_bit.query_range(1, 3)}")
    
    # Range update
    range_range_bit.update_range(1, 3, 10)
    print(f"After adding 10 to range [1, 3]:")
    print(f"Range sum [1, 3]: {range_range_bit.query_range(1, 3)}")
    print(f"Range sum [0, 4]: {range_range_bit.query_range(0, 4)}")
    print()
    
    # Example 4: 2D BIT
    print("4. 2D Binary Indexed Tree:")
    
    bit_2d = BIT2D(4, 4)
    
    # Add some values
    updates_2d = [(1, 1, 5), (2, 2, 3), (3, 3, 7), (4, 4, 2)]
    
    for r, c, val in updates_2d:
        bit_2d.update(r, c, val)
        print(f"Added {val} at ({r}, {c})")
    
    # Query ranges
    print(f"Sum in rectangle (1,1) to (2,2): {bit_2d.range_query(1, 1, 2, 2)}")
    print(f"Sum in rectangle (2,2) to (4,4): {bit_2d.range_query(2, 2, 4, 4)}")
    print()
    
    # Example 5: Coordinate Compression
    print("5. Coordinate Compression:")
    
    large_coords = [1000000, 500, 999999, 1, 750000]
    compressed_bit = BITWithCoordinateCompression(large_coords)
    
    # Add values at large coordinates
    compressed_bit.update(1000000, 10)
    compressed_bit.update(500, 5)
    compressed_bit.update(999999, 15)
    
    print(f"Sum of coordinates <= 500: {compressed_bit.query_less_equal(500)}")
    print(f"Sum of coordinates <= 750000: {compressed_bit.query_less_equal(750000)}")
    print(f"Sum of coordinates <= 1000000: {compressed_bit.query_less_equal(1000000)}")
    print()
    
    # Example 6: Inversion Counting
    print("6. Inversion Counting:")
    
    inversion_arr = [5, 2, 6, 1, 3, 4]
    inversion_counter = BITInversions()
    
    inversions = inversion_counter.count_inversions(inversion_arr)
    print(f"Array: {inversion_arr}")
    print(f"Number of inversions: {inversions}")
    
    # Verify manually for small example
    manual_inversions = 0
    for i in range(len(inversion_arr)):
        for j in range(i + 1, len(inversion_arr)):
            if inversion_arr[i] > inversion_arr[j]:
                manual_inversions += 1
    print(f"Manual count (verification): {manual_inversions}")
    print()
    
    # Example 7: Kth Smallest Element
    print("7. Kth Smallest Element:")
    
    kth_bit = BITKthElement(100)
    
    # Add elements
    elements = [10, 5, 15, 3, 12, 8, 20]
    for elem in elements:
        kth_bit.add_element(elem)
        print(f"Added {elem}")
    
    # Find kth smallest
    for k in range(1, min(6, len(elements) + 1)):
        kth_smallest = kth_bit.find_kth_smallest(k)
        print(f"{k}th smallest element: {kth_smallest}")
    
    # Remove an element and find kth smallest again
    kth_bit.remove_element(10)
    print("After removing 10:")
    print(f"3rd smallest element: {kth_bit.find_kth_smallest(3)}")
    print()
    
    # Example 8: Multi-dimensional BIT
    print("8. Multi-dimensional BIT:")
    
    # 3D BIT example
    multi_bit = BITMultiDimensional([3, 3, 3])
    
    # Update some coordinates
    multi_bit.update([1, 1, 1], 5)
    multi_bit.update([2, 2, 2], 3)
    multi_bit.update([3, 3, 3], 7)
    
    print("3D BIT updates:")
    print(f"Sum from (1,1,1) to (2,2,2): {multi_bit.query([2, 2, 2])}")
    print(f"Sum from (1,1,1) to (3,3,3): {multi_bit.query([3, 3, 3])}")
    
    print("\n=== Demo Complete ===") 