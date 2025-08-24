"""
378. Kth Smallest Element in a Sorted Matrix - Multiple Approaches
Difficulty: Medium

Given an n x n matrix where each of the rows and columns is sorted in ascending order, return the kth smallest element in the matrix.

Note that it is the kth smallest element in the sorted order, not the kth distinct element.

You must find a solution with a memory complexity better than O(n²).
"""

from typing import List
import heapq

class KthSmallestElementInSortedMatrix:
    """Multiple approaches to find kth smallest element in sorted matrix"""
    
    def kthSmallest_min_heap(self, matrix: List[List[int]], k: int) -> int:
        """
        Approach 1: Min Heap (Optimal for small k)
        
        Use min heap to track smallest elements.
        
        Time: O(k log n), Space: O(n)
        """
        n = len(matrix)
        heap = []
        
        # Initialize heap with first row
        for j in range(n):
            heapq.heappush(heap, (matrix[0][j], 0, j))
        
        # Extract k-1 elements
        for _ in range(k - 1):
            val, i, j = heapq.heappop(heap)
            
            # Add next element from same column
            if i + 1 < n:
                heapq.heappush(heap, (matrix[i + 1][j], i + 1, j))
        
        return heap[0][0]
    
    def kthSmallest_binary_search(self, matrix: List[List[int]], k: int) -> int:
        """
        Approach 2: Binary Search (Optimal for large k)
        
        Binary search on the value range.
        
        Time: O(n log(max-min)), Space: O(1)
        """
        n = len(matrix)
        left, right = matrix[0][0], matrix[n-1][n-1]
        
        def count_less_equal(target: int) -> int:
            """Count elements <= target"""
            count = 0
            row, col = n - 1, 0  # Start from bottom-left
            
            while row >= 0 and col < n:
                if matrix[row][col] <= target:
                    count += row + 1  # All elements in this column up to row
                    col += 1
                else:
                    row -= 1
            
            return count
        
        while left < right:
            mid = (left + right) // 2
            
            if count_less_equal(mid) < k:
                left = mid + 1
            else:
                right = mid
        
        return left
    
    def kthSmallest_max_heap(self, matrix: List[List[int]], k: int) -> int:
        """
        Approach 3: Max Heap of Size K
        
        Maintain max heap of k smallest elements.
        
        Time: O(n² log k), Space: O(k)
        """
        max_heap = []
        
        for row in matrix:
            for val in row:
                if len(max_heap) < k:
                    heapq.heappush(max_heap, -val)
                elif val < -max_heap[0]:
                    heapq.heappop(max_heap)
                    heapq.heappush(max_heap, -val)
        
        return -max_heap[0]
    
    def kthSmallest_merge_sorted_lists(self, matrix: List[List[int]], k: int) -> int:
        """
        Approach 4: Merge K Sorted Lists
        
        Treat each row as a sorted list and merge.
        
        Time: O(k log n), Space: O(n)
        """
        n = len(matrix)
        heap = []
        
        # Initialize heap with first element of each row
        for i in range(n):
            heapq.heappush(heap, (matrix[i][0], i, 0))
        
        # Extract k elements
        for _ in range(k):
            val, row, col = heapq.heappop(heap)
            
            if _ == k - 1:  # kth element
                return val
            
            # Add next element from same row
            if col + 1 < n:
                heapq.heappush(heap, (matrix[row][col + 1], row, col + 1))
        
        return -1  # Should not reach here
    
    def kthSmallest_flatten_and_sort(self, matrix: List[List[int]], k: int) -> int:
        """
        Approach 5: Flatten and Sort (Not optimal for space)
        
        Flatten matrix and sort.
        
        Time: O(n² log n²), Space: O(n²)
        """
        flattened = []
        for row in matrix:
            flattened.extend(row)
        
        flattened.sort()
        return flattened[k - 1]


def test_kth_smallest_element_in_sorted_matrix():
    """Test kth smallest element algorithms"""
    solver = KthSmallestElementInSortedMatrix()
    
    test_cases = [
        ([[1,5,9],[10,11,13],[12,13,15]], 8, 13, "Example 1"),
        ([[-5]], 1, -5, "Single element"),
        ([[1,2],[1,3]], 2, 1, "2x2 matrix"),
        ([[1,2],[1,3]], 3, 2, "2x2 matrix, k=3"),
        ([[1,3,5],[6,7,12],[11,14,14]], 6, 11, "3x3 matrix"),
        ([[1,4,7,11],[2,5,8,12],[3,6,9,16],[10,13,14,17]], 5, 5, "4x4 matrix"),
    ]
    
    algorithms = [
        ("Min Heap", solver.kthSmallest_min_heap),
        ("Binary Search", solver.kthSmallest_binary_search),
        ("Max Heap", solver.kthSmallest_max_heap),
        ("Merge Sorted Lists", solver.kthSmallest_merge_sorted_lists),
        ("Flatten and Sort", solver.kthSmallest_flatten_and_sort),
    ]
    
    print("=== Testing Kth Smallest Element in Sorted Matrix ===")
    
    for matrix, k, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print("Matrix:")
        for row in matrix:
            print(f"  {row}")
        print(f"k: {k}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(matrix, k)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_binary_search_approach():
    """Demonstrate binary search approach"""
    print("\n=== Binary Search Approach Demonstration ===")
    
    matrix = [
        [1,  5,  9],
        [10, 11, 13],
        [12, 13, 15]
    ]
    k = 8
    
    print("Matrix:")
    for row in matrix:
        print(f"  {row}")
    print(f"Finding {k}th smallest element")
    
    n = len(matrix)
    left, right = matrix[0][0], matrix[n-1][n-1]
    
    print(f"\nSearch range: [{left}, {right}]")
    
    def count_less_equal(target: int) -> int:
        """Count elements <= target with visualization"""
        count = 0
        row, col = n - 1, 0
        path = []
        
        while row >= 0 and col < n:
            path.append((row, col, matrix[row][col]))
            
            if matrix[row][col] <= target:
                count += row + 1
                col += 1
            else:
                row -= 1
        
        print(f"    Counting elements <= {target}:")
        print(f"    Path: {path}")
        print(f"    Count: {count}")
        
        return count
    
    step = 1
    while left < right:
        mid = (left + right) // 2
        print(f"\nStep {step}: mid = {mid}")
        
        count = count_less_equal(mid)
        
        if count < k:
            left = mid + 1
            print(f"    Count {count} < k={k}, search right: [{left}, {right}]")
        else:
            right = mid
            print(f"    Count {count} >= k={k}, search left: [{left}, {right}]")
        
        step += 1
    
    print(f"\nResult: {left}")


if __name__ == "__main__":
    test_kth_smallest_element_in_sorted_matrix()
    demonstrate_binary_search_approach()

"""
Kth Smallest Element in a Sorted Matrix demonstrates advanced heap and
binary search techniques for 2D sorted arrays, including space-efficient
solutions and multiple optimization strategies.
"""
