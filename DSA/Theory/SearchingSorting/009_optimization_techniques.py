"""
Optimization Techniques for Searching and Sorting
=================================================

Topics: Cache optimization, memory efficiency, parallel techniques
Companies: High-performance computing, systems programming
Difficulty: Advanced
"""

from typing import List, Optional, Callable
import math

class OptimizationTechniques:
    
    # ==========================================
    # 1. CACHE-FRIENDLY ALGORITHMS
    # ==========================================
    
    def cache_friendly_merge_sort(self, arr: List[int], block_size: int = 64) -> List[int]:
        """Cache-friendly merge sort using blocking
        Time: O(n log n), Better cache performance
        """
        arr = arr.copy()
        n = len(arr)
        
        # Use insertion sort for small blocks
        for start in range(0, n, block_size):
            end = min(start + block_size - 1, n - 1)
            self._insertion_sort_range(arr, start, end)
        
        # Merge blocks
        size = block_size
        while size < n:
            for start in range(0, n, size * 2):
                mid = min(start + size - 1, n - 1)
                end = min(start + size * 2 - 1, n - 1)
                
                if mid < end:
                    self._merge_in_place(arr, start, mid, end)
            
            size *= 2
        
        return arr
    
    def _insertion_sort_range(self, arr: List[int], left: int, right: int):
        """Insertion sort for a range"""
        for i in range(left + 1, right + 1):
            key = arr[i]
            j = i - 1
            while j >= left and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
    
    def _merge_in_place(self, arr: List[int], left: int, mid: int, right: int):
        """In-place merge for cache efficiency"""
        start2 = mid + 1
        
        if arr[mid] <= arr[start2]:
            return
        
        while left <= mid and start2 <= right:
            if arr[left] <= arr[start2]:
                left += 1
            else:
                value = arr[start2]
                index = start2
                
                while index != left:
                    arr[index] = arr[index - 1]
                    index -= 1
                
                arr[left] = value
                left += 1
                mid += 1
                start2 += 1
    
    # ==========================================
    # 2. MEMORY-EFFICIENT TECHNIQUES
    # ==========================================
    
    def in_place_merge_sort(self, arr: List[int]) -> List[int]:
        """In-place merge sort with O(1) extra space
        Time: O(n log² n), Space: O(1)
        """
        arr = arr.copy()
        n = len(arr)
        
        curr_size = 1
        while curr_size < n:
            left_start = 0
            
            while left_start < n - 1:
                mid = min(left_start + curr_size - 1, n - 1)
                right_end = min(left_start + curr_size * 2 - 1, n - 1)
                
                if mid < right_end:
                    self._merge_in_place(arr, left_start, mid, right_end)
                
                left_start += curr_size * 2
            
            curr_size *= 2
        
        return arr
    
    def iterative_quick_sort(self, arr: List[int]) -> List[int]:
        """Iterative quick sort to avoid recursion overhead
        Time: O(n log n) average, Space: O(log n)
        """
        arr = arr.copy()
        stack = [(0, len(arr) - 1)]
        
        while stack:
            low, high = stack.pop()
            
            if low < high:
                pivot = self._partition(arr, low, high)
                
                # Push larger subarray first to minimize stack size
                if pivot - low < high - pivot:
                    stack.append((pivot + 1, high))
                    stack.append((low, pivot - 1))
                else:
                    stack.append((low, pivot - 1))
                    stack.append((pivot + 1, high))
        
        return arr
    
    def _partition(self, arr: List[int], low: int, high: int) -> int:
        """Partition function for quick sort"""
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    # ==========================================
    # 3. ADAPTIVE ALGORITHMS
    # ==========================================
    
    def adaptive_sort(self, arr: List[int]) -> List[int]:
        """Adaptive sorting algorithm that chooses strategy based on data
        Time: Varies based on input, Space: O(1) to O(n)
        """
        n = len(arr)
        
        if n <= 1:
            return arr.copy()
        
        # Analyze data characteristics
        is_sorted = self._check_sorted(arr)
        is_reverse_sorted = self._check_reverse_sorted(arr)
        run_length = self._get_longest_run(arr)
        
        # Choose algorithm based on analysis
        if is_sorted:
            return arr.copy()
        elif is_reverse_sorted:
            return arr[::-1]
        elif run_length > n * 0.7:
            return self._tim_sort_simplified(arr)
        elif n < 50:
            return self._insertion_sort(arr)
        else:
            return self._intro_sort(arr)
    
    def _check_sorted(self, arr: List[int]) -> bool:
        """Check if array is already sorted"""
        return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))
    
    def _check_reverse_sorted(self, arr: List[int]) -> bool:
        """Check if array is reverse sorted"""
        return all(arr[i] >= arr[i + 1] for i in range(len(arr) - 1))
    
    def _get_longest_run(self, arr: List[int]) -> int:
        """Get length of longest sorted subsequence"""
        if not arr:
            return 0
        
        max_run = 1
        current_run = 1
        
        for i in range(1, len(arr)):
            if arr[i] >= arr[i - 1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        
        return max_run
    
    def _tim_sort_simplified(self, arr: List[int]) -> List[int]:
        """Simplified Tim Sort for nearly sorted data"""
        arr = arr.copy()
        min_run = 32
        
        # Sort small runs
        for start in range(0, len(arr), min_run):
            end = min(start + min_run - 1, len(arr) - 1)
            self._insertion_sort_range(arr, start, end)
        
        # Merge runs
        size = min_run
        while size < len(arr):
            for start in range(0, len(arr), size * 2):
                mid = min(start + size - 1, len(arr) - 1)
                end = min(start + size * 2 - 1, len(arr) - 1)
                
                if mid < end:
                    self._merge_in_place(arr, start, mid, end)
            
            size *= 2
        
        return arr
    
    def _insertion_sort(self, arr: List[int]) -> List[int]:
        """Simple insertion sort"""
        arr = arr.copy()
        
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            
            arr[j + 1] = key
        
        return arr
    
    def _intro_sort(self, arr: List[int]) -> List[int]:
        """Simplified Intro Sort"""
        arr = arr.copy()
        max_depth = 2 * int(math.log2(len(arr)))
        self._intro_sort_helper(arr, 0, len(arr) - 1, max_depth)
        return arr
    
    def _intro_sort_helper(self, arr: List[int], low: int, high: int, depth: int):
        """Intro sort helper"""
        if high - low < 16:
            self._insertion_sort_range(arr, low, high)
        elif depth == 0:
            self._heap_sort_range(arr, low, high)
        else:
            pivot = self._partition(arr, low, high)
            self._intro_sort_helper(arr, low, pivot - 1, depth - 1)
            self._intro_sort_helper(arr, pivot + 1, high, depth - 1)
    
    def _heap_sort_range(self, arr: List[int], low: int, high: int):
        """Heap sort for a range"""
        def heapify(arr, n, i, offset):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n and arr[offset + left] > arr[offset + largest]:
                largest = left
            
            if right < n and arr[offset + right] > arr[offset + largest]:
                largest = right
            
            if largest != i:
                arr[offset + i], arr[offset + largest] = arr[offset + largest], arr[offset + i]
                heapify(arr, n, largest, offset)
        
        n = high - low + 1
        
        # Build heap
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i, low)
        
        # Extract elements
        for i in range(n - 1, 0, -1):
            arr[low], arr[low + i] = arr[low + i], arr[low]
            heapify(arr, i, 0, low)
    
    # ==========================================
    # 4. BRANCH PREDICTION OPTIMIZATION
    # ==========================================
    
    def branchless_binary_search(self, arr: List[int], target: int) -> int:
        """Binary search optimized for branch prediction
        Time: O(log n), Fewer branch mispredictions
        """
        n = len(arr)
        pos = -1
        
        for step in [1 << i for i in range(int(math.log2(n)) + 1)][::-1]:
            if pos + step < n and arr[pos + step] <= target:
                pos += step
        
        return pos if pos >= 0 and arr[pos] == target else -1
    
    def optimized_partition(self, arr: List[int], low: int, high: int) -> int:
        """Optimized partition with better branch prediction"""
        pivot = arr[high]
        
        # Three-way partitioning to handle duplicates efficiently
        i = low
        j = low
        k = high
        
        while j < k:
            if arr[j] < pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
                j += 1
            elif arr[j] > pivot:
                k -= 1
                arr[j], arr[k] = arr[k], arr[j]
            else:
                j += 1
        
        arr[j], arr[high] = arr[high], arr[j]
        return j

# Test Examples
def run_examples():
    optimizer = OptimizationTechniques()
    
    print("=== OPTIMIZATION TECHNIQUES ===\n")
    
    # Test data
    import random
    test_arr = [random.randint(1, 1000) for _ in range(100)]
    nearly_sorted = sorted(test_arr)
    # Add some disorder
    for i in range(0, len(nearly_sorted), 10):
        if i + 1 < len(nearly_sorted):
            nearly_sorted[i], nearly_sorted[i + 1] = nearly_sorted[i + 1], nearly_sorted[i]
    
    print("1. CACHE-FRIENDLY MERGE SORT:")
    cache_friendly = optimizer.cache_friendly_merge_sort(test_arr)
    print(f"Sorted correctly: {cache_friendly == sorted(test_arr)}")
    
    print("\n2. IN-PLACE MERGE SORT:")
    in_place = optimizer.in_place_merge_sort(test_arr)
    print(f"Sorted correctly: {in_place == sorted(test_arr)}")
    
    print("\n3. ITERATIVE QUICK SORT:")
    iterative = optimizer.iterative_quick_sort(test_arr)
    print(f"Sorted correctly: {iterative == sorted(test_arr)}")
    
    print("\n4. ADAPTIVE SORT:")
    # Test on different data types
    test_cases = [
        (sorted(test_arr), "Already sorted"),
        (sorted(test_arr)[::-1], "Reverse sorted"),
        (nearly_sorted, "Nearly sorted"),
        (test_arr, "Random data")
    ]
    
    for data, description in test_cases:
        adaptive_result = optimizer.adaptive_sort(data)
        print(f"{description}: {adaptive_result == sorted(data)}")
    
    print("\n5. BRANCHLESS BINARY SEARCH:")
    sorted_arr = sorted(test_arr)
    target = sorted_arr[len(sorted_arr) // 2]
    branchless_result = optimizer.branchless_binary_search(sorted_arr, target)
    print(f"Found target {target} at index: {branchless_result}")
    print(f"Correct result: {sorted_arr[branchless_result] == target if branchless_result >= 0 else False}")

if __name__ == "__main__":
    run_examples() 