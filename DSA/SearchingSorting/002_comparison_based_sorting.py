"""
Comparison-Based Sorting Algorithms
==================================

Topics: Bubble, Selection, Insertion, Merge, Quick, Heap Sort
Companies: Google, Facebook, Amazon, Microsoft, Apple
Difficulty: Easy to Hard
"""

from typing import List
import random

class ComparisonBasedSorting:
    
    # ==========================================
    # 1. SIMPLE SORTING ALGORITHMS (O(n²))
    # ==========================================
    
    def bubble_sort(self, arr: List[int]) -> List[int]:
        """Bubble Sort - Simple but inefficient
        Time: O(n²), Space: O(1)
        Stable: Yes, In-place: Yes
        """
        arr = arr.copy()
        n = len(arr)
        
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True
            
            # Optimization: if no swapping occurred, array is sorted
            if not swapped:
                break
        
        return arr
    
    def selection_sort(self, arr: List[int]) -> List[int]:
        """Selection Sort - Find minimum and place at beginning
        Time: O(n²), Space: O(1)
        Stable: No, In-place: Yes
        """
        arr = arr.copy()
        n = len(arr)
        
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        
        return arr
    
    def insertion_sort(self, arr: List[int]) -> List[int]:
        """Insertion Sort - Insert each element in correct position
        Time: O(n²), Space: O(1)
        Stable: Yes, In-place: Yes
        Best for small arrays or nearly sorted arrays
        """
        arr = arr.copy()
        
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            
            # Move elements greater than key one position ahead
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            
            arr[j + 1] = key
        
        return arr
    
    def binary_insertion_sort(self, arr: List[int]) -> List[int]:
        """Binary Insertion Sort - Use binary search to find position
        Time: O(n²), Space: O(1)
        Better than regular insertion sort for comparison-heavy operations
        """
        arr = arr.copy()
        
        for i in range(1, len(arr)):
            key = arr[i]
            # Find position to insert using binary search
            left, right = 0, i
            
            while left < right:
                mid = (left + right) // 2
                if arr[mid] > key:
                    right = mid
                else:
                    left = mid + 1
            
            # Shift elements and insert
            for j in range(i, left, -1):
                arr[j] = arr[j - 1]
            arr[left] = key
        
        return arr
    
    # ==========================================
    # 2. DIVIDE AND CONQUER ALGORITHMS
    # ==========================================
    
    def merge_sort(self, arr: List[int]) -> List[int]:
        """Merge Sort - Divide and conquer approach
        Time: O(n log n), Space: O(n)
        Stable: Yes, In-place: No
        """
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = self.merge_sort(arr[:mid])
        right = self.merge_sort(arr[mid:])
        
        return self._merge(left, right)
    
    def _merge(self, left: List[int], right: List[int]) -> List[int]:
        """Helper method to merge two sorted arrays"""
        result = []
        i = j = 0
        
        # Merge the two arrays
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        # Add remaining elements
        result.extend(left[i:])
        result.extend(right[j:])
        
        return result
    
    def merge_sort_inplace(self, arr: List[int]) -> List[int]:
        """In-place merge sort (more complex implementation)
        Time: O(n log n), Space: O(log n) for recursion
        """
        arr = arr.copy()
        self._merge_sort_inplace_helper(arr, 0, len(arr) - 1)
        return arr
    
    def _merge_sort_inplace_helper(self, arr: List[int], left: int, right: int):
        """Helper for in-place merge sort"""
        if left < right:
            mid = (left + right) // 2
            self._merge_sort_inplace_helper(arr, left, mid)
            self._merge_sort_inplace_helper(arr, mid + 1, right)
            self._merge_inplace(arr, left, mid, right)
    
    def _merge_inplace(self, arr: List[int], left: int, mid: int, right: int):
        """In-place merge operation"""
        start2 = mid + 1
        
        if arr[mid] <= arr[start2]:
            return
        
        while left <= mid and start2 <= right:
            if arr[left] <= arr[start2]:
                left += 1
            else:
                value = arr[start2]
                index = start2
                
                # Shift elements
                while index != left:
                    arr[index] = arr[index - 1]
                    index -= 1
                
                arr[left] = value
                left += 1
                mid += 1
                start2 += 1
    
    def quick_sort(self, arr: List[int]) -> List[int]:
        """Quick Sort - Divide around pivot
        Time: O(n log n) average, O(n²) worst, Space: O(log n)
        Stable: No, In-place: Yes
        """
        arr = arr.copy()
        self._quick_sort_helper(arr, 0, len(arr) - 1)
        return arr
    
    def _quick_sort_helper(self, arr: List[int], low: int, high: int):
        """Helper for quick sort"""
        if low < high:
            pivot_index = self._partition(arr, low, high)
            self._quick_sort_helper(arr, low, pivot_index - 1)
            self._quick_sort_helper(arr, pivot_index + 1, high)
    
    def _partition(self, arr: List[int], low: int, high: int) -> int:
        """Lomuto partition scheme"""
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    def quick_sort_randomized(self, arr: List[int]) -> List[int]:
        """Randomized Quick Sort - Better average performance
        Time: O(n log n) expected, Space: O(log n)
        """
        arr = arr.copy()
        self._quick_sort_randomized_helper(arr, 0, len(arr) - 1)
        return arr
    
    def _quick_sort_randomized_helper(self, arr: List[int], low: int, high: int):
        """Helper for randomized quick sort"""
        if low < high:
            # Randomly select pivot
            random_index = random.randint(low, high)
            arr[random_index], arr[high] = arr[high], arr[random_index]
            
            pivot_index = self._partition(arr, low, high)
            self._quick_sort_randomized_helper(arr, low, pivot_index - 1)
            self._quick_sort_randomized_helper(arr, pivot_index + 1, high)
    
    def quick_sort_3way(self, arr: List[int]) -> List[int]:
        """3-Way Quick Sort - Efficient for arrays with many duplicates
        Time: O(n log n), Space: O(log n)
        """
        arr = arr.copy()
        self._quick_sort_3way_helper(arr, 0, len(arr) - 1)
        return arr
    
    def _quick_sort_3way_helper(self, arr: List[int], low: int, high: int):
        """Helper for 3-way quick sort"""
        if low < high:
            lt, gt = self._partition_3way(arr, low, high)
            self._quick_sort_3way_helper(arr, low, lt - 1)
            self._quick_sort_3way_helper(arr, gt + 1, high)
    
    def _partition_3way(self, arr: List[int], low: int, high: int) -> tuple:
        """3-way partition"""
        pivot = arr[low]
        i = low
        lt = low
        gt = high
        
        while i <= gt:
            if arr[i] < pivot:
                arr[lt], arr[i] = arr[i], arr[lt]
                lt += 1
                i += 1
            elif arr[i] > pivot:
                arr[i], arr[gt] = arr[gt], arr[i]
                gt -= 1
            else:
                i += 1
        
        return lt, gt
    
    # ==========================================
    # 3. HEAP SORT
    # ==========================================
    
    def heap_sort(self, arr: List[int]) -> List[int]:
        """Heap Sort - Use heap data structure
        Time: O(n log n), Space: O(1)
        Stable: No, In-place: Yes
        """
        arr = arr.copy()
        n = len(arr)
        
        # Build max heap
        for i in range(n // 2 - 1, -1, -1):
            self._heapify(arr, n, i)
        
        # Extract elements from heap one by one
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            self._heapify(arr, i, 0)
        
        return arr
    
    def _heapify(self, arr: List[int], n: int, i: int):
        """Maintain max heap property"""
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        if left < n and arr[left] > arr[largest]:
            largest = left
        
        if right < n and arr[right] > arr[largest]:
            largest = right
        
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self._heapify(arr, n, largest)
    
    # ==========================================
    # 4. HYBRID SORTING ALGORITHMS
    # ==========================================
    
    def intro_sort(self, arr: List[int]) -> List[int]:
        """Intro Sort - Hybrid of Quick Sort, Heap Sort, and Insertion Sort
        Time: O(n log n), Space: O(log n)
        Used in many standard libraries
        """
        arr = arr.copy()
        max_depth = 2 * (len(arr).bit_length() - 1)
        self._intro_sort_helper(arr, 0, len(arr) - 1, max_depth)
        return arr
    
    def _intro_sort_helper(self, arr: List[int], low: int, high: int, depth_limit: int):
        """Helper for intro sort"""
        size = high - low + 1
        
        if size <= 16:  # Use insertion sort for small arrays
            self._insertion_sort_range(arr, low, high)
        elif depth_limit == 0:  # Use heap sort when depth limit reached
            self._heap_sort_range(arr, low, high)
        else:  # Use quick sort
            pivot = self._partition(arr, low, high)
            self._intro_sort_helper(arr, low, pivot - 1, depth_limit - 1)
            self._intro_sort_helper(arr, pivot + 1, high, depth_limit - 1)
    
    def _insertion_sort_range(self, arr: List[int], low: int, high: int):
        """Insertion sort for a range"""
        for i in range(low + 1, high + 1):
            key = arr[i]
            j = i - 1
            while j >= low and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
    
    def _heap_sort_range(self, arr: List[int], low: int, high: int):
        """Heap sort for a range"""
        # Build heap
        for i in range((high - low) // 2 - 1, -1, -1):
            self._heapify_range(arr, low, high, low + i)
        
        # Extract elements
        for i in range(high, low, -1):
            arr[low], arr[i] = arr[i], arr[low]
            self._heapify_range(arr, low, i - 1, low)
    
    def _heapify_range(self, arr: List[int], low: int, high: int, i: int):
        """Heapify for a range"""
        largest = i
        left = 2 * (i - low) + 1 + low
        right = 2 * (i - low) + 2 + low
        
        if left <= high and arr[left] > arr[largest]:
            largest = left
        
        if right <= high and arr[right] > arr[largest]:
            largest = right
        
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self._heapify_range(arr, low, high, largest)

# Test Examples
def run_examples():
    cbs = ComparisonBasedSorting()
    
    print("=== COMPARISON-BASED SORTING ALGORITHMS ===\n")
    
    # Test arrays
    test_arr = [64, 34, 25, 12, 22, 11, 90, 5, 77, 30]
    small_arr = [5, 2, 4, 6, 1, 3]
    duplicates_arr = [4, 2, 2, 8, 3, 3, 1]
    
    print("Original array:", test_arr)
    print()
    
    # Test simple sorting algorithms
    print("1. SIMPLE SORTING (O(n²)):")
    print("Bubble Sort:", cbs.bubble_sort(test_arr))
    print("Selection Sort:", cbs.selection_sort(test_arr))
    print("Insertion Sort:", cbs.insertion_sort(test_arr))
    print("Binary Insertion Sort:", cbs.binary_insertion_sort(small_arr))
    
    print("\n2. DIVIDE AND CONQUER (O(n log n)):")
    print("Merge Sort:", cbs.merge_sort(test_arr))
    print("Quick Sort:", cbs.quick_sort(test_arr))
    print("3-Way Quick Sort:", cbs.quick_sort_3way(duplicates_arr))
    
    print("\n3. HEAP SORT:")
    print("Heap Sort:", cbs.heap_sort(test_arr))
    
    print("\n4. HYBRID ALGORITHMS:")
    print("Intro Sort:", cbs.intro_sort(test_arr))

if __name__ == "__main__":
    run_examples() 