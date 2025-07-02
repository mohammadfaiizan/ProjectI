"""
Array Sorting Algorithms - Complete Implementation
================================================

Topics: All major sorting algorithms with analysis
Companies: Fundamental for all tech interviews
Difficulty: Easy to Hard
"""

from typing import List
import random, time

class ArraySortingAlgorithms:
    
    def __init__(self):
        """Initialize with comparison counter for analysis"""
        self.comparisons = 0
        self.swaps = 0
    
    def reset_counters(self):
        """Reset operation counters"""
        self.comparisons = 0
        self.swaps = 0
    
    # ==========================================
    # 1. SIMPLE SORTING ALGORITHMS O(n²)
    # ==========================================
    
    def bubble_sort(self, arr: List[int]) -> List[int]:
        """Bubble Sort - Stable
        Time: O(n²), Space: O(1)
        """
        self.reset_counters()
        n = len(arr)
        
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                self.comparisons += 1
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    self.swaps += 1
                    swapped = True
            
            if not swapped:  # Optimization: early termination
                break
        
        return arr
    
    def selection_sort(self, arr: List[int]) -> List[int]:
        """Selection Sort - Unstable
        Time: O(n²), Space: O(1)
        """
        self.reset_counters()
        n = len(arr)
        
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                self.comparisons += 1
                if arr[j] < arr[min_idx]:
                    min_idx = j
            
            if min_idx != i:
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
                self.swaps += 1
        
        return arr
    
    def insertion_sort(self, arr: List[int]) -> List[int]:
        """Insertion Sort - Stable
        Time: O(n²), Space: O(1)
        """
        self.reset_counters()
        
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            
            while j >= 0:
                self.comparisons += 1
                if arr[j] > key:
                    arr[j + 1] = arr[j]
                    self.swaps += 1
                    j -= 1
                else:
                    break
            
            arr[j + 1] = key
        
        return arr
    
    # ==========================================
    # 2. EFFICIENT SORTING ALGORITHMS O(n log n)
    # ==========================================
    
    def merge_sort(self, arr: List[int]) -> List[int]:
        """Merge Sort - Stable, Divide & Conquer
        Time: O(n log n), Space: O(n)
        """
        if len(arr) <= 1:
            return arr
        
        def merge(left: List[int], right: List[int]) -> List[int]:
            result = []
            i = j = 0
            
            while i < len(left) and j < len(right):
                self.comparisons += 1
                if left[i] <= right[j]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
            
            result.extend(left[i:])
            result.extend(right[j:])
            return result
        
        mid = len(arr) // 2
        left = self.merge_sort(arr[:mid])
        right = self.merge_sort(arr[mid:])
        
        return merge(left, right)
    
    def quick_sort(self, arr: List[int]) -> List[int]:
        """Quick Sort - Unstable, Divide & Conquer
        Time: O(n log n) average, O(n²) worst, Space: O(log n)
        """
        def quicksort_helper(low: int, high: int):
            if low < high:
                pi = partition(low, high)
                quicksort_helper(low, pi - 1)
                quicksort_helper(pi + 1, high)
        
        def partition(low: int, high: int) -> int:
            # Choose rightmost element as pivot
            pivot = arr[high]
            i = low - 1
            
            for j in range(low, high):
                self.comparisons += 1
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
                    self.swaps += 1
            
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            self.swaps += 1
            return i + 1
        
        self.reset_counters()
        quicksort_helper(0, len(arr) - 1)
        return arr
    
    def heap_sort(self, arr: List[int]) -> List[int]:
        """Heap Sort - Unstable
        Time: O(n log n), Space: O(1)
        """
        def heapify(n: int, i: int):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n:
                self.comparisons += 1
                if arr[left] > arr[largest]:
                    largest = left
            
            if right < n:
                self.comparisons += 1
                if arr[right] > arr[largest]:
                    largest = right
            
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self.swaps += 1
                heapify(n, largest)
        
        self.reset_counters()
        n = len(arr)
        
        # Build max heap
        for i in range(n // 2 - 1, -1, -1):
            heapify(n, i)
        
        # Extract elements from heap
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            self.swaps += 1
            heapify(i, 0)
        
        return arr
    
    # ==========================================
    # 3. LINEAR TIME SORTING ALGORITHMS
    # ==========================================
    
    def counting_sort(self, arr: List[int]) -> List[int]:
        """Counting Sort - Stable, for non-negative integers
        Time: O(n + k), Space: O(k) where k is range
        """
        if not arr:
            return arr
        
        # Find range
        max_val = max(arr)
        min_val = min(arr)
        range_val = max_val - min_val + 1
        
        # Count occurrences
        count = [0] * range_val
        for num in arr:
            count[num - min_val] += 1
        
        # Cumulative count for stable sorting
        for i in range(1, range_val):
            count[i] += count[i - 1]
        
        # Build output array
        output = [0] * len(arr)
        for i in range(len(arr) - 1, -1, -1):
            output[count[arr[i] - min_val] - 1] = arr[i]
            count[arr[i] - min_val] -= 1
        
        return output
    
    def radix_sort(self, arr: List[int]) -> List[int]:
        """Radix Sort - Stable, for non-negative integers
        Time: O(d * (n + k)), Space: O(n + k)
        """
        if not arr:
            return arr
        
        def counting_sort_for_radix(arr: List[int], exp: int) -> List[int]:
            n = len(arr)
            output = [0] * n
            count = [0] * 10
            
            # Count occurrences
            for num in arr:
                index = (num // exp) % 10
                count[index] += 1
            
            # Cumulative count
            for i in range(1, 10):
                count[i] += count[i - 1]
            
            # Build output array
            for i in range(n - 1, -1, -1):
                index = (arr[i] // exp) % 10
                output[count[index] - 1] = arr[i]
                count[index] -= 1
            
            for i in range(n):
                arr[i] = output[i]
            
            return arr
        
        max_val = max(arr)
        exp = 1
        
        while max_val // exp > 0:
            counting_sort_for_radix(arr, exp)
            exp *= 10
        
        return arr
    
    def bucket_sort(self, arr: List[float], num_buckets: int = 10) -> List[float]:
        """Bucket Sort - for uniformly distributed data
        Time: O(n + k), Space: O(n + k)
        """
        if not arr:
            return arr
        
        # Create buckets
        buckets = [[] for _ in range(num_buckets)]
        
        # Distribute elements into buckets
        max_val = max(arr)
        min_val = min(arr)
        range_val = max_val - min_val
        
        for num in arr:
            if range_val == 0:
                bucket_index = 0
            else:
                bucket_index = int((num - min_val) / range_val * (num_buckets - 1))
            buckets[bucket_index].append(num)
        
        # Sort individual buckets and concatenate
        result = []
        for bucket in buckets:
            bucket.sort()  # Using built-in sort for simplicity
            result.extend(bucket)
        
        return result
    
    # ==========================================
    # 4. SPECIALIZED SORTING ALGORITHMS
    # ==========================================
    
    def shell_sort(self, arr: List[int]) -> List[int]:
        """Shell Sort - Unstable, improved insertion sort
        Time: O(n²) worst, O(n log n) average, Space: O(1)
        """
        self.reset_counters()
        n = len(arr)
        gap = n // 2
        
        while gap > 0:
            for i in range(gap, n):
                temp = arr[i]
                j = i
                
                while j >= gap:
                    self.comparisons += 1
                    if arr[j - gap] > temp:
                        arr[j] = arr[j - gap]
                        self.swaps += 1
                        j -= gap
                    else:
                        break
                
                arr[j] = temp
            
            gap //= 2
        
        return arr
    
    def cocktail_sort(self, arr: List[int]) -> List[int]:
        """Cocktail Sort (Bidirectional Bubble Sort)
        Time: O(n²), Space: O(1)
        """
        self.reset_counters()
        n = len(arr)
        swapped = True
        start = 0
        end = n - 1
        
        while swapped:
            swapped = False
            
            # Forward pass
            for i in range(start, end):
                self.comparisons += 1
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    self.swaps += 1
                    swapped = True
            
            if not swapped:
                break
            
            end -= 1
            swapped = False
            
            # Backward pass
            for i in range(end - 1, start - 1, -1):
                self.comparisons += 1
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    self.swaps += 1
                    swapped = True
            
            start += 1
        
        return arr
    
    # ==========================================
    # 5. SORTING ANALYSIS AND UTILITIES
    # ==========================================
    
    def is_sorted(self, arr: List[int]) -> bool:
        """Check if array is sorted"""
        return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))
    
    def benchmark_sorts(self, arr: List[int]) -> dict:
        """Benchmark different sorting algorithms"""
        algorithms = {
            'Bubble Sort': self.bubble_sort,
            'Selection Sort': self.selection_sort,
            'Insertion Sort': self.insertion_sort,
            'Merge Sort': self.merge_sort,
            'Quick Sort': self.quick_sort,
            'Heap Sort': self.heap_sort,
            'Shell Sort': self.shell_sort,
            'Cocktail Sort': self.cocktail_sort
        }
        
        results = {}
        
        for name, algorithm in algorithms.items():
            arr_copy = arr.copy()
            start_time = time.time()
            
            try:
                algorithm(arr_copy)
                end_time = time.time()
                
                results[name] = {
                    'time': end_time - start_time,
                    'comparisons': self.comparisons,
                    'swaps': self.swaps,
                    'sorted': self.is_sorted(arr_copy)
                }
            except Exception as e:
                results[name] = {'error': str(e)}
        
        return results
    
    def generate_test_arrays(self) -> dict:
        """Generate different types of test arrays"""
        return {
            'random': [random.randint(1, 100) for _ in range(20)],
            'sorted': list(range(1, 21)),
            'reverse_sorted': list(range(20, 0, -1)),
            'nearly_sorted': [1, 2, 3, 5, 4, 6, 7, 8, 9, 10],
            'duplicates': [5, 2, 8, 2, 9, 1, 5, 5, 3, 8],
            'single_element': [42],
            'empty': []
        }

# Test Examples and Benchmarking
def run_examples():
    sorter = ArraySortingAlgorithms()
    
    print("=== ARRAY SORTING ALGORITHMS EXAMPLES ===\n")
    
    # Test array
    test_arr = [64, 34, 25, 12, 22, 11, 90, 88, 76, 50, 42]
    print(f"Original array: {test_arr}\n")
    
    # Test different sorting algorithms
    algorithms_to_test = [
        ('Bubble Sort', sorter.bubble_sort),
        ('Selection Sort', sorter.selection_sort),
        ('Insertion Sort', sorter.insertion_sort),
        ('Merge Sort', sorter.merge_sort),
        ('Quick Sort', sorter.quick_sort),
        ('Heap Sort', sorter.heap_sort),
        ('Shell Sort', sorter.shell_sort),
        ('Cocktail Sort', sorter.cocktail_sort)
    ]
    
    print("1. COMPARISON-BASED SORTING:")
    for name, algorithm in algorithms_to_test:
        arr_copy = test_arr.copy()
        result = algorithm(arr_copy)
        print(f"{name:15}: {result}")
        print(f"{'':15}  Comparisons: {sorter.comparisons}, Swaps: {sorter.swaps}")
    
    # Test linear time sorting
    print("\n2. LINEAR TIME SORTING:")
    
    # Counting sort
    counting_arr = [4, 2, 2, 8, 3, 3, 1]
    print(f"Counting Sort:   {counting_arr} -> {sorter.counting_sort(counting_arr)}")
    
    # Radix sort
    radix_arr = [170, 45, 75, 90, 2, 802, 24, 66]
    print(f"Radix Sort:      {radix_arr}")
    print(f"                 -> {sorter.radix_sort(radix_arr)}")
    
    # Bucket sort
    bucket_arr = [0.897, 0.565, 0.656, 0.1234, 0.665, 0.3434]
    print(f"Bucket Sort:     {bucket_arr}")
    print(f"                 -> {sorter.bucket_sort(bucket_arr)}")
    
    # Benchmark
    print("\n3. PERFORMANCE BENCHMARK:")
    benchmark_arr = [random.randint(1, 100) for _ in range(50)]
    results = sorter.benchmark_sorts(benchmark_arr)
    
    print(f"{'Algorithm':<15} {'Time (s)':<10} {'Comparisons':<12} {'Swaps':<8} {'Correct'}")
    print("-" * 60)
    
    for name, stats in results.items():
        if 'error' not in stats:
            print(f"{name:<15} {stats['time']:<10.6f} {stats['comparisons']:<12} "
                  f"{stats['swaps']:<8} {stats['sorted']}")

if __name__ == "__main__":
    run_examples() 