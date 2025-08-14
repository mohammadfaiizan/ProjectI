"""
Heap Sort and Heapify Algorithms
================================

Topics: Heap sort implementation, heapify algorithms, sorting analysis
Companies: Google, Amazon, Microsoft, Facebook, Apple
Difficulty: Medium to Hard
Time Complexity: O(n log n) for heap sort, O(n) for build heap
Space Complexity: O(1) in-place sorting
"""

from typing import List, Callable, Optional, Any
import time
import random

class HeapSortAlgorithms:
    
    def __init__(self):
        """Initialize with algorithm tracking"""
        self.comparison_count = 0
        self.swap_count = 0
        self.operation_count = 0
    
    # ==========================================
    # 1. HEAPIFY ALGORITHMS
    # ==========================================
    
    def max_heapify(self, arr: List[int], n: int, i: int, verbose: bool = True) -> None:
        """
        Max heapify algorithm - ensure subtree rooted at i satisfies max heap property
        
        Company: Core algorithm for heap sort
        Difficulty: Medium
        Time: O(log n), Space: O(1) iterative, O(log n) recursive
        
        Args:
            arr: Array to heapify
            n: Size of heap
            i: Root index of subtree to heapify
            verbose: Print detailed steps
        """
        largest = i  # Initialize largest as root
        left = 2 * i + 1  # Left child
        right = 2 * i + 2  # Right child
        
        if verbose:
            print(f"      Heapifying subtree rooted at index {i} (value: {arr[i]})")
            print(f"        Left child: {left} ({'exists' if left < n else 'none'})")
            print(f"        Right child: {right} ({'exists' if right < n else 'none'})")
        
        # Check if left child exists and is greater than root
        if left < n:
            self.comparison_count += 1
            if arr[left] > arr[largest]:
                largest = left
                if verbose:
                    print(f"        Left child {arr[left]} > current largest {arr[i]}")
        
        # Check if right child exists and is greater than current largest
        if right < n:
            self.comparison_count += 1
            if arr[right] > arr[largest]:
                largest = right
                if verbose:
                    print(f"        Right child {arr[right]} > current largest")
        
        # If largest is not root, swap and continue heapifying
        if largest != i:
            if verbose:
                print(f"        Swapping {arr[i]} ↔ {arr[largest]} (indices {i} ↔ {largest})")
            
            arr[i], arr[largest] = arr[largest], arr[i]
            self.swap_count += 1
            
            if verbose:
                print(f"        Array after swap: {arr}")
            
            # Recursively heapify the affected subtree
            self.max_heapify(arr, n, largest, verbose)
        else:
            if verbose:
                print(f"        No swap needed - heap property satisfied")
    
    def min_heapify(self, arr: List[int], n: int, i: int, verbose: bool = True) -> None:
        """
        Min heapify algorithm - ensure subtree rooted at i satisfies min heap property
        
        Similar to max_heapify but finds minimum instead of maximum
        """
        smallest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        if verbose:
            print(f"      Min heapifying subtree rooted at index {i} (value: {arr[i]})")
        
        # Check if left child exists and is smaller than root
        if left < n:
            self.comparison_count += 1
            if arr[left] < arr[smallest]:
                smallest = left
                if verbose:
                    print(f"        Left child {arr[left]} < current smallest {arr[i]}")
        
        # Check if right child exists and is smaller than current smallest
        if right < n:
            self.comparison_count += 1
            if arr[right] < arr[smallest]:
                smallest = right
                if verbose:
                    print(f"        Right child {arr[right]} < current smallest")
        
        # If smallest is not root, swap and continue heapifying
        if smallest != i:
            if verbose:
                print(f"        Swapping {arr[i]} ↔ {arr[smallest]} (indices {i} ↔ {smallest})")
            
            arr[i], arr[smallest] = arr[smallest], arr[i]
            self.swap_count += 1
            
            if verbose:
                print(f"        Array after swap: {arr}")
            
            # Recursively heapify the affected subtree
            self.min_heapify(arr, n, smallest, verbose)
        else:
            if verbose:
                print(f"        No swap needed - min heap property satisfied")
    
    def build_max_heap(self, arr: List[int], verbose: bool = True) -> None:
        """
        Build max heap from arbitrary array in O(n) time
        
        Algorithm: Start from last non-leaf node and heapify each node
        
        Company: Foundation for heap sort
        Difficulty: Medium
        Time: O(n), Space: O(1)
        """
        n = len(arr)
        
        if verbose:
            print(f"Building max heap from array: {arr}")
            print(f"Array size: {n}")
        
        # Start from last non-leaf node (parent of last element)
        last_non_leaf = (n // 2) - 1
        
        if verbose:
            print(f"Last non-leaf node index: {last_non_leaf}")
            print(f"Starting heapify from index {last_non_leaf} down to 0")
            print()
        
        # Heapify each node from last non-leaf to root
        for i in range(last_non_leaf, -1, -1):
            if verbose:
                print(f"Step {last_non_leaf - i + 1}: Heapifying node at index {i}")
            
            self.max_heapify(arr, n, i, verbose)
            
            if verbose:
                print(f"    Array after heapifying index {i}: {arr}")
                print()
        
        if verbose:
            print(f"Max heap construction complete: {arr}")
    
    def build_min_heap(self, arr: List[int], verbose: bool = True) -> None:
        """Build min heap from arbitrary array"""
        n = len(arr)
        
        if verbose:
            print(f"Building min heap from array: {arr}")
        
        # Start from last non-leaf node
        for i in range((n // 2) - 1, -1, -1):
            if verbose:
                print(f"Min heapifying node at index {i}")
            
            self.min_heapify(arr, n, i, verbose)
            
            if verbose:
                print(f"    Array after min heapifying index {i}: {arr}")
        
        if verbose:
            print(f"Min heap construction complete: {arr}")
    
    # ==========================================
    # 2. HEAP SORT IMPLEMENTATION
    # ==========================================
    
    def heap_sort(self, arr: List[int], verbose: bool = True) -> List[int]:
        """
        Heap sort algorithm - sort array using heap data structure
        
        Company: Google, Amazon, Microsoft (fundamental sorting algorithm)
        Difficulty: Medium
        Time: O(n log n), Space: O(1) - in-place sorting
        
        Algorithm:
        1. Build max heap from array - O(n)
        2. Extract max element n times - O(n log n)
        
        Returns sorted array in ascending order
        """
        if not arr:
            return arr
        
        # Reset counters
        self.comparison_count = 0
        self.swap_count = 0
        
        result = arr[:]  # Create copy to avoid modifying original
        n = len(result)
        
        if verbose:
            print("=== HEAP SORT ALGORITHM ===")
            print(f"Input array: {arr}")
            print(f"Array size: {n}")
            print()
        
        # Phase 1: Build max heap
        if verbose:
            print("PHASE 1: Build Max Heap")
            print("-" * 30)
        
        self.build_max_heap(result, verbose)
        
        if verbose:
            print(f"\nMax heap built: {result}")
            print(f"Largest element (root): {result[0]}")
            print()
        
        # Phase 2: Extract elements one by one
        if verbose:
            print("PHASE 2: Extract Elements (Sort)")
            print("-" * 35)
        
        # Extract elements from heap one by one
        for i in range(n - 1, 0, -1):
            if verbose:
                print(f"Extraction {n - i}: Heap size {i + 1}")
                print(f"    Current heap: {result[:i+1]}")
                print(f"    Sorted portion: {result[i+1:]}")
            
            # Move current root (maximum) to end
            if verbose:
                print(f"    Swapping root {result[0]} with last element {result[i]}")
            
            result[0], result[i] = result[i], result[0]
            self.swap_count += 1
            
            if verbose:
                print(f"    After swap: {result}")
                print(f"    New sorted portion: {result[i:]}")
            
            # Reduce heap size and heapify root
            if verbose:
                print(f"    Heapifying reduced heap of size {i}")
            
            self.max_heapify(result, i, 0, verbose)
            
            if verbose:
                print(f"    Heap after heapify: {result[:i]}")
                print()
        
        if verbose:
            print("HEAP SORT COMPLETE!")
            print(f"Final sorted array: {result}")
            print(f"Total comparisons: {self.comparison_count}")
            print(f"Total swaps: {self.swap_count}")
        
        return result
    
    def heap_sort_descending(self, arr: List[int], verbose: bool = True) -> List[int]:
        """
        Heap sort for descending order using min heap
        
        Returns sorted array in descending order
        """
        if not arr:
            return arr
        
        result = arr[:]
        n = len(result)
        
        if verbose:
            print("=== HEAP SORT (DESCENDING) ===")
            print(f"Input array: {arr}")
        
        # Build min heap
        if verbose:
            print("\nBuilding min heap for descending sort:")
        
        self.build_min_heap(result, verbose)
        
        # Extract elements (smallest first, building descending order)
        for i in range(n - 1, 0, -1):
            if verbose:
                print(f"\nExtracting minimum {result[0]} to position {i}")
            
            # Move current root (minimum) to end
            result[0], result[i] = result[i], result[0]
            self.swap_count += 1
            
            # Heapify reduced min heap
            self.min_heapify(result, i, 0, verbose)
            
            if verbose:
                print(f"Current state: {result}")
        
        if verbose:
            print(f"\nFinal descending sorted array: {result}")
        
        return result
    
    # ==========================================
    # 3. HEAP SORT VARIATIONS
    # ==========================================
    
    def iterative_heapify(self, arr: List[int], n: int, i: int) -> None:
        """
        Iterative version of heapify (avoids recursion stack)
        
        Better for very large heaps to avoid stack overflow
        """
        while True:
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            # Find largest among node and its children
            if left < n and arr[left] > arr[largest]:
                largest = left
            
            if right < n and arr[right] > arr[largest]:
                largest = right
            
            # If largest is not current node, swap and continue
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                i = largest  # Move down to swapped position
            else:
                break  # Heap property satisfied
    
    def k_largest_heap_sort(self, arr: List[int], k: int) -> List[int]:
        """
        Find k largest elements using heap sort concept
        
        Company: Amazon, Google
        Difficulty: Medium
        Time: O(n + k log n), Space: O(1)
        
        More efficient than full sort when k << n
        """
        if not arr or k <= 0:
            return []
        
        result = arr[:]
        n = len(result)
        k = min(k, n)  # Ensure k doesn't exceed array size
        
        print(f"Finding {k} largest elements from: {arr}")
        
        # Build max heap
        self.build_max_heap(result, verbose=False)
        print(f"Max heap built: {result}")
        
        largest_elements = []
        
        # Extract k largest elements
        for i in range(k):
            # Extract max (root)
            largest_elements.append(result[0])
            
            # Move last element to root and reduce heap size
            result[0] = result[n - 1 - i]
            
            # Heapify reduced heap
            self.max_heapify(result, n - 1 - i, 0, verbose=False)
            
            print(f"Extracted {largest_elements[-1]}, remaining heap size: {n - 1 - i}")
        
        print(f"K largest elements: {largest_elements}")
        return largest_elements
    
    # ==========================================
    # 4. HEAP SORT ANALYSIS AND COMPARISON
    # ==========================================
    
    def compare_sorting_algorithms(self, arr: List[int]) -> None:
        """
        Compare heap sort with other sorting algorithms
        
        Measures time complexity and other metrics
        """
        import copy
        
        print("=== SORTING ALGORITHMS COMPARISON ===")
        print(f"Array size: {len(arr)}")
        print(f"Sample array: {arr[:10]}{'...' if len(arr) > 10 else ''}")
        print()
        
        algorithms = {
            'Heap Sort': self.heap_sort,
            'Quick Sort': self.quick_sort,
            'Merge Sort': self.merge_sort,
            'Bubble Sort': self.bubble_sort
        }
        
        results = {}
        
        for name, algorithm in algorithms.items():
            print(f"Testing {name}:")
            
            # Reset counters
            self.comparison_count = 0
            self.swap_count = 0
            
            # Create copy of array
            test_arr = copy.deepcopy(arr)
            
            # Measure time
            start_time = time.time()
            sorted_arr = algorithm(test_arr, verbose=False)
            end_time = time.time()
            
            # Store results
            results[name] = {
                'time': end_time - start_time,
                'comparisons': self.comparison_count,
                'swaps': self.swap_count,
                'is_sorted': sorted_arr == sorted(arr)
            }
            
            print(f"    Time: {results[name]['time']:.6f} seconds")
            print(f"    Comparisons: {results[name]['comparisons']}")
            print(f"    Swaps: {results[name]['swaps']}")
            print(f"    Correctly sorted: {results[name]['is_sorted']}")
            print()
        
        # Print summary
        print("PERFORMANCE SUMMARY:")
        print("┌─────────────┬───────────┬─────────────┬─────────┬──────────┐")
        print("│ Algorithm   │ Time (s)  │ Comparisons │ Swaps   │ Correct  │")
        print("├─────────────┼───────────┼─────────────┼─────────┼──────────┤")
        
        for name, metrics in results.items():
            print(f"│ {name:<11} │ {metrics['time']:<9.6f} │ {metrics['comparisons']:<11} │ {metrics['swaps']:<7} │ {metrics['is_sorted']:<8} │")
        
        print("└─────────────┴───────────┴─────────────┴─────────┴──────────┘")
    
    def analyze_heap_sort_complexity(self, sizes: List[int]) -> None:
        """
        Analyze heap sort complexity empirically
        
        Test with different input sizes to verify O(n log n) complexity
        """
        print("=== HEAP SORT COMPLEXITY ANALYSIS ===")
        print()
        
        results = []
        
        for size in sizes:
            # Generate random array
            arr = [random.randint(1, 1000) for _ in range(size)]
            
            # Reset counters
            self.comparison_count = 0
            self.swap_count = 0
            
            # Time the sort
            start_time = time.time()
            self.heap_sort(arr, verbose=False)
            end_time = time.time()
            
            # Theoretical complexity: n * log(n)
            theoretical_ops = size * math.log2(size) if size > 0 else 0
            
            results.append({
                'size': size,
                'time': end_time - start_time,
                'comparisons': self.comparison_count,
                'theoretical': theoretical_ops
            })
            
            print(f"Size: {size:>6}, Time: {end_time - start_time:.6f}s, "
                  f"Comparisons: {self.comparison_count:>8}, "
                  f"Theoretical O(n log n): {theoretical_ops:.0f}")
        
        print()
        print("Complexity Verification:")
        print("• If algorithm is O(n log n), comparisons should grow proportionally to n log n")
        print("• Actual vs theoretical ratio should remain roughly constant")
        
        if len(results) > 1:
            for i in range(1, len(results)):
                prev = results[i-1]
                curr = results[i]
                
                actual_ratio = curr['comparisons'] / prev['comparisons'] if prev['comparisons'] > 0 else 0
                theoretical_ratio = curr['theoretical'] / prev['theoretical'] if prev['theoretical'] > 0 else 0
                
                print(f"Size {prev['size']} → {curr['size']}: "
                      f"Actual ratio: {actual_ratio:.2f}, "
                      f"Theoretical ratio: {theoretical_ratio:.2f}")
    
    # ==========================================
    # 5. HELPER SORTING ALGORITHMS FOR COMPARISON
    # ==========================================
    
    def quick_sort(self, arr: List[int], verbose: bool = False) -> List[int]:
        """Quick sort implementation for comparison"""
        if len(arr) <= 1:
            return arr
        
        def partition(arr, low, high):
            pivot = arr[high]
            i = low - 1
            
            for j in range(low, high):
                self.comparison_count += 1
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
                    self.swap_count += 1
            
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            self.swap_count += 1
            return i + 1
        
        def quick_sort_helper(arr, low, high):
            if low < high:
                pi = partition(arr, low, high)
                quick_sort_helper(arr, low, pi - 1)
                quick_sort_helper(arr, pi + 1, high)
        
        result = arr[:]
        quick_sort_helper(result, 0, len(result) - 1)
        return result
    
    def merge_sort(self, arr: List[int], verbose: bool = False) -> List[int]:
        """Merge sort implementation for comparison"""
        if len(arr) <= 1:
            return arr
        
        def merge(left, right):
            result = []
            i = j = 0
            
            while i < len(left) and j < len(right):
                self.comparison_count += 1
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
        left = self.merge_sort(arr[:mid], verbose)
        right = self.merge_sort(arr[mid:], verbose)
        
        return merge(left, right)
    
    def bubble_sort(self, arr: List[int], verbose: bool = False) -> List[int]:
        """Bubble sort implementation for comparison"""
        result = arr[:]
        n = len(result)
        
        for i in range(n):
            for j in range(0, n - i - 1):
                self.comparison_count += 1
                if result[j] > result[j + 1]:
                    result[j], result[j + 1] = result[j + 1], result[j]
                    self.swap_count += 1
        
        return result


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_heap_sort_algorithms():
    """Demonstrate all heap sort algorithms and concepts"""
    print("=== HEAP SORT ALGORITHMS DEMONSTRATION ===\n")
    
    algorithms = HeapSortAlgorithms()
    
    # 1. Basic heapify operations
    print("=== HEAPIFY OPERATIONS ===")
    
    print("1. Max Heapify Example:")
    # Array that violates heap property at root
    test_array = [4, 10, 3, 5, 1]
    print(f"Original array (heap property violated): {test_array}")
    algorithms.max_heapify(test_array, len(test_array), 0)
    print(f"After max heapify: {test_array}")
    print()
    
    print("2. Build Max Heap:")
    build_array = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
    print(f"Building max heap from: {build_array}")
    algorithms.build_max_heap(build_array)
    print()
    
    print("="*60 + "\n")
    
    # 2. Complete heap sort demonstration
    print("=== HEAP SORT DEMONSTRATION ===")
    
    print("1. Basic Heap Sort:")
    unsorted = [12, 11, 13, 5, 6, 7]
    print(f"Sorting array: {unsorted}")
    sorted_array = algorithms.heap_sort(unsorted)
    print()
    
    print("2. Heap Sort (Descending):")
    algorithms.heap_sort_descending([64, 34, 25, 12, 22, 11, 90])
    print()
    
    print("="*60 + "\n")
    
    # 3. K largest elements
    print("=== K LARGEST ELEMENTS ===")
    large_array = [3, 2, 1, 5, 6, 4, 7, 9, 8]
    algorithms.k_largest_heap_sort(large_array, 3)
    print()
    
    print("="*60 + "\n")
    
    # 4. Algorithm comparison
    print("=== SORTING ALGORITHMS COMPARISON ===")
    comparison_array = [64, 34, 25, 12, 22, 11, 90, 5, 77, 30]
    algorithms.compare_sorting_algorithms(comparison_array)
    print()
    
    # 5. Complexity analysis
    print("=== COMPLEXITY ANALYSIS ===")
    test_sizes = [100, 200, 500, 1000]
    algorithms.analyze_heap_sort_complexity(test_sizes)


if __name__ == "__main__":
    import math
    demonstrate_heap_sort_algorithms()
    
    print("\n=== HEAP SORT MASTERY GUIDE ===")
    
    print("\n🎯 HEAP SORT KEY CONCEPTS:")
    print("• Two-phase algorithm: Build heap + Extract elements")
    print("• In-place sorting with O(1) space complexity")
    print("• Guaranteed O(n log n) time complexity")
    print("• Not stable (doesn't preserve relative order of equal elements)")
    print("• No best/worst case variation - always O(n log n)")
    
    print("\n📊 COMPLEXITY ANALYSIS:")
    print("• Build Heap Phase: O(n) time")
    print("• Extract Phase: O(n log n) time")
    print("• Overall: O(n log n) time, O(1) space")
    print("• Comparisons: ~2n log n")
    print("• Swaps: ~n log n")
    
    print("\n⚡ HEAP SORT ADVANTAGES:")
    print("• Guaranteed O(n log n) performance")
    print("• In-place sorting (O(1) extra space)")
    print("• No quadratic worst-case behavior")
    print("• Good cache performance due to array-based heap")
    print("• Can find k largest/smallest efficiently")
    
    print("\n⚠️ HEAP SORT DISADVANTAGES:")
    print("• Not stable sorting algorithm")
    print("• Generally slower than quicksort in practice")
    print("• More complex than simple algorithms")
    print("• Poor performance on small arrays")
    print("• Not adaptive (doesn't benefit from partially sorted input)")
    
    print("\n🔧 WHEN TO USE HEAP SORT:")
    print("• Need guaranteed O(n log n) performance")
    print("• Memory is severely constrained (O(1) space)")
    print("• Want to avoid quicksort's worst-case O(n²)")
    print("• Finding k largest/smallest elements")
    print("• Embedded systems with strict memory limits")
    
    print("\n🏆 HEAP SORT VS OTHER ALGORITHMS:")
    print("• vs Quick Sort: Guaranteed O(n log n) but slower average case")
    print("• vs Merge Sort: In-place O(1) space vs O(n) space")
    print("• vs Insertion Sort: Better for large arrays")
    print("• vs Selection Sort: Much better time complexity")
    
    print("\n📚 HEAP SORT APPLICATIONS:")
    print("• Priority queue implementation")
    print("• Finding k largest/smallest elements")
    print("• External sorting for large datasets")
    print("• Embedded systems with memory constraints")
    print("• Real-time systems needing predictable performance")
    
    print("\n🎓 IMPLEMENTATION TIPS:")
    print("• Use 0-based indexing: parent=(i-1)/2, children=2i+1, 2i+2")
    print("• Consider iterative heapify for very large datasets")
    print("• Build heap bottom-up for O(n) construction")
    print("• Handle edge cases: empty arrays, single elements")
    print("• Optimize for specific use cases (k largest, etc.)")
