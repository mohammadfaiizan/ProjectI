"""
Sorting Algorithms Analysis and Complexity
==========================================

Topics: Time/Space complexity, stability, performance analysis
Companies: All major tech companies
Difficulty: Educational/Analysis
"""

import time
import random
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

class SortingAnalysisAndComplexity:
    
    def __init__(self):
        self.comparison_count = 0
        self.swap_count = 0
    
    def reset_counters(self):
        """Reset comparison and swap counters"""
        self.comparison_count = 0
        self.swap_count = 0
    
    # ==========================================
    # INSTRUMENTED SORTING ALGORITHMS
    # ==========================================
    
    def bubble_sort_instrumented(self, arr: List[int]) -> List[int]:
        """Bubble sort with performance counters"""
        arr = arr.copy()
        n = len(arr)
        
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                self.comparison_count += 1
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    self.swap_count += 1
                    swapped = True
            
            if not swapped:
                break
        
        return arr
    
    def selection_sort_instrumented(self, arr: List[int]) -> List[int]:
        """Selection sort with performance counters"""
        arr = arr.copy()
        n = len(arr)
        
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                self.comparison_count += 1
                if arr[j] < arr[min_idx]:
                    min_idx = j
            
            if min_idx != i:
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
                self.swap_count += 1
        
        return arr
    
    def insertion_sort_instrumented(self, arr: List[int]) -> List[int]:
        """Insertion sort with performance counters"""
        arr = arr.copy()
        
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            
            while j >= 0:
                self.comparison_count += 1
                if arr[j] > key:
                    arr[j + 1] = arr[j]
                    self.swap_count += 1
                    j -= 1
                else:
                    break
            
            arr[j + 1] = key
        
        return arr
    
    def merge_sort_instrumented(self, arr: List[int]) -> List[int]:
        """Merge sort with performance counters"""
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = self.merge_sort_instrumented(arr[:mid])
        right = self.merge_sort_instrumented(arr[mid:])
        
        return self._merge_instrumented(left, right)
    
    def _merge_instrumented(self, left: List[int], right: List[int]) -> List[int]:
        """Merge operation with counters"""
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
    
    def quick_sort_instrumented(self, arr: List[int]) -> List[int]:
        """Quick sort with performance counters"""
        arr = arr.copy()
        self._quick_sort_helper_instrumented(arr, 0, len(arr) - 1)
        return arr
    
    def _quick_sort_helper_instrumented(self, arr: List[int], low: int, high: int):
        """Quick sort helper with counters"""
        if low < high:
            pivot_index = self._partition_instrumented(arr, low, high)
            self._quick_sort_helper_instrumented(arr, low, pivot_index - 1)
            self._quick_sort_helper_instrumented(arr, pivot_index + 1, high)
    
    def _partition_instrumented(self, arr: List[int], low: int, high: int) -> int:
        """Partition with counters"""
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            self.comparison_count += 1
            if arr[j] <= pivot:
                i += 1
                if i != j:
                    arr[i], arr[j] = arr[j], arr[i]
                    self.swap_count += 1
        
        if i + 1 != high:
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            self.swap_count += 1
        
        return i + 1
    
    # ==========================================
    # COMPLEXITY ANALYSIS
    # ==========================================
    
    def get_sorting_complexities(self) -> Dict[str, Dict[str, str]]:
        """Get theoretical complexities of sorting algorithms"""
        return {
            "Bubble Sort": {
                "Best": "O(n)",
                "Average": "O(n²)",
                "Worst": "O(n²)",
                "Space": "O(1)",
                "Stable": "Yes"
            },
            "Selection Sort": {
                "Best": "O(n²)",
                "Average": "O(n²)",
                "Worst": "O(n²)",
                "Space": "O(1)",
                "Stable": "No"
            },
            "Insertion Sort": {
                "Best": "O(n)",
                "Average": "O(n²)",
                "Worst": "O(n²)",
                "Space": "O(1)",
                "Stable": "Yes"
            },
            "Merge Sort": {
                "Best": "O(n log n)",
                "Average": "O(n log n)",
                "Worst": "O(n log n)",
                "Space": "O(n)",
                "Stable": "Yes"
            },
            "Quick Sort": {
                "Best": "O(n log n)",
                "Average": "O(n log n)",
                "Worst": "O(n²)",
                "Space": "O(log n)",
                "Stable": "No"
            },
            "Heap Sort": {
                "Best": "O(n log n)",
                "Average": "O(n log n)",
                "Worst": "O(n log n)",
                "Space": "O(1)",
                "Stable": "No"
            },
            "Counting Sort": {
                "Best": "O(n + k)",
                "Average": "O(n + k)",
                "Worst": "O(n + k)",
                "Space": "O(k)",
                "Stable": "Yes"
            },
            "Radix Sort": {
                "Best": "O(d(n + k))",
                "Average": "O(d(n + k))",
                "Worst": "O(d(n + k))",
                "Space": "O(n + k)",
                "Stable": "Yes"
            }
        }
    
    def analyze_stability(self, algorithm_name: str, arr: List[Tuple[int, str]]) -> bool:
        """Test if sorting algorithm is stable"""
        # Create array with duplicate keys but different values
        test_arr = [(3, 'a'), (1, 'x'), (3, 'b'), (2, 'y'), (1, 'z')]
        
        # Sort using Python's stable sort for comparison
        expected = sorted(test_arr, key=lambda x: x[0])
        
        # For this demo, we'll just return the theoretical stability
        complexities = self.get_sorting_complexities()
        return complexities.get(algorithm_name, {}).get("Stable", "Unknown") == "Yes"
    
    # ==========================================
    # PERFORMANCE BENCHMARKING
    # ==========================================
    
    def benchmark_algorithm(self, sort_func, arr: List[int]) -> Dict[str, float]:
        """Benchmark a sorting algorithm"""
        self.reset_counters()
        
        start_time = time.time()
        sorted_arr = sort_func(arr)
        end_time = time.time()
        
        return {
            "time": end_time - start_time,
            "comparisons": self.comparison_count,
            "swaps": self.swap_count,
            "is_sorted": sorted_arr == sorted(arr)
        }
    
    def compare_algorithms(self, sizes: List[int]) -> Dict[str, List[float]]:
        """Compare multiple algorithms across different input sizes"""
        algorithms = {
            "Bubble Sort": self.bubble_sort_instrumented,
            "Selection Sort": self.selection_sort_instrumented,
            "Insertion Sort": self.insertion_sort_instrumented,
            "Merge Sort": self.merge_sort_instrumented,
            "Quick Sort": self.quick_sort_instrumented
        }
        
        results = {name: [] for name in algorithms}
        
        for size in sizes:
            # Generate random array
            arr = [random.randint(1, 1000) for _ in range(size)]
            
            for name, func in algorithms.items():
                benchmark = self.benchmark_algorithm(func, arr)
                results[name].append(benchmark["time"])
        
        return results
    
    def analyze_best_worst_cases(self) -> Dict[str, Dict[str, List[int]]]:
        """Generate best and worst case inputs for different algorithms"""
        size = 100
        
        return {
            "Bubble Sort": {
                "best": list(range(size)),  # Already sorted
                "worst": list(range(size, 0, -1))  # Reverse sorted
            },
            "Insertion Sort": {
                "best": list(range(size)),  # Already sorted
                "worst": list(range(size, 0, -1))  # Reverse sorted
            },
            "Quick Sort": {
                "best": [random.randint(1, 1000) for _ in range(size)],  # Random
                "worst": list(range(size))  # Already sorted (with first pivot)
            },
            "Merge Sort": {
                "best": [random.randint(1, 1000) for _ in range(size)],  # Consistent
                "worst": [random.randint(1, 1000) for _ in range(size)]  # Consistent
            }
        }
    
    # ==========================================
    # PRACTICAL RECOMMENDATIONS
    # ==========================================
    
    def get_algorithm_recommendations(self) -> Dict[str, str]:
        """Get practical recommendations for algorithm selection"""
        return {
            "Small arrays (n < 50)": "Insertion Sort - Simple and efficient for small data",
            "Nearly sorted data": "Insertion Sort or Bubble Sort - O(n) best case",
            "General purpose": "Merge Sort or Quick Sort - O(n log n) average",
            "Memory constrained": "Heap Sort or Quick Sort - O(1) or O(log n) space",
            "Stability required": "Merge Sort or Stable Sort variants",
            "Integer data with small range": "Counting Sort - O(n + k) linear time",
            "Large datasets": "External Merge Sort - For data that doesn't fit in memory",
            "Parallel processing": "Merge Sort - Easy to parallelize",
            "Cache-friendly": "Heap Sort - Good locality of reference"
        }
    
    def adaptive_sort_selector(self, arr: List[int]) -> str:
        """Select best sorting algorithm based on input characteristics"""
        n = len(arr)
        
        # Check if already sorted or nearly sorted
        inversions = sum(1 for i in range(n-1) if arr[i] > arr[i+1])
        is_nearly_sorted = inversions < n // 10
        
        # Check range of values
        if n > 0:
            value_range = max(arr) - min(arr) + 1
            has_small_range = value_range <= n * 2
        else:
            has_small_range = False
        
        # Select algorithm
        if n <= 10:
            return "Insertion Sort"
        elif is_nearly_sorted:
            return "Insertion Sort"
        elif has_small_range and n > 100:
            return "Counting Sort"
        elif n > 1000:
            return "Merge Sort"
        else:
            return "Quick Sort"

# Test Examples and Analysis
def run_analysis():
    analyzer = SortingAnalysisAndComplexity()
    
    print("=== SORTING ALGORITHMS ANALYSIS ===\n")
    
    # Display complexities
    print("1. THEORETICAL COMPLEXITIES:")
    complexities = analyzer.get_sorting_complexities()
    for alg, complexity in complexities.items():
        print(f"\n{alg}:")
        for case, value in complexity.items():
            print(f"  {case}: {value}")
    
    # Performance comparison
    print("\n2. PERFORMANCE ANALYSIS:")
    test_arr = [random.randint(1, 100) for _ in range(50)]
    
    algorithms = [
        ("Bubble Sort", analyzer.bubble_sort_instrumented),
        ("Selection Sort", analyzer.selection_sort_instrumented),
        ("Insertion Sort", analyzer.insertion_sort_instrumented),
        ("Merge Sort", analyzer.merge_sort_instrumented),
        ("Quick Sort", analyzer.quick_sort_instrumented)
    ]
    
    for name, func in algorithms:
        result = analyzer.benchmark_algorithm(func, test_arr)
        print(f"{name}:")
        print(f"  Time: {result['time']:.6f}s")
        print(f"  Comparisons: {result['comparisons']}")
        print(f"  Swaps: {result['swaps']}")
    
    # Algorithm recommendations
    print("\n3. ALGORITHM RECOMMENDATIONS:")
    recommendations = analyzer.get_algorithm_recommendations()
    for situation, recommendation in recommendations.items():
        print(f"{situation}: {recommendation}")
    
    # Adaptive selection
    print("\n4. ADAPTIVE ALGORITHM SELECTION:")
    test_cases = [
        ([1, 2, 3, 4, 5], "Sorted array"),
        ([5, 4, 3, 2, 1], "Reverse sorted"),
        ([random.randint(1, 10) for _ in range(100)], "Small range"),
        ([random.randint(1, 1000) for _ in range(1000)], "Large random")
    ]
    
    for arr, description in test_cases:
        selected = analyzer.adaptive_sort_selector(arr)
        print(f"{description}: {selected}")

if __name__ == "__main__":
    run_analysis() 