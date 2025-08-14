"""
Sorting Algorithms Complexity Analysis
=====================================

Topics: Time/Space complexity, performance analysis, algorithm selection
Companies: All tech companies (theoretical knowledge)
Difficulty: Educational/Analysis
"""

import time
import random
from typing import List, Dict

class SortingComplexityAnalysis:
    
    def __init__(self):
        self.comparison_count = 0
        self.swap_count = 0
    
    def reset_counters(self):
        self.comparison_count = 0
        self.swap_count = 0
    
    # Instrumented sorting algorithms for analysis
    def bubble_sort_instrumented(self, arr: List[int]) -> List[int]:
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
    
    def insertion_sort_instrumented(self, arr: List[int]) -> List[int]:
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
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = self.merge_sort_instrumented(arr[:mid])
        right = self.merge_sort_instrumented(arr[mid:])
        
        return self._merge_instrumented(left, right)
    
    def _merge_instrumented(self, left: List[int], right: List[int]) -> List[int]:
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
    
    def get_sorting_complexities(self) -> Dict[str, Dict[str, str]]:
        """Theoretical complexities of sorting algorithms"""
        return {
            "Bubble Sort": {
                "Best": "O(n)", "Average": "O(n²)", "Worst": "O(n²)",
                "Space": "O(1)", "Stable": "Yes"
            },
            "Selection Sort": {
                "Best": "O(n²)", "Average": "O(n²)", "Worst": "O(n²)",
                "Space": "O(1)", "Stable": "No"
            },
            "Insertion Sort": {
                "Best": "O(n)", "Average": "O(n²)", "Worst": "O(n²)",
                "Space": "O(1)", "Stable": "Yes"
            },
            "Merge Sort": {
                "Best": "O(n log n)", "Average": "O(n log n)", "Worst": "O(n log n)",
                "Space": "O(n)", "Stable": "Yes"
            },
            "Quick Sort": {
                "Best": "O(n log n)", "Average": "O(n log n)", "Worst": "O(n²)",
                "Space": "O(log n)", "Stable": "No"
            },
            "Heap Sort": {
                "Best": "O(n log n)", "Average": "O(n log n)", "Worst": "O(n log n)",
                "Space": "O(1)", "Stable": "No"
            },
            "Counting Sort": {
                "Best": "O(n + k)", "Average": "O(n + k)", "Worst": "O(n + k)",
                "Space": "O(k)", "Stable": "Yes"
            },
            "Radix Sort": {
                "Best": "O(d(n + k))", "Average": "O(d(n + k))", "Worst": "O(d(n + k))",
                "Space": "O(n + k)", "Stable": "Yes"
            }
        }
    
    def benchmark_algorithm(self, sort_func, arr: List[int]) -> Dict[str, float]:
        """Benchmark sorting algorithm performance"""
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
    
    def get_algorithm_recommendations(self) -> Dict[str, str]:
        """Practical recommendations for algorithm selection"""
        return {
            "Small arrays (n < 50)": "Insertion Sort",
            "Nearly sorted data": "Insertion Sort or Bubble Sort",
            "General purpose": "Merge Sort or Quick Sort",
            "Memory constrained": "Heap Sort or Quick Sort",
            "Stability required": "Merge Sort",
            "Integer data, small range": "Counting Sort",
            "Large datasets": "External Merge Sort",
            "Parallel processing": "Merge Sort",
            "Cache-friendly": "Heap Sort"
        }
    
    def adaptive_sort_selector(self, arr: List[int]) -> str:
        """Select best algorithm based on input characteristics"""
        n = len(arr)
        
        if n <= 10:
            return "Insertion Sort"
        
        # Check if nearly sorted
        inversions = sum(1 for i in range(n-1) if arr[i] > arr[i+1])
        if inversions < n // 10:
            return "Insertion Sort"
        
        # Check value range
        if n > 0:
            value_range = max(arr) - min(arr) + 1
            if value_range <= n * 2 and n > 100:
                return "Counting Sort"
        
        return "Merge Sort" if n > 1000 else "Quick Sort"

# Test Examples
def run_examples():
    analyzer = SortingComplexityAnalysis()
    
    print("=== SORTING COMPLEXITY ANALYSIS ===\n")
    
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
        ("Insertion Sort", analyzer.insertion_sort_instrumented),
        ("Merge Sort", analyzer.merge_sort_instrumented)
    ]
    
    for name, func in algorithms:
        result = analyzer.benchmark_algorithm(func, test_arr)
        print(f"{name}: Time={result['time']:.6f}s, "
              f"Comparisons={result['comparisons']}, Swaps={result['swaps']}")
    
    # Recommendations
    print("\n3. ALGORITHM RECOMMENDATIONS:")
    recommendations = analyzer.get_algorithm_recommendations()
    for situation, recommendation in recommendations.items():
        print(f"{situation}: {recommendation}")

if __name__ == "__main__":
    run_examples() 