"""
Divide and Conquer with Recursion
=================================

Topics: Merge sort, quick sort, binary search, maximum subarray, closest pair
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix
Difficulty: Medium to Hard
Time Complexity: O(n log n) typical for divide and conquer
Space Complexity: O(log n) to O(n)
"""

from typing import List, Tuple, Optional
import random
import math

class DivideAndConquer:
    
    def __init__(self):
        """Initialize with comparison counting for analysis"""
        self.comparisons = 0
        self.recursive_calls = 0
    
    # ==========================================
    # 1. MERGE SORT - Classic Divide and Conquer
    # ==========================================
    
    def merge_sort(self, arr: List[int]) -> List[int]:
        """
        Sort array using merge sort (divide and conquer)
        
        Algorithm:
        1. Divide: Split array into two halves
        2. Conquer: Recursively sort both halves
        3. Combine: Merge the sorted halves
        
        Time: O(n log n), Space: O(n)
        """
        self.recursive_calls += 1
        
        # Base case: array with 0 or 1 element is already sorted
        if len(arr) <= 1:
            return arr[:]
        
        # Divide: split array into two halves
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]
        
        print(f"Dividing: {arr} → {left_half} | {right_half}")
        
        # Conquer: recursively sort both halves
        sorted_left = self.merge_sort(left_half)
        sorted_right = self.merge_sort(right_half)
        
        # Combine: merge the sorted halves
        merged = self.merge(sorted_left, sorted_right)
        print(f"Merging: {sorted_left} + {sorted_right} → {merged}")
        
        return merged
    
    def merge(self, left: List[int], right: List[int]) -> List[int]:
        """
        Merge two sorted arrays into one sorted array
        
        Time: O(n), Space: O(n)
        """
        merged = []
        i = j = 0
        
        # Compare elements from both arrays and add smaller one
        while i < len(left) and j < len(right):
            self.comparisons += 1
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        
        # Add remaining elements (one array might be exhausted)
        merged.extend(left[i:])
        merged.extend(right[j:])
        
        return merged
    
    # ==========================================
    # 2. QUICK SORT - Divide and Conquer
    # ==========================================
    
    def quick_sort(self, arr: List[int], low: int = 0, high: int = None) -> None:
        """
        Sort array in-place using quick sort
        
        Algorithm:
        1. Choose a pivot element
        2. Partition array around pivot
        3. Recursively sort elements before and after partition
        
        Time: O(n²) worst, O(n log n) average, Space: O(log n)
        """
        if high is None:
            high = len(arr) - 1
        
        self.recursive_calls += 1
        
        if low < high:
            # Partition and get pivot index
            pivot_index = self.partition(arr, low, high)
            
            print(f"Partitioned around {arr[pivot_index]}: {arr[low:pivot_index]} | [{arr[pivot_index]}] | {arr[pivot_index+1:high+1]}")
            
            # Recursively sort elements before and after partition
            self.quick_sort(arr, low, pivot_index - 1)
            self.quick_sort(arr, pivot_index + 1, high)
    
    def partition(self, arr: List[int], low: int, high: int) -> int:
        """
        Partition array around pivot (last element)
        Returns index of pivot in final sorted position
        """
        pivot = arr[high]  # Choose last element as pivot
        i = low - 1  # Index of smaller element
        
        for j in range(low, high):
            self.comparisons += 1
            # If current element is smaller than or equal to pivot
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        # Place pivot in correct position
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    # ==========================================
    # 3. BINARY SEARCH - Divide and Conquer
    # ==========================================
    
    def binary_search(self, arr: List[int], target: int, left: int = 0, right: int = None) -> int:
        """
        Search for target in sorted array using binary search
        
        Algorithm:
        1. Compare target with middle element
        2. If equal, return index
        3. If target is smaller, search left half
        4. If target is larger, search right half
        
        Time: O(log n), Space: O(log n)
        """
        if right is None:
            right = len(arr) - 1
        
        self.recursive_calls += 1
        self.comparisons += 1
        
        # Base case: element not found
        if left > right:
            return -1
        
        # Divide: find middle point
        mid = (left + right) // 2
        print(f"Searching in range [{left}, {right}], mid = {mid}, value = {arr[mid]}")
        
        # Base case: element found
        if arr[mid] == target:
            return mid
        
        # Conquer: search in appropriate half
        elif arr[mid] > target:
            return self.binary_search(arr, target, left, mid - 1)
        else:
            return self.binary_search(arr, target, mid + 1, right)
    
    def binary_search_iterative(self, arr: List[int], target: int) -> int:
        """
        Iterative version of binary search for comparison
        
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        
        return -1
    
    # ==========================================
    # 4. MAXIMUM SUBARRAY - Divide and Conquer
    # ==========================================
    
    def max_subarray_sum(self, arr: List[int], low: int = 0, high: int = None) -> Tuple[int, int, int]:
        """
        Find maximum subarray sum using divide and conquer
        
        Algorithm:
        1. Divide array into two halves
        2. Find max subarray in left half
        3. Find max subarray in right half
        4. Find max subarray crossing the middle
        5. Return maximum of the three
        
        Returns: (max_sum, start_index, end_index)
        Time: O(n log n), Space: O(log n)
        """
        if high is None:
            high = len(arr) - 1
        
        self.recursive_calls += 1
        
        # Base case: single element
        if low == high:
            return (arr[low], low, high)
        
        # Divide: find middle point
        mid = (low + high) // 2
        
        # Conquer: find max subarray in left and right halves
        left_sum, left_start, left_end = self.max_subarray_sum(arr, low, mid)
        right_sum, right_start, right_end = self.max_subarray_sum(arr, mid + 1, high)
        
        # Find max subarray crossing the middle
        cross_sum, cross_start, cross_end = self.max_crossing_subarray(arr, low, mid, high)
        
        # Return maximum of the three
        if left_sum >= right_sum and left_sum >= cross_sum:
            return (left_sum, left_start, left_end)
        elif right_sum >= left_sum and right_sum >= cross_sum:
            return (right_sum, right_start, right_end)
        else:
            return (cross_sum, cross_start, cross_end)
    
    def max_crossing_subarray(self, arr: List[int], low: int, mid: int, high: int) -> Tuple[int, int, int]:
        """
        Find maximum subarray that crosses the middle point
        """
        # Find max sum for left side
        left_sum = float('-inf')
        current_sum = 0
        max_left = mid
        
        for i in range(mid, low - 1, -1):
            current_sum += arr[i]
            if current_sum > left_sum:
                left_sum = current_sum
                max_left = i
        
        # Find max sum for right side
        right_sum = float('-inf')
        current_sum = 0
        max_right = mid + 1
        
        for i in range(mid + 1, high + 1):
            current_sum += arr[i]
            if current_sum > right_sum:
                right_sum = current_sum
                max_right = i
        
        return (left_sum + right_sum, max_left, max_right)
    
    # ==========================================
    # 5. POWER CALCULATION - Optimized
    # ==========================================
    
    def power(self, base: float, exponent: int) -> float:
        """
        Calculate base^exponent using divide and conquer
        
        Algorithm:
        - If exponent is even: base^n = (base^(n/2))^2
        - If exponent is odd: base^n = base * base^(n-1)
        
        Time: O(log n), Space: O(log n)
        """
        self.recursive_calls += 1
        
        # Base case
        if exponent == 0:
            return 1
        
        # Handle negative exponents
        if exponent < 0:
            return 1 / self.power(base, -exponent)
        
        # Divide and conquer
        if exponent % 2 == 0:
            # Even exponent: base^n = (base^(n/2))^2
            half_power = self.power(base, exponent // 2)
            return half_power * half_power
        else:
            # Odd exponent: base^n = base * base^(n-1)
            return base * self.power(base, exponent - 1)
    
    # ==========================================
    # 6. CLOSEST PAIR OF POINTS
    # ==========================================
    
    def closest_pair(self, points: List[Tuple[float, float]]) -> Tuple[float, Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Find closest pair of points using divide and conquer
        
        Time: O(n log n), Space: O(n)
        """
        def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        def closest_pair_rec(px: List[Tuple[float, float]], py: List[Tuple[float, float]]) -> Tuple[float, Tuple[Tuple[float, float], Tuple[float, float]]]:
            n = len(px)
            
            # Base case: brute force for small arrays
            if n <= 3:
                min_dist = float('inf')
                closest_points = None
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = distance(px[i], px[j])
                        if dist < min_dist:
                            min_dist = dist
                            closest_points = (px[i], px[j])
                return min_dist, closest_points
            
            # Divide
            mid = n // 2
            midpoint = px[mid]
            
            pyl = [point for point in py if point[0] <= midpoint[0]]
            pyr = [point for point in py if point[0] > midpoint[0]]
            
            # Conquer
            dl, pair_l = closest_pair_rec(px[:mid], pyl)
            dr, pair_r = closest_pair_rec(px[mid:], pyr)
            
            # Find minimum of the two halves
            if dl <= dr:
                d = dl
                closest_points = pair_l
            else:
                d = dr
                closest_points = pair_r
            
            # Check for closer points across the divide
            strip = [point for point in py if abs(point[0] - midpoint[0]) < d]
            
            for i in range(len(strip)):
                j = i + 1
                while j < len(strip) and (strip[j][1] - strip[i][1]) < d:
                    dist = distance(strip[i], strip[j])
                    if dist < d:
                        d = dist
                        closest_points = (strip[i], strip[j])
                    j += 1
            
            return d, closest_points
        
        # Sort points by x and y coordinates
        px = sorted(points, key=lambda p: p[0])
        py = sorted(points, key=lambda p: p[1])
        
        return closest_pair_rec(px, py)
    
    # ==========================================
    # 7. MATRIX MULTIPLICATION - Strassen's Algorithm
    # ==========================================
    
    def matrix_multiply_strassen(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        """
        Matrix multiplication using Strassen's divide and conquer algorithm
        
        Time: O(n^2.807), Space: O(n^2)
        Note: This is a simplified version for square matrices of power-of-2 size
        """
        n = len(A)
        
        # Base case: 1x1 matrix
        if n == 1:
            return [[A[0][0] * B[0][0]]]
        
        # Divide matrices into quadrants
        mid = n // 2
        
        # A = [[A11, A12], [A21, A22]]
        A11 = [[A[i][j] for j in range(mid)] for i in range(mid)]
        A12 = [[A[i][j] for j in range(mid, n)] for i in range(mid)]
        A21 = [[A[i][j] for j in range(mid)] for i in range(mid, n)]
        A22 = [[A[i][j] for j in range(mid, n)] for i in range(mid, n)]
        
        # B = [[B11, B12], [B21, B22]]
        B11 = [[B[i][j] for j in range(mid)] for i in range(mid)]
        B12 = [[B[i][j] for j in range(mid, n)] for i in range(mid)]
        B21 = [[B[i][j] for j in range(mid)] for i in range(mid, n)]
        B22 = [[B[i][j] for j in range(mid, n)] for i in range(mid, n)]
        
        # Calculate 7 products using Strassen's formulas
        M1 = self.matrix_multiply_strassen(
            self.matrix_add(A11, A22), self.matrix_add(B11, B22))
        M2 = self.matrix_multiply_strassen(
            self.matrix_add(A21, A22), B11)
        M3 = self.matrix_multiply_strassen(
            A11, self.matrix_subtract(B12, B22))
        M4 = self.matrix_multiply_strassen(
            A22, self.matrix_subtract(B21, B11))
        M5 = self.matrix_multiply_strassen(
            self.matrix_add(A11, A12), B22)
        M6 = self.matrix_multiply_strassen(
            self.matrix_subtract(A21, A11), self.matrix_add(B11, B12))
        M7 = self.matrix_multiply_strassen(
            self.matrix_subtract(A12, A22), self.matrix_add(B21, B22))
        
        # Calculate result quadrants
        C11 = self.matrix_subtract(self.matrix_add(self.matrix_add(M1, M4), M7), M5)
        C12 = self.matrix_add(M3, M5)
        C21 = self.matrix_add(M2, M4)
        C22 = self.matrix_subtract(self.matrix_subtract(self.matrix_add(M1, M3), M2), M6)
        
        # Combine quadrants
        C = [[0] * n for _ in range(n)]
        
        for i in range(mid):
            for j in range(mid):
                C[i][j] = C11[i][j]
                C[i][j + mid] = C12[i][j]
                C[i + mid][j] = C21[i][j]
                C[i + mid][j + mid] = C22[i][j]
        
        return C
    
    def matrix_add(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        """Add two matrices"""
        n = len(A)
        return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]
    
    def matrix_subtract(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        """Subtract two matrices"""
        n = len(A)
        return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]

# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_divide_and_conquer():
    """Demonstrate all divide and conquer algorithms"""
    print("=== DIVIDE AND CONQUER DEMONSTRATION ===\n")
    
    dc = DivideAndConquer()
    
    # 1. Merge Sort
    print("=== MERGE SORT ===")
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original array: {test_array}")
    
    dc.recursive_calls = 0
    dc.comparisons = 0
    sorted_array = dc.merge_sort(test_array)
    
    print(f"Sorted array: {sorted_array}")
    print(f"Recursive calls: {dc.recursive_calls}")
    print(f"Comparisons: {dc.comparisons}")
    print()
    
    # 2. Quick Sort
    print("=== QUICK SORT ===")
    test_array2 = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original array: {test_array2}")
    
    dc.recursive_calls = 0
    dc.comparisons = 0
    dc.quick_sort(test_array2)
    
    print(f"Sorted array: {test_array2}")
    print(f"Recursive calls: {dc.recursive_calls}")
    print(f"Comparisons: {dc.comparisons}")
    print()
    
    # 3. Binary Search
    print("=== BINARY SEARCH ===")
    sorted_arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    target = 7
    print(f"Searching for {target} in {sorted_arr}")
    
    dc.recursive_calls = 0
    dc.comparisons = 0
    result = dc.binary_search(sorted_arr, target)
    
    print(f"Found at index: {result}")
    print(f"Recursive calls: {dc.recursive_calls}")
    print(f"Comparisons: {dc.comparisons}")
    print()
    
    # 4. Maximum Subarray
    print("=== MAXIMUM SUBARRAY ===")
    test_array3 = [-2, -3, 4, -1, -2, 1, 5, -3]
    print(f"Array: {test_array3}")
    
    dc.recursive_calls = 0
    max_sum, start, end = dc.max_subarray_sum(test_array3)
    
    print(f"Maximum subarray sum: {max_sum}")
    print(f"Subarray: {test_array3[start:end+1]} (indices {start} to {end})")
    print(f"Recursive calls: {dc.recursive_calls}")
    print()
    
    # 5. Power Calculation
    print("=== POWER CALCULATION ===")
    base, exp = 2, 10
    
    dc.recursive_calls = 0
    result = dc.power(base, exp)
    
    print(f"{base}^{exp} = {result}")
    print(f"Recursive calls: {dc.recursive_calls}")
    print(f"Compared to naive O(n): {exp} calls")
    print()
    
    # 6. Closest Pair of Points
    print("=== CLOSEST PAIR OF POINTS ===")
    points = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
    print(f"Points: {points}")
    
    min_dist, closest_points = dc.closest_pair(points)
    print(f"Closest pair: {closest_points}")
    print(f"Distance: {min_dist:.2f}")

if __name__ == "__main__":
    demonstrate_divide_and_conquer()
    
    print("\n=== DIVIDE AND CONQUER PRINCIPLES ===")
    print("1. 🔸 Divide: Break problem into smaller subproblems")
    print("2. 🔸 Conquer: Solve subproblems recursively")
    print("3. 🔸 Combine: Merge solutions to solve original problem")
    
    print("\n=== WHEN TO USE DIVIDE AND CONQUER ===")
    print("✅ Problem can be broken into similar subproblems")
    print("✅ Optimal substructure exists")
    print("✅ Subproblems can be solved independently")
    print("✅ Solutions can be combined efficiently")
    
    print("\n=== COMMON PATTERNS ===")
    print("📌 Sorting: Merge sort, quick sort")
    print("📌 Searching: Binary search, finding extremes")
    print("📌 Mathematical: Fast exponentiation, matrix multiplication")
    print("📌 Geometric: Closest pair, convex hull")
    print("📌 String: Pattern matching (advanced)")
    
    print("\n=== COMPLEXITY ANALYSIS ===")
    print("⏱️  Master Theorem: T(n) = aT(n/b) + f(n)")
    print("   - a: number of subproblems")
    print("   - b: factor by which input size is divided")
    print("   - f(n): cost of divide and combine steps")
    print("📊 Common complexities: O(n log n), O(log n), O(n^log₂₃)")
