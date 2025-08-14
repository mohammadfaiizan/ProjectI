"""
Array Searching Techniques - Complete Implementation
==================================================

Topics: Binary search, ternary search, exponential search
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Easy to Hard
"""

from typing import List, Optional, Tuple
import math

class ArraySearchingTechniques:
    
    # ==========================================
    # 1. LINEAR SEARCH VARIATIONS
    # ==========================================
    
    def linear_search(self, arr: List[int], target: int) -> int:
        """Basic linear search - Time: O(n), Space: O(1)"""
        for i, element in enumerate(arr):
            if element == target:
                return i
        return -1
    
    def linear_search_recursive(self, arr: List[int], target: int, index: int = 0) -> int:
        """Recursive linear search - Time: O(n), Space: O(n)"""
        if index >= len(arr):
            return -1
        
        if arr[index] == target:
            return index
        
        return self.linear_search_recursive(arr, target, index + 1)
    
    def find_all_occurrences(self, arr: List[int], target: int) -> List[int]:
        """Find all indices of target - Time: O(n), Space: O(k)"""
        indices = []
        for i, element in enumerate(arr):
            if element == target:
                indices.append(i)
        return indices
    
    # ==========================================
    # 2. BINARY SEARCH VARIATIONS
    # ==========================================
    
    def binary_search(self, arr: List[int], target: int) -> int:
        """Standard binary search - Time: O(log n), Space: O(1)"""
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    
    def binary_search_recursive(self, arr: List[int], target: int, left: int = 0, right: int = None) -> int:
        """Recursive binary search - Time: O(log n), Space: O(log n)"""
        if right is None:
            right = len(arr) - 1
        
        if left > right:
            return -1
        
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            return self.binary_search_recursive(arr, target, mid + 1, right)
        else:
            return self.binary_search_recursive(arr, target, left, mid - 1)
    
    def find_first_occurrence(self, arr: List[int], target: int) -> int:
        """Find first occurrence in sorted array with duplicates"""
        left, right = 0, len(arr) - 1
        result = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] == target:
                result = mid
                right = mid - 1  # Continue searching left
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    def find_last_occurrence(self, arr: List[int], target: int) -> int:
        """Find last occurrence in sorted array with duplicates"""
        left, right = 0, len(arr) - 1
        result = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] == target:
                result = mid
                left = mid + 1  # Continue searching right
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    def count_occurrences(self, arr: List[int], target: int) -> int:
        """Count occurrences using binary search"""
        first = self.find_first_occurrence(arr, target)
        if first == -1:
            return 0
        
        last = self.find_last_occurrence(arr, target)
        return last - first + 1
    
    def search_range(self, arr: List[int], target: int) -> List[int]:
        """LC 34: Find First and Last Position"""
        first = self.find_first_occurrence(arr, target)
        if first == -1:
            return [-1, -1]
        
        last = self.find_last_occurrence(arr, target)
        return [first, last]
    
    # ==========================================
    # 3. BINARY SEARCH ON ANSWER
    # ==========================================
    
    def search_insert_position(self, arr: List[int], target: int) -> int:
        """LC 35: Search Insert Position"""
        left, right = 0, len(arr)
        
        while left < right:
            mid = left + (right - left) // 2
            
            if arr[mid] < target:
                left = mid + 1
            else:
                right = mid
        
        return left
    
    def find_peak_element(self, arr: List[int]) -> int:
        """LC 162: Find Peak Element"""
        left, right = 0, len(arr) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            if arr[mid] > arr[mid + 1]:
                right = mid
            else:
                left = mid + 1
        
        return left
    
    def search_rotated_array(self, arr: List[int], target: int) -> int:
        """LC 33: Search in Rotated Sorted Array"""
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] == target:
                return mid
            
            # Left half is sorted
            if arr[left] <= arr[mid]:
                if arr[left] <= target < arr[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            # Right half is sorted
            else:
                if arr[mid] < target <= arr[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1
    
    def find_minimum_rotated(self, arr: List[int]) -> int:
        """LC 153: Find Minimum in Rotated Sorted Array"""
        left, right = 0, len(arr) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            if arr[mid] > arr[right]:
                left = mid + 1
            else:
                right = mid
        
        return arr[left]
    
    # ==========================================
    # 4. ADVANCED SEARCHING TECHNIQUES
    # ==========================================
    
    def ternary_search(self, arr: List[int], target: int) -> int:
        """Ternary search - Time: O(log₃ n), Space: O(1)"""
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid1 = left + (right - left) // 3
            mid2 = right - (right - left) // 3
            
            if arr[mid1] == target:
                return mid1
            if arr[mid2] == target:
                return mid2
            
            if target < arr[mid1]:
                right = mid1 - 1
            elif target > arr[mid2]:
                left = mid2 + 1
            else:
                left = mid1 + 1
                right = mid2 - 1
        
        return -1
    
    def exponential_search(self, arr: List[int], target: int) -> int:
        """Exponential search - Time: O(log n), Space: O(1)"""
        if arr[0] == target:
            return 0
        
        # Find range for binary search
        i = 1
        while i < len(arr) and arr[i] <= target:
            i *= 2
        
        # Binary search in found range
        return self.binary_search_range(arr, target, i // 2, min(i, len(arr) - 1))
    
    def binary_search_range(self, arr: List[int], target: int, left: int, right: int) -> int:
        """Binary search in specific range"""
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    
    def interpolation_search(self, arr: List[int], target: int) -> int:
        """Interpolation search - Time: O(log log n) avg, Space: O(1)"""
        left, right = 0, len(arr) - 1
        
        while left <= right and target >= arr[left] and target <= arr[right]:
            if left == right:
                return left if arr[left] == target else -1
            
            # Interpolation formula
            pos = left + int(((target - arr[left]) / (arr[right] - arr[left])) * (right - left))
            
            if arr[pos] == target:
                return pos
            elif arr[pos] < target:
                left = pos + 1
            else:
                right = pos - 1
        
        return -1
    
    def jump_search(self, arr: List[int], target: int) -> int:
        """Jump search - Time: O(√n), Space: O(1)"""
        n = len(arr)
        step = int(math.sqrt(n))
        prev = 0
        
        # Find block containing element
        while arr[min(step, n) - 1] < target:
            prev = step
            step += int(math.sqrt(n))
            if prev >= n:
                return -1
        
        # Linear search in block
        while arr[prev] < target:
            prev += 1
            if prev == min(step, n):
                return -1
        
        return prev if arr[prev] == target else -1
    
    # ==========================================
    # 5. 2D ARRAY SEARCHING
    # ==========================================
    
    def search_2d_matrix_i(self, matrix: List[List[int]], target: int) -> bool:
        """LC 74: Search 2D Matrix (sorted)"""
        if not matrix or not matrix[0]:
            return False
        
        m, n = len(matrix), len(matrix[0])
        left, right = 0, m * n - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            mid_val = matrix[mid // n][mid % n]
            
            if mid_val == target:
                return True
            elif mid_val < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return False
    
    def search_2d_matrix_ii(self, matrix: List[List[int]], target: int) -> bool:
        """LC 240: Search 2D Matrix II"""
        if not matrix or not matrix[0]:
            return False
        
        row, col = 0, len(matrix[0]) - 1
        
        while row < len(matrix) and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1
        
        return False

# Test Examples
def run_examples():
    ast = ArraySearchingTechniques()
    
    print("=== ARRAY SEARCHING TECHNIQUES EXAMPLES ===\n")
    
    # Basic searches
    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    target = 7
    
    print("1. BASIC SEARCHES:")
    print(f"Array: {arr}, Target: {target}")
    print(f"Linear search: {ast.linear_search(arr, target)}")
    print(f"Binary search: {ast.binary_search(arr, target)}")
    print(f"Ternary search: {ast.ternary_search(arr, target)}")
    print(f"Jump search: {ast.jump_search(arr, target)}")
    
    # Duplicate handling
    print("\n2. DUPLICATE HANDLING:")
    arr_dup = [1, 2, 2, 2, 3, 4, 4, 5]
    target_dup = 2
    print(f"Array with duplicates: {arr_dup}, Target: {target_dup}")
    print(f"First occurrence: {ast.find_first_occurrence(arr_dup, target_dup)}")
    print(f"Last occurrence: {ast.find_last_occurrence(arr_dup, target_dup)}")
    print(f"Count occurrences: {ast.count_occurrences(arr_dup, target_dup)}")
    print(f"Search range: {ast.search_range(arr_dup, target_dup)}")
    
    # Rotated array
    print("\n3. ROTATED ARRAY:")
    rotated = [4, 5, 6, 7, 0, 1, 2]
    target_rot = 0
    print(f"Rotated array: {rotated}, Target: {target_rot}")
    print(f"Search result: {ast.search_rotated_array(rotated, target_rot)}")
    print(f"Minimum element: {ast.find_minimum_rotated(rotated)}")
    
    # 2D Matrix
    print("\n4. 2D MATRIX SEARCH:")
    matrix = [[1, 4, 7, 11], [2, 5, 8, 12], [3, 6, 9, 16]]
    target_2d = 5
    print(f"Matrix search for {target_2d}: {ast.search_2d_matrix_ii(matrix, target_2d)}")

if __name__ == "__main__":
    run_examples() 