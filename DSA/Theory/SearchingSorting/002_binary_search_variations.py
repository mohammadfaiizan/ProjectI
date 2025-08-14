"""
Binary Search Variations
========================

Topics: Advanced binary search problems, rotated arrays, peak finding
Companies: Google, Facebook, Amazon, Microsoft, Apple
Difficulty: Medium to Hard
"""

from typing import List, Optional
import math

class BinarySearchVariations:
    
    # ==========================================
    # 1. ROTATED ARRAY SEARCHES
    # ==========================================
    
    def search_rotated_array(self, nums: List[int], target: int) -> int:
        """LC 33: Search in Rotated Sorted Array
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return mid
            
            # Left half is sorted
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            # Right half is sorted
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1
    
    def search_rotated_array_duplicates(self, nums: List[int], target: int) -> bool:
        """LC 81: Search in Rotated Sorted Array II (with duplicates)
        Time: O(log n) average, O(n) worst case, Space: O(1)
        """
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return True
            
            # Handle duplicates
            if nums[left] == nums[mid] == nums[right]:
                left += 1
                right -= 1
            elif nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return False
    
    def find_minimum_rotated_array(self, nums: List[int]) -> int:
        """LC 153: Find Minimum in Rotated Sorted Array
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
        
        return nums[left]
    
    def find_minimum_rotated_array_duplicates(self, nums: List[int]) -> int:
        """LC 154: Find Minimum in Rotated Sorted Array II
        Time: O(log n) average, O(n) worst case, Space: O(1)
        """
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            if nums[mid] > nums[right]:
                left = mid + 1
            elif nums[mid] < nums[right]:
                right = mid
            else:
                right -= 1  # Handle duplicates
        
        return nums[left]
    
    def find_rotation_count(self, nums: List[int]) -> int:
        """Find number of rotations in rotated sorted array
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(nums) - 1
        
        # Array is not rotated
        if nums[left] <= nums[right]:
            return 0
        
        while left <= right:
            mid = left + (right - left) // 2
            
            # Check if mid is the rotation point
            if mid < right and nums[mid] > nums[mid + 1]:
                return mid + 1
            
            if mid > left and nums[mid] < nums[mid - 1]:
                return mid
            
            # Decide which half to search
            if nums[mid] >= nums[left]:
                left = mid + 1
            else:
                right = mid - 1
        
        return 0
    
    # ==========================================
    # 2. PEAK FINDING ALGORITHMS
    # ==========================================
    
    def find_peak_element(self, nums: List[int]) -> int:
        """LC 162: Find Peak Element
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            if nums[mid] > nums[mid + 1]:
                right = mid
            else:
                left = mid + 1
        
        return left
    
    def find_peak_element_2d(self, matrix: List[List[int]]) -> List[int]:
        """LC 1901: Find Peak Element II in 2D Matrix
        Time: O(m log n), Space: O(1)
        """
        m, n = len(matrix), len(matrix[0])
        
        def find_max_in_col(col):
            max_row = 0
            for i in range(1, m):
                if matrix[i][col] > matrix[max_row][col]:
                    max_row = i
            return max_row
        
        left, right = 0, n - 1
        
        while left <= right:
            mid_col = left + (right - left) // 2
            max_row = find_max_in_col(mid_col)
            
            left_val = matrix[max_row][mid_col - 1] if mid_col > 0 else -1
            right_val = matrix[max_row][mid_col + 1] if mid_col < n - 1 else -1
            
            if matrix[max_row][mid_col] > left_val and matrix[max_row][mid_col] > right_val:
                return [max_row, mid_col]
            elif matrix[max_row][mid_col] < left_val:
                right = mid_col - 1
            else:
                left = mid_col + 1
        
        return [-1, -1]
    
    def find_local_minimum(self, nums: List[int]) -> int:
        """Find local minimum in array where nums[i] < nums[i-1] and nums[i] < nums[i+1]
        Time: O(log n), Space: O(1)
        """
        n = len(nums)
        
        # Check boundary conditions
        if n == 1:
            return 0
        if nums[0] < nums[1]:
            return 0
        if nums[n-1] < nums[n-2]:
            return n - 1
        
        left, right = 1, n - 2
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] < nums[mid - 1] and nums[mid] < nums[mid + 1]:
                return mid
            elif nums[mid] > nums[mid - 1]:
                right = mid - 1
            else:
                left = mid + 1
        
        return -1
    
    # ==========================================
    # 3. SQUARE ROOT AND POWER FUNCTIONS
    # ==========================================
    
    def sqrt_binary_search(self, x: int) -> int:
        """LC 69: Sqrt(x) using binary search
        Time: O(log x), Space: O(1)
        """
        if x < 2:
            return x
        
        left, right = 2, x // 2
        
        while left <= right:
            mid = left + (right - left) // 2
            square = mid * mid
            
            if square == x:
                return mid
            elif square < x:
                left = mid + 1
            else:
                right = mid - 1
        
        return right
    
    def is_perfect_square(self, num: int) -> bool:
        """LC 367: Valid Perfect Square
        Time: O(log num), Space: O(1)
        """
        if num < 2:
            return True
        
        left, right = 2, num // 2
        
        while left <= right:
            mid = left + (right - left) // 2
            square = mid * mid
            
            if square == num:
                return True
            elif square < num:
                left = mid + 1
            else:
                right = mid - 1
        
        return False
    
    def power_function(self, x: float, n: int) -> float:
        """LC 50: Pow(x, n) using binary search approach
        Time: O(log n), Space: O(log n) due to recursion
        """
        if n == 0:
            return 1
        
        if n < 0:
            return 1 / self.power_function(x, -n)
        
        if n % 2 == 0:
            half = self.power_function(x, n // 2)
            return half * half
        else:
            return x * self.power_function(x, n - 1)
    
    def power_iterative(self, x: float, n: int) -> float:
        """Iterative power function
        Time: O(log n), Space: O(1)
        """
        if n == 0:
            return 1
        
        if n < 0:
            x = 1 / x
            n = -n
        
        result = 1
        current_power = x
        
        while n > 0:
            if n % 2 == 1:
                result *= current_power
            current_power *= current_power
            n //= 2
        
        return result
    
    # ==========================================
    # 4. SEARCH IN SPECIALIZED ARRAYS
    # ==========================================
    
    def search_matrix(self, matrix: List[List[int]], target: int) -> bool:
        """LC 74: Search a 2D Matrix (treat as 1D sorted array)
        Time: O(log(m*n)), Space: O(1)
        """
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
    
    def search_matrix_ii(self, matrix: List[List[int]], target: int) -> bool:
        """LC 240: Search a 2D Matrix II (row and column sorted)
        Time: O(m + n), Space: O(1)
        """
        if not matrix or not matrix[0]:
            return False
        
        m, n = len(matrix), len(matrix[0])
        row, col = 0, n - 1
        
        while row < m and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1
        
        return False
    
    def search_range(self, nums: List[int], target: int) -> List[int]:
        """LC 34: Find First and Last Position of Element
        Time: O(log n), Space: O(1)
        """
        def find_boundary(nums, target, is_first):
            left, right = 0, len(nums) - 1
            boundary_index = -1
            
            while left <= right:
                mid = left + (right - left) // 2
                
                if nums[mid] == target:
                    boundary_index = mid
                    if is_first:
                        right = mid - 1
                    else:
                        left = mid + 1
                elif nums[mid] > target:
                    right = mid - 1
                else:
                    left = mid + 1
            
            return boundary_index
        
        first_pos = find_boundary(nums, target, True)
        if first_pos == -1:
            return [-1, -1]
        
        last_pos = find_boundary(nums, target, False)
        return [first_pos, last_pos]
    
    # ==========================================
    # 5. ADVANCED BINARY SEARCH APPLICATIONS
    # ==========================================
    
    def find_k_closest_elements(self, arr: List[int], k: int, x: int) -> List[int]:
        """LC 658: Find K Closest Elements
        Time: O(log n + k), Space: O(1)
        """
        left, right = 0, len(arr) - k
        
        while left < right:
            mid = left + (right - left) // 2
            
            if x - arr[mid] > arr[mid + k] - x:
                left = mid + 1
            else:
                right = mid
        
        return arr[left:left + k]
    
    def search_unknown_size(self, reader, target: int) -> int:
        """LC 702: Search in a Sorted Array of Unknown Size
        Time: O(log n), Space: O(1)
        """
        # Find the right boundary using exponential search
        left, right = 0, 1
        
        while reader.get(right) < target:
            left = right
            right *= 2
        
        # Binary search in the found range
        while left <= right:
            mid = left + (right - left) // 2
            mid_val = reader.get(mid)
            
            if mid_val == target:
                return mid
            elif mid_val < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    
    def find_duplicate_number(self, nums: List[int]) -> int:
        """LC 287: Find the Duplicate Number using binary search
        Time: O(n log n), Space: O(1)
        """
        left, right = 1, len(nums) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            count = sum(1 for num in nums if num <= mid)
            
            if count <= mid:
                left = mid + 1
            else:
                right = mid
        
        return left

# Test Examples
def run_examples():
    bsv = BinarySearchVariations()
    
    print("=== BINARY SEARCH VARIATIONS ===\n")
    
    # Rotated array search
    print("1. ROTATED ARRAY SEARCH:")
    rotated_arr = [4, 5, 6, 7, 0, 1, 2]
    target = 0
    result = bsv.search_rotated_array(rotated_arr, target)
    print(f"Search {target} in rotated array {rotated_arr}: index {result}")
    
    min_val = bsv.find_minimum_rotated_array(rotated_arr)
    print(f"Minimum in rotated array: {min_val}")
    
    rotation_count = bsv.find_rotation_count(rotated_arr)
    print(f"Number of rotations: {rotation_count}")
    
    # Peak finding
    print("\n2. PEAK FINDING:")
    peak_arr = [1, 2, 3, 1]
    peak_index = bsv.find_peak_element(peak_arr)
    print(f"Peak element in {peak_arr}: index {peak_index} (value: {peak_arr[peak_index]})")
    
    # Square root
    print("\n3. SQUARE ROOT:")
    x = 8
    sqrt_result = bsv.sqrt_binary_search(x)
    print(f"Square root of {x}: {sqrt_result}")
    
    # Perfect square
    perfect_square_test = bsv.is_perfect_square(16)
    print(f"Is 16 a perfect square: {perfect_square_test}")
    
    # Power function
    print("\n4. POWER FUNCTION:")
    power_result = bsv.power_iterative(2.0, 10)
    print(f"2^10 = {power_result}")
    
    # Search range
    print("\n5. SEARCH RANGE:")
    nums_with_duplicates = [5, 7, 7, 8, 8, 10]
    target = 8
    range_result = bsv.search_range(nums_with_duplicates, target)
    print(f"Range of {target} in {nums_with_duplicates}: {range_result}")
    
    # K closest elements
    print("\n6. K CLOSEST ELEMENTS:")
    arr = [1, 2, 3, 4, 5]
    k, x = 4, 3
    closest = bsv.find_k_closest_elements(arr, k, x)
    print(f"{k} closest elements to {x} in {arr}: {closest}")

if __name__ == "__main__":
    run_examples() 