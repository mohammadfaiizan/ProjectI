"""
Basic Searching Algorithms
==========================

Topics: Linear search, binary search, ternary search, jump search
Companies: Google, Facebook, Amazon, Microsoft, Apple
Difficulty: Easy to Medium
LeetCode Problems: 704, 35, 33, 81, 153, 154
"""

from typing import List, Optional, Tuple
import math

class BasicSearchingAlgorithms:
    
    # ==========================================
    # 1. LINEAR SEARCH ALGORITHMS
    # ==========================================
    
    def linear_search(self, arr: List[int], target: int) -> int:
        """LC 704 variant: Basic linear search
        Time: O(n), Space: O(1)
        """
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1
    
    def linear_search_all_occurrences(self, arr: List[int], target: int) -> List[int]:
        """Find all occurrences using linear search
        Time: O(n), Space: O(k) where k is number of occurrences
        """
        indices = []
        for i in range(len(arr)):
            if arr[i] == target:
                indices.append(i)
        return indices
    
    def linear_search_recursive(self, arr: List[int], target: int, index: int = 0) -> int:
        """Recursive linear search
        Time: O(n), Space: O(n) due to recursion
        """
        if index >= len(arr):
            return -1
        
        if arr[index] == target:
            return index
        
        return self.linear_search_recursive(arr, target, index + 1)
    
    def sentinel_linear_search(self, arr: List[int], target: int) -> int:
        """Sentinel linear search - reduces comparisons
        Time: O(n), Space: O(1)
        """
        if not arr:
            return -1
        
        n = len(arr)
        last = arr[n - 1]
        arr[n - 1] = target
        
        i = 0
        while arr[i] != target:
            i += 1
        
        arr[n - 1] = last  # Restore original value
        
        if i < n - 1 or arr[n - 1] == target:
            return i
        return -1
    
    # ==========================================
    # 2. BINARY SEARCH ALGORITHMS
    # ==========================================
    
    def binary_search_iterative(self, arr: List[int], target: int) -> int:
        """LC 704: Binary Search
        Time: O(log n), Space: O(1)
        """
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
        """Recursive binary search
        Time: O(log n), Space: O(log n)
        """
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
    
    def binary_search_leftmost(self, arr: List[int], target: int) -> int:
        """Find leftmost occurrence of target
        Time: O(log n), Space: O(1)
        """
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
    
    def binary_search_rightmost(self, arr: List[int], target: int) -> int:
        """Find rightmost occurrence of target
        Time: O(log n), Space: O(1)
        """
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
    
    def search_insert_position(self, nums: List[int], target: int) -> int:
        """LC 35: Search Insert Position
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(nums)
        
        while left < right:
            mid = left + (right - left) // 2
            
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        
        return left
    
    # ==========================================
    # 3. ROTATED ARRAY SEARCHES
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
        """LC 81: Search in Rotated Sorted Array II
        Time: O(log n) average, O(n) worst, Space: O(1)
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
    
    def find_minimum_rotated(self, nums: List[int]) -> int:
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
    
    # ==========================================
    # 4. SPECIALIZED SEARCH ALGORITHMS
    # ==========================================
    
    def jump_search(self, arr: List[int], target: int) -> int:
        """Jump search algorithm
        Time: O(√n), Space: O(1)
        """
        n = len(arr)
        step = int(math.sqrt(n))
        prev = 0
        
        # Jump to find the block containing target
        while arr[min(step, n) - 1] < target:
            prev = step
            step += int(math.sqrt(n))
            if prev >= n:
                return -1
        
        # Linear search in the identified block
        while arr[prev] < target:
            prev += 1
            if prev == min(step, n):
                return -1
        
        if arr[prev] == target:
            return prev
        
        return -1
    
    def interpolation_search(self, arr: List[int], target: int) -> int:
        """Interpolation search for uniformly distributed data
        Time: O(log log n) average, O(n) worst, Space: O(1)
        """
        left, right = 0, len(arr) - 1
        
        while left <= right and target >= arr[left] and target <= arr[right]:
            if left == right:
                return left if arr[left] == target else -1
            
            # Calculate position using interpolation formula
            pos = left + ((target - arr[left]) * (right - left)) // (arr[right] - arr[left])
            pos = max(left, min(pos, right))
            
            if arr[pos] == target:
                return pos
            elif arr[pos] < target:
                left = pos + 1
            else:
                right = pos - 1
        
        return -1
    
    def exponential_search(self, arr: List[int], target: int) -> int:
        """Exponential search (galloping search)
        Time: O(log n), Space: O(1)
        """
        n = len(arr)
        
        if arr[0] == target:
            return 0
        
        # Find range for binary search
        i = 1
        while i < n and arr[i] <= target:
            i *= 2
        
        # Binary search in the found range
        return self._binary_search_range(arr, target, i // 2, min(i, n - 1))
    
    def _binary_search_range(self, arr: List[int], target: int, left: int, right: int) -> int:
        """Helper method for binary search in range"""
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1

# Test Examples
def run_examples():
    bsa = BasicSearchingAlgorithms()
    
    print("=== BASIC SEARCHING ALGORITHMS ===\n")
    
    # Test arrays
    sorted_arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    unsorted_arr = [64, 34, 25, 12, 22, 11, 90]
    duplicates_arr = [1, 2, 2, 2, 3, 4, 4, 5]
    rotated_arr = [4, 5, 6, 7, 0, 1, 2]
    
    print("1. LINEAR SEARCH:")
    target = 11
    result = bsa.linear_search(unsorted_arr, target)
    print(f"Linear search for {target}: index {result}")
    
    all_occurrences = bsa.linear_search_all_occurrences(duplicates_arr, 2)
    print(f"All occurrences of 2: {all_occurrences}")
    
    print("\n2. BINARY SEARCH:")
    target = 7
    result = bsa.binary_search_iterative(sorted_arr, target)
    print(f"Binary search for {target}: index {result}")
    
    insert_pos = bsa.search_insert_position(sorted_arr, 8)
    print(f"Insert position for 8: {insert_pos}")
    
    print("\n3. ROTATED ARRAY SEARCH:")
    target = 0
    result = bsa.search_rotated_array(rotated_arr, target)
    print(f"Search {target} in rotated array: index {result}")
    
    min_val = bsa.find_minimum_rotated(rotated_arr)
    print(f"Minimum in rotated array: {min_val}")
    
    print("\n4. SPECIALIZED SEARCHES:")
    result = bsa.jump_search(sorted_arr, 15)
    print(f"Jump search for 15: index {result}")
    
    result = bsa.exponential_search(sorted_arr, 9)
    print(f"Exponential search for 9: index {result}")

if __name__ == "__main__":
    run_examples() 