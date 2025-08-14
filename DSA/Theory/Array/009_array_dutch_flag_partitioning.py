"""
Array Dutch Flag Partitioning - Three-way Partitioning
======================================================

Topics: Dutch National Flag, three-way partitioning, color sorting
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Medium
"""

from typing import List
import random

class ArrayDutchFlagPartitioning:
    
    # ==========================================
    # 1. CLASSIC DUTCH NATIONAL FLAG
    # ==========================================
    
    def sort_colors(self, nums: List[int]) -> None:
        """LC 75: Sort Colors - Dutch National Flag Algorithm
        Time: O(n), Space: O(1)
        """
        # Three-way partitioning: 0s, 1s, 2s
        low = mid = 0
        high = len(nums) - 1
        
        while mid <= high:
            if nums[mid] == 0:
                nums[low], nums[mid] = nums[mid], nums[low]
                low += 1
                mid += 1
            elif nums[mid] == 1:
                mid += 1
            else:  # nums[mid] == 2
                nums[mid], nums[high] = nums[high], nums[mid]
                high -= 1
                # Don't increment mid here as we need to check swapped element
    
    def three_way_partition(self, arr: List[int], pivot: int) -> List[int]:
        """Three-way partition around pivot value
        Time: O(n), Space: O(1)
        """
        low = mid = 0
        high = len(arr) - 1
        
        while mid <= high:
            if arr[mid] < pivot:
                arr[low], arr[mid] = arr[mid], arr[low]
                low += 1
                mid += 1
            elif arr[mid] == pivot:
                mid += 1
            else:  # arr[mid] > pivot
                arr[mid], arr[high] = arr[high], arr[mid]
                high -= 1
        
        return arr
    
    def three_way_quicksort(self, arr: List[int], left: int = 0, right: int = None) -> List[int]:
        """Three-way quicksort using Dutch National Flag
        Time: O(n log n) average, Space: O(log n)
        """
        if right is None:
            right = len(arr) - 1
        
        if left >= right:
            return arr
        
        def three_way_partition_quicksort(arr: List[int], left: int, right: int, pivot_val: int):
            """Returns (lt, gt) where elements < pivot are in [left, lt)
            and elements > pivot are in (gt, right]"""
            lt = left
            gt = right
            i = left
            
            while i <= gt:
                if arr[i] < pivot_val:
                    arr[lt], arr[i] = arr[i], arr[lt]
                    lt += 1
                    i += 1
                elif arr[i] > pivot_val:
                    arr[i], arr[gt] = arr[gt], arr[i]
                    gt -= 1
                    # Don't increment i
                else:
                    i += 1
            
            return lt, gt
        
        # Choose random pivot
        pivot_index = random.randint(left, right)
        pivot_val = arr[pivot_index]
        
        lt, gt = three_way_partition_quicksort(arr, left, right, pivot_val)
        
        # Recursively sort left and right parts
        self.three_way_quicksort(arr, left, lt - 1)
        self.three_way_quicksort(arr, gt + 1, right)
        
        return arr
    
    # ==========================================
    # 2. PARTITION VARIATIONS
    # ==========================================
    
    def partition_by_parity(self, nums: List[int]) -> List[int]:
        """LC 905: Sort Array By Parity (even before odd)
        Time: O(n), Space: O(1)
        """
        left = 0
        right = len(nums) - 1
        
        while left < right:
            if nums[left] % 2 == 1:  # left is odd
                nums[left], nums[right] = nums[right], nums[left]
                right -= 1
            else:
                left += 1
        
        return nums
    
    def partition_by_parity_ii(self, nums: List[int]) -> List[int]:
        """LC 922: Sort Array By Parity II (even at even indices, odd at odd)
        Time: O(n), Space: O(1)
        """
        n = len(nums)
        even_idx = 0
        odd_idx = 1
        
        while even_idx < n and odd_idx < n:
            if nums[even_idx] % 2 == 0:
                even_idx += 2
            elif nums[odd_idx] % 2 == 1:
                odd_idx += 2
            else:
                nums[even_idx], nums[odd_idx] = nums[odd_idx], nums[even_idx]
        
        return nums
    
    def segregate_positive_negative(self, arr: List[int]) -> List[int]:
        """Segregate positive and negative numbers
        Time: O(n), Space: O(1)
        """
        left = 0
        right = len(arr) - 1
        
        while left < right:
            if arr[left] < 0:
                left += 1
            elif arr[right] >= 0:
                right -= 1
            else:
                arr[left], arr[right] = arr[right], arr[left]
                left += 1
                right -= 1
        
        return arr
    
    def rearrange_positive_negative_alternating(self, arr: List[int]) -> List[int]:
        """Rearrange positive and negative numbers alternately
        Time: O(n), Space: O(1) - assumes equal positive and negative
        """
        # First segregate
        self.segregate_positive_negative(arr)
        
        # Find the partition point
        pos_start = 0
        while pos_start < len(arr) and arr[pos_start] < 0:
            pos_start += 1
        
        # Rearrange alternately starting from negative
        neg_idx = 0
        pos_idx = pos_start
        
        # Create new array for alternating arrangement
        result = []
        while neg_idx < pos_start and pos_idx < len(arr):
            result.append(arr[neg_idx])
            result.append(arr[pos_idx])
            neg_idx += 1
            pos_idx += 1
        
        # Add remaining elements
        while neg_idx < pos_start:
            result.append(arr[neg_idx])
            neg_idx += 1
        
        while pos_idx < len(arr):
            result.append(arr[pos_idx])
            pos_idx += 1
        
        return result
    
    # ==========================================
    # 3. ADVANCED PARTITIONING PROBLEMS
    # ==========================================
    
    def wiggle_sort(self, nums: List[int]) -> None:
        """LC 280: Wiggle Sort (nums[0] < nums[1] > nums[2] < nums[3]...)
        Time: O(n), Space: O(1)
        """
        for i in range(1, len(nums)):
            if (i % 2 == 1 and nums[i] < nums[i-1]) or \
               (i % 2 == 0 and nums[i] > nums[i-1]):
                nums[i], nums[i-1] = nums[i-1], nums[i]
    
    def wiggle_sort_ii(self, nums: List[int]) -> None:
        """LC 324: Wiggle Sort II (nums[0] < nums[1] > nums[2] < nums[3]...)
        All elements are unique after sorting
        Time: O(n log n), Space: O(n)
        """
        # Sort the array first
        nums.sort()
        n = len(nums)
        
        # Create result array
        result = [0] * n
        
        # Fill odd positions with larger half (in reverse)
        odd_idx = 1
        for i in range(n - 1, n // 2 - 1, -1):
            if odd_idx < n:
                result[odd_idx] = nums[i]
                odd_idx += 2
        
        # Fill even positions with smaller half (in reverse)
        even_idx = 0
        for i in range(n // 2 - 1, -1, -1):
            if even_idx < n:
                result[even_idx] = nums[i]
                even_idx += 2
        
        # Copy back to original array
        for i in range(n):
            nums[i] = result[i]
    
    def k_closest_to_origin(self, points: List[List[int]], k: int) -> List[List[int]]:
        """LC 973: K Closest Points to Origin - using partitioning
        Time: O(n) average, Space: O(1)
        """
        def distance_squared(point):
            return point[0] ** 2 + point[1] ** 2
        
        def partition(left: int, right: int) -> int:
            pivot_dist = distance_squared(points[right])
            i = left
            
            for j in range(left, right):
                if distance_squared(points[j]) <= pivot_dist:
                    points[i], points[j] = points[j], points[i]
                    i += 1
            
            points[i], points[right] = points[right], points[i]
            return i
        
        def quickselect(left: int, right: int, k: int):
            if left == right:
                return
            
            pivot_idx = partition(left, right)
            
            if pivot_idx == k:
                return
            elif pivot_idx < k:
                quickselect(pivot_idx + 1, right, k)
            else:
                quickselect(left, pivot_idx - 1, k)
        
        quickselect(0, len(points) - 1, k)
        return points[:k]
    
    # ==========================================
    # 4. CUSTOM PARTITIONING ALGORITHMS
    # ==========================================
    
    def partition_array_into_three_parts(self, arr: List[int], x: int, y: int) -> List[List[int]]:
        """Partition array into three parts: < x, [x, y], > y
        Time: O(n), Space: O(1) excluding output
        """
        less_than_x = []
        between_x_y = []
        greater_than_y = []
        
        for num in arr:
            if num < x:
                less_than_x.append(num)
            elif num > y:
                greater_than_y.append(num)
            else:
                between_x_y.append(num)
        
        return [less_than_x, between_x_y, greater_than_y]
    
    def rainbow_sort(self, arr: List[int], k: int) -> List[int]:
        """Sort array with k distinct values (generalized Dutch flag)
        Time: O(n*k), Space: O(1)
        """
        # Use counting sort approach for k colors
        count = [0] * k
        
        # Count occurrences
        for num in arr:
            count[num] += 1
        
        # Reconstruct array
        idx = 0
        for color in range(k):
            for _ in range(count[color]):
                arr[idx] = color
                idx += 1
        
        return arr
    
    def move_zeros_to_end(self, nums: List[int]) -> None:
        """LC 283: Move Zeroes to end while maintaining order
        Time: O(n), Space: O(1)
        """
        write_idx = 0
        
        # Move non-zero elements to front
        for read_idx in range(len(nums)):
            if nums[read_idx] != 0:
                nums[write_idx] = nums[read_idx]
                write_idx += 1
        
        # Fill remaining with zeros
        while write_idx < len(nums):
            nums[write_idx] = 0
            write_idx += 1

# Test Examples
def run_examples():
    adfp = ArrayDutchFlagPartitioning()
    
    print("=== DUTCH FLAG PARTITIONING EXAMPLES ===\n")
    
    # Classic Dutch flag
    print("1. CLASSIC DUTCH NATIONAL FLAG:")
    colors = [2, 0, 2, 1, 1, 0]
    print(f"Before sorting colors: {colors}")
    adfp.sort_colors(colors)
    print(f"After sorting colors: {colors}")
    
    # Three-way partition
    print("\n2. THREE-WAY PARTITION:")
    arr = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    pivot = 4
    print(f"Before partition around {pivot}: {arr}")
    adfp.three_way_partition(arr.copy(), pivot)
    print(f"After partition: {adfp.three_way_partition(arr, pivot)}")
    
    # Parity partitioning
    print("\n3. PARITY PARTITIONING:")
    nums = [3, 1, 2, 4]
    print(f"Before parity sort: {nums}")
    result = adfp.partition_by_parity(nums.copy())
    print(f"After parity sort: {result}")
    
    # Positive-negative segregation
    print("\n4. POSITIVE-NEGATIVE SEGREGATION:")
    arr = [1, -2, 3, -4, 5, -6]
    print(f"Before segregation: {arr}")
    result = adfp.segregate_positive_negative(arr.copy())
    print(f"After segregation: {result}")
    
    # Wiggle sort
    print("\n5. WIGGLE SORT:")
    nums = [3, 5, 2, 1, 6, 4]
    print(f"Before wiggle sort: {nums}")
    adfp.wiggle_sort(nums)
    print(f"After wiggle sort: {nums}")
    
    # K closest points
    print("\n6. K CLOSEST POINTS:")
    points = [[1, 1], [2, 2], [3, 3]]
    k = 2
    print(f"Points: {points}, k={k}")
    result = adfp.k_closest_to_origin(points, k)
    print(f"K closest: {result}")
    
    # Move zeros
    print("\n7. MOVE ZEROS:")
    nums = [0, 1, 0, 3, 12]
    print(f"Before moving zeros: {nums}")
    adfp.move_zeros_to_end(nums)
    print(f"After moving zeros: {nums}")

if __name__ == "__main__":
    run_examples() 