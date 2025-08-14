"""
Array Two Pointers Techniques - Complete Implementation
=====================================================

Topics: Two pointers, fast-slow pointers, opposite direction
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Easy to Hard
"""

from typing import List, Tuple, Optional

class ArrayTwoPointers:
    
    # ==========================================
    # 1. OPPOSITE DIRECTION TWO POINTERS
    # ==========================================
    
    def two_sum_sorted(self, arr: List[int], target: int) -> List[int]:
        """LC 167: Two Sum II - Input array is sorted
        Time: O(n), Space: O(1)
        """
        left, right = 0, len(arr) - 1
        
        while left < right:
            current_sum = arr[left] + arr[right]
            
            if current_sum == target:
                return [left + 1, right + 1]  # 1-indexed
            elif current_sum < target:
                left += 1
            else:
                right -= 1
        
        return []
    
    def three_sum(self, nums: List[int]) -> List[List[int]]:
        """LC 15: 3Sum - Find all unique triplets that sum to zero
        Time: O(n²), Space: O(1)
        """
        nums.sort()
        result = []
        
        for i in range(len(nums) - 2):
            # Skip duplicates for first element
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            
            left, right = i + 1, len(nums) - 1
            
            while left < right:
                current_sum = nums[i] + nums[left] + nums[right]
                
                if current_sum == 0:
                    result.append([nums[i], nums[left], nums[right]])
                    
                    # Skip duplicates
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
                elif current_sum < 0:
                    left += 1
                else:
                    right -= 1
        
        return result
    
    def three_sum_closest(self, nums: List[int], target: int) -> int:
        """LC 16: 3Sum Closest
        Time: O(n²), Space: O(1)
        """
        nums.sort()
        closest_sum = float('inf')
        
        for i in range(len(nums) - 2):
            left, right = i + 1, len(nums) - 1
            
            while left < right:
                current_sum = nums[i] + nums[left] + nums[right]
                
                if abs(current_sum - target) < abs(closest_sum - target):
                    closest_sum = current_sum
                
                if current_sum < target:
                    left += 1
                else:
                    right -= 1
        
        return closest_sum
    
    def four_sum(self, nums: List[int], target: int) -> List[List[int]]:
        """LC 18: 4Sum
        Time: O(n³), Space: O(1)
        """
        nums.sort()
        result = []
        n = len(nums)
        
        for i in range(n - 3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            
            for j in range(i + 1, n - 2):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                
                left, right = j + 1, n - 1
                
                while left < right:
                    current_sum = nums[i] + nums[j] + nums[left] + nums[right]
                    
                    if current_sum == target:
                        result.append([nums[i], nums[j], nums[left], nums[right]])
                        
                        while left < right and nums[left] == nums[left + 1]:
                            left += 1
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1
                        
                        left += 1
                        right -= 1
                    elif current_sum < target:
                        left += 1
                    else:
                        right -= 1
        
        return result
    
    def container_with_most_water(self, height: List[int]) -> int:
        """LC 11: Container With Most Water
        Time: O(n), Space: O(1)
        """
        left, right = 0, len(height) - 1
        max_area = 0
        
        while left < right:
            width = right - left
            current_area = width * min(height[left], height[right])
            max_area = max(max_area, current_area)
            
            # Move pointer with smaller height
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        
        return max_area
    
    def trapping_rain_water(self, height: List[int]) -> int:
        """LC 42: Trapping Rain Water
        Time: O(n), Space: O(1)
        """
        if not height:
            return 0
        
        left, right = 0, len(height) - 1
        left_max = right_max = 0
        water_trapped = 0
        
        while left < right:
            if height[left] < height[right]:
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    water_trapped += left_max - height[left]
                left += 1
            else:
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    water_trapped += right_max - height[right]
                right -= 1
        
        return water_trapped
    
    # ==========================================
    # 2. SAME DIRECTION TWO POINTERS
    # ==========================================
    
    def remove_duplicates(self, nums: List[int]) -> int:
        """LC 26: Remove Duplicates from Sorted Array
        Time: O(n), Space: O(1)
        """
        if not nums:
            return 0
        
        write_index = 1
        
        for read_index in range(1, len(nums)):
            if nums[read_index] != nums[read_index - 1]:
                nums[write_index] = nums[read_index]
                write_index += 1
        
        return write_index
    
    def remove_duplicates_ii(self, nums: List[int]) -> int:
        """LC 80: Remove Duplicates II - Allow at most 2 duplicates
        Time: O(n), Space: O(1)
        """
        if len(nums) <= 2:
            return len(nums)
        
        write_index = 2
        
        for read_index in range(2, len(nums)):
            if nums[read_index] != nums[write_index - 2]:
                nums[write_index] = nums[read_index]
                write_index += 1
        
        return write_index
    
    def remove_element(self, nums: List[int], val: int) -> int:
        """LC 27: Remove Element
        Time: O(n), Space: O(1)
        """
        write_index = 0
        
        for read_index in range(len(nums)):
            if nums[read_index] != val:
                nums[write_index] = nums[read_index]
                write_index += 1
        
        return write_index
    
    def move_zeros(self, nums: List[int]) -> None:
        """LC 283: Move Zeroes
        Time: O(n), Space: O(1)
        """
        write_index = 0
        
        # Move non-zero elements
        for read_index in range(len(nums)):
            if nums[read_index] != 0:
                nums[write_index] = nums[read_index]
                write_index += 1
        
        # Fill remaining with zeros
        while write_index < len(nums):
            nums[write_index] = 0
            write_index += 1
    
    # ==========================================
    # 3. FAST-SLOW POINTERS (FLOYD'S ALGORITHM)
    # ==========================================
    
    def find_duplicate(self, nums: List[int]) -> int:
        """LC 287: Find the Duplicate Number - Floyd's Cycle Detection
        Time: O(n), Space: O(1)
        """
        # Phase 1: Find intersection point
        slow = fast = nums[0]
        
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        
        # Phase 2: Find entrance to cycle
        slow = nums[0]
        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]
        
        return slow
    
    def find_duplicates_all(self, nums: List[int]) -> List[int]:
        """LC 442: Find All Duplicates - Numbers 1 to n
        Time: O(n), Space: O(1)
        """
        result = []
        
        for num in nums:
            index = abs(num) - 1
            if nums[index] < 0:
                result.append(abs(num))
            else:
                nums[index] = -nums[index]
        
        return result
    
    def find_disappeared_numbers(self, nums: List[int]) -> List[int]:
        """LC 448: Find All Numbers Disappeared
        Time: O(n), Space: O(1)
        """
        # Mark present numbers
        for num in nums:
            index = abs(num) - 1
            nums[index] = -abs(nums[index])
        
        # Find missing numbers
        result = []
        for i in range(len(nums)):
            if nums[i] > 0:
                result.append(i + 1)
        
        return result
    
    # ==========================================
    # 4. WINDOW-BASED TWO POINTERS
    # ==========================================
    
    def subarray_sum_equals_k(self, nums: List[int], k: int) -> int:
        """LC 560: Subarray Sum Equals K
        Time: O(n), Space: O(n)
        """
        from collections import defaultdict
        
        count = 0
        current_sum = 0
        sum_count = defaultdict(int)
        sum_count[0] = 1
        
        for num in nums:
            current_sum += num
            count += sum_count[current_sum - k]
            sum_count[current_sum] += 1
        
        return count
    
    def max_consecutive_ones_iii(self, nums: List[int], k: int) -> int:
        """LC 1004: Max Consecutive Ones III
        Time: O(n), Space: O(1)
        """
        left = 0
        zero_count = 0
        max_length = 0
        
        for right in range(len(nums)):
            if nums[right] == 0:
                zero_count += 1
            
            while zero_count > k:
                if nums[left] == 0:
                    zero_count -= 1
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    def longest_subarray_delete_one(self, nums: List[int]) -> int:
        """LC 1493: Longest Subarray of 1's After Deleting One Element
        Time: O(n), Space: O(1)
        """
        left = 0
        zero_count = 0
        max_length = 0
        
        for right in range(len(nums)):
            if nums[right] == 0:
                zero_count += 1
            
            while zero_count > 1:
                if nums[left] == 0:
                    zero_count -= 1
                left += 1
            
            # -1 because we must delete one element
            max_length = max(max_length, right - left)
        
        return max_length

# Test Examples
def run_examples():
    atp = ArrayTwoPointers()
    
    print("=== ARRAY TWO POINTERS EXAMPLES ===\n")
    
    # Two Sum
    print("1. TWO SUM PROBLEMS:")
    sorted_arr = [2, 7, 11, 15]
    target = 9
    print(f"Two Sum (sorted): {sorted_arr}, target={target}")
    print(f"Result: {atp.two_sum_sorted(sorted_arr, target)}")
    
    # Three Sum
    print("\n2. THREE SUM PROBLEMS:")
    nums = [-1, 0, 1, 2, -1, -4]
    print(f"Three Sum: {nums}")
    print(f"Result: {atp.three_sum(nums)}")
    
    # Container with water
    print("\n3. CONTAINER WITH MOST WATER:")
    heights = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    print(f"Heights: {heights}")
    print(f"Max area: {atp.container_with_most_water(heights)}")
    
    # Rain water trapping
    print("\n4. TRAPPING RAIN WATER:")
    heights = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    print(f"Heights: {heights}")
    print(f"Water trapped: {atp.trapping_rain_water(heights)}")
    
    # Remove duplicates
    print("\n5. REMOVE DUPLICATES:")
    arr_dup = [1, 1, 2, 2, 2, 3, 3]
    print(f"Original: {arr_dup}")
    new_length = atp.remove_duplicates(arr_dup.copy())
    print(f"After removing duplicates: {arr_dup[:new_length]}")
    
    # Find duplicate
    print("\n6. FIND DUPLICATE:")
    nums_dup = [1, 3, 4, 2, 2]
    print(f"Array: {nums_dup}")
    print(f"Duplicate: {atp.find_duplicate(nums_dup)}")

if __name__ == "__main__":
    run_examples() 