"""
Problem: Find an Element in a Rotated Sorted Array
URL: https://leetcode.com/problems/search-in-rotated-sorted-array/description/

Problem Statement:
There is an integer array nums sorted in ascending order (with distinct values).
Prior to being passed to your function, nums is possibly rotated at some pivot index.
Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

Sample Input/Output:
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
Explanation: Target 0 is at index 4

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
Explanation: Target 3 is not in the array
"""

from typing import List

class Solution:
    def Search_Rotated_Array_Linear_Search(self, nums: List[int], target: int) -> int:
        """
        Linear Search Approach - Sequential search
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        for i in range(len(nums)):
            if nums[i] == target:
                return i
        return -1
    
    def Search_Rotated_Array_Find_Pivot_Then_Search(self, nums: List[int], target: int) -> int:
        """
        Find Pivot Then Search - Two phase approach
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        def Find_Pivot():
            left, right = 0, len(nums) - 1
            
            while left < right:
                mid = left + (right - left) // 2
                
                if nums[mid] > nums[right]:
                    left = mid + 1
                else:
                    right = mid
            
            return left
        
        def Binary_Search(left: int, right: int) -> int:
            while left <= right:
                mid = left + (right - left) // 2
                
                if nums[mid] == target:
                    return mid
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return -1
        
        if not nums:
            return -1
        
        pivot = Find_Pivot()
        
        result = Binary_Search(0, pivot - 1)
        if result != -1:
            return result
        
        return Binary_Search(pivot, len(nums) - 1)
    
    def Search_Rotated_Array_Binary_Search_Optimal(self, nums: List[int], target: int) -> int:
        """
        Binary Search Optimal - Single pass binary search
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if not nums:
            return -1
        
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return mid
            
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1
    
    def Search_Rotated_Array_Modified_Binary_Search(self, nums: List[int], target: int) -> int:
        """
        Modified Binary Search - Handle rotation cases
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if not nums:
            return -1
        
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if nums[mid] == target:
                return mid
            
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
        
        return -1
    
    def Search_Rotated_Array_Recursive(self, nums: List[int], target: int) -> int:
        """
        Recursive Approach - Recursive binary search
        Time Complexity: O(log n)
        Space Complexity: O(log n)
        """
        def Search_Helper(left: int, right: int) -> int:
            if left > right:
                return -1
            
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return mid
            
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    return Search_Helper(left, mid - 1)
                else:
                    return Search_Helper(mid + 1, right)
            else:
                if nums[mid] < target <= nums[right]:
                    return Search_Helper(mid + 1, right)
                else:
                    return Search_Helper(left, mid - 1)
        
        if not nums:
            return -1
        
        return Search_Helper(0, len(nums) - 1)
    
    def Search_Rotated_Array_Edge_Cases(self, nums: List[int], target: int) -> int:
        """
        Edge Cases Handling - Handle duplicates and edge cases
        Time Complexity: O(log n) average, O(n) worst case
        Space Complexity: O(1)
        """
        if not nums:
            return -1
        
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return mid
            
            if nums[left] == nums[mid] == nums[right]:
                left += 1
                right -= 1
                continue
            
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1

def Test_Search_Rotated_Array():
    solution = Solution()
    
    test_cases = [
        ([4,5,6,7,0,1,2], 0, 4),
        ([4,5,6,7,0,1,2], 3, -1),
        ([1], 0, -1),
        ([1], 1, 0),
        ([1,3], 3, 1),
        ([3,1], 1, 1),
        ([2,5,6,0,0,1,2], 0, 3)
    ]
    
    for nums, target, expected in test_cases:
        result1 = solution.Search_Rotated_Array_Linear_Search(nums.copy(), target)
        result2 = solution.Search_Rotated_Array_Find_Pivot_Then_Search(nums.copy(), target)
        result3 = solution.Search_Rotated_Array_Binary_Search_Optimal(nums.copy(), target)
        result4 = solution.Search_Rotated_Array_Modified_Binary_Search(nums.copy(), target)
        result5 = solution.Search_Rotated_Array_Recursive(nums.copy(), target)
        result6 = solution.Search_Rotated_Array_Edge_Cases(nums.copy(), target)
        
        print(f"Array: {nums}, Target: {target}")
        print(f"Expected: {expected}")
        print(f"Linear Search: {result1}")
        print(f"Find Pivot Then Search: {result2}")
        print(f"Binary Search Optimal: {result3}")
        print(f"Modified Binary Search: {result4}")
        print(f"Recursive: {result5}")
        print(f"Edge Cases: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Search_Rotated_Array()
