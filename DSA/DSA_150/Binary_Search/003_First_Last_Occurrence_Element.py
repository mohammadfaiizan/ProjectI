"""
Problem: First and Last occurrence of an element in a Sorted Array
URL: https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/

Problem Statement:
Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.
If target is not found in the array, return [-1, -1].

Sample Input/Output:
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
Explanation: Target 8 appears at indices 3 and 4

Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
Explanation: Target 6 is not found
"""

from typing import List

class Solution:
    def Search_Range_Linear_Search(self, nums: List[int], target: int) -> List[int]:
        """
        Linear Search Approach - Scan entire array
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        first = last = -1
        
        for i in range(len(nums)):
            if nums[i] == target:
                if first == -1:
                    first = i
                last = i
        
        return [first, last]
    
    def Search_Range_Two_Scans(self, nums: List[int], target: int) -> List[int]:
        """
        Two Scans Approach - Forward and backward scan
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        first = last = -1
        
        for i in range(len(nums)):
            if nums[i] == target:
                first = i
                break
        
        if first == -1:
            return [-1, -1]
        
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] == target:
                last = i
                break
        
        return [first, last]
    
    def Search_Range_Binary_Search_Optimal(self, nums: List[int], target: int) -> List[int]:
        """
        Binary Search Approach - Find first and last separately
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        def Find_First():
            left, right = 0, len(nums) - 1
            first = -1
            
            while left <= right:
                mid = left + (right - left) // 2
                
                if nums[mid] == target:
                    first = mid
                    right = mid - 1
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return first
        
        def Find_Last():
            left, right = 0, len(nums) - 1
            last = -1
            
            while left <= right:
                mid = left + (right - left) // 2
                
                if nums[mid] == target:
                    last = mid
                    left = mid + 1
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return last
        
        first = Find_First()
        if first == -1:
            return [-1, -1]
        
        last = Find_Last()
        return [first, last]
    
    def Search_Range_Modified_Binary_Search(self, nums: List[int], target: int) -> List[int]:
        """
        Modified Binary Search - Find leftmost and rightmost
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        def Binary_Search(is_first: bool) -> int:
            left, right = 0, len(nums) - 1
            result = -1
            
            while left <= right:
                mid = (left + right) // 2
                
                if nums[mid] == target:
                    result = mid
                    if is_first:
                        right = mid - 1
                    else:
                        left = mid + 1
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return result
        
        first = Binary_Search(True)
        last = Binary_Search(False)
        
        return [first, last]
    
    def Search_Range_Lower_Upper_Bound(self, nums: List[int], target: int) -> List[int]:
        """
        Lower Upper Bound Approach - Using bisect logic
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        def Lower_Bound(target: int) -> int:
            left, right = 0, len(nums)
            
            while left < right:
                mid = (left + right) // 2
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid
            
            return left
        
        def Upper_Bound(target: int) -> int:
            left, right = 0, len(nums)
            
            while left < right:
                mid = (left + right) // 2
                if nums[mid] <= target:
                    left = mid + 1
                else:
                    right = mid
            
            return left
        
        first = Lower_Bound(target)
        if first >= len(nums) or nums[first] != target:
            return [-1, -1]
        
        last = Upper_Bound(target) - 1
        return [first, last]
    
    def Search_Range_Built_In(self, nums: List[int], target: int) -> List[int]:
        """
        Built-in Approach - Using bisect module
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        import bisect
        
        first = bisect.bisect_left(nums, target)
        if first >= len(nums) or nums[first] != target:
            return [-1, -1]
        
        last = bisect.bisect_right(nums, target) - 1
        return [first, last]

def Test_Search_Range():
    solution = Solution()
    
    test_cases = [
        ([5,7,7,8,8,10], 8, [3,4]),
        ([5,7,7,8,8,10], 6, [-1,-1]),
        ([], 0, [-1,-1]),
        ([1], 1, [0,0]),
        ([1,1,1,1,1], 1, [0,4]),
        ([2,2], 2, [0,1])
    ]
    
    for nums, target, expected in test_cases:
        result1 = solution.Search_Range_Linear_Search(nums.copy(), target)
        result2 = solution.Search_Range_Two_Scans(nums.copy(), target)
        result3 = solution.Search_Range_Binary_Search_Optimal(nums.copy(), target)
        result4 = solution.Search_Range_Modified_Binary_Search(nums.copy(), target)
        result5 = solution.Search_Range_Lower_Upper_Bound(nums.copy(), target)
        result6 = solution.Search_Range_Built_In(nums.copy(), target)
        
        print(f"Array: {nums}, Target: {target}")
        print(f"Expected: {expected}")
        print(f"Linear Search: {result1}")
        print(f"Two Scans: {result2}")
        print(f"Binary Search Optimal: {result3}")
        print(f"Modified Binary Search: {result4}")
        print(f"Lower Upper Bound: {result5}")
        print(f"Built-in: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Search_Range()
