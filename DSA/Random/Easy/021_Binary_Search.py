"""
Problem: Binary Search
URL: https://leetcode.com/problems/binary-search/

Problem Statement:
Given an array of integers nums which is sorted in ascending order, and an integer target,
write a function to search target in nums. If target exists, return its index. Otherwise, return -1.

You must write an algorithm with O(log n) runtime complexity.

Sample Input/Output:
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4

Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
Explanation: 2 does not exist in nums so return -1

Input: nums = [5], target = 5
Output: 0
"""

from typing import List

class Solution:
    def Binary_Search_Iterative(self, nums: List[int], target: int) -> int:
        """
        Iterative Binary Search - Optimal solution
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    
    def Binary_Search_Recursive(self, nums: List[int], target: int) -> int:
        """
        Recursive Binary Search
        Time Complexity: O(log n)
        Space Complexity: O(log n)
        """
        def Search(left: int, right: int) -> int:
            if left > right:
                return -1
            
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                return Search(mid + 1, right)
            else:
                return Search(left, mid - 1)
        
        return Search(0, len(nums) - 1)
    
    def Binary_Search_Template_1(self, nums: List[int], target: int) -> int:
        """
        Binary Search Template 1
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
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    
    def Binary_Search_Template_2(self, nums: List[int], target: int) -> int:
        """
        Binary Search Template 2
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if not nums:
            return -1
        
        left, right = 0, len(nums)
        
        while left < right:
            mid = (left + right) // 2
            
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        
        if left < len(nums) and nums[left] == target:
            return left
        
        return -1
    
    def Binary_Search_Built_In(self, nums: List[int], target: int) -> int:
        """
        Built-in Binary Search using bisect
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        import bisect
        
        index = bisect.bisect_left(nums, target)
        
        if index < len(nums) and nums[index] == target:
            return index
        
        return -1
    
    def Binary_Search_Linear(self, nums: List[int], target: int) -> int:
        """
        Linear Search Approach - For comparison
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        for i in range(len(nums)):
            if nums[i] == target:
                return i
        
        return -1

def Test_Binary_Search():
    solution = Solution()
    
    test_cases = [
        ([-1,0,3,5,9,12], 9, 4),
        ([-1,0,3,5,9,12], 2, -1),
        ([5], 5, 0),
        ([5], -5, -1),
        ([1,2,3,4,5], 1, 0),
        ([1,2,3,4,5], 5, 4)
    ]
    
    for nums, target, expected in test_cases:
        result1 = solution.Binary_Search_Iterative(nums.copy(), target)
        result2 = solution.Binary_Search_Recursive(nums.copy(), target)
        result3 = solution.Binary_Search_Template_1(nums.copy(), target)
        result4 = solution.Binary_Search_Template_2(nums.copy(), target)
        result5 = solution.Binary_Search_Built_In(nums.copy(), target)
        result6 = solution.Binary_Search_Linear(nums.copy(), target)
        
        print(f"Array: {nums}, Target: {target}")
        print(f"Expected: {expected}")
        print(f"Iterative: {result1}")
        print(f"Recursive: {result2}")
        print(f"Template 1: {result3}")
        print(f"Template 2: {result4}")
        print(f"Built-in: {result5}")
        print(f"Linear: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Binary_Search()

