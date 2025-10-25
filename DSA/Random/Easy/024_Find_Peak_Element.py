"""
Problem: Find Peak Element
URL: https://leetcode.com/problems/find-peak-element/

Problem Statement:
A peak element is an element that is strictly greater than its neighbors.

Given a 0-indexed integer array nums, find a peak element, and return its index. 
If the array contains multiple peaks, return the index to any of the peaks.

You may imagine that nums[-1] = nums[n] = -∞. In other words, an element is always 
considered to be strictly greater than a neighbor that is outside the array.

You must write an algorithm that runs in O(log n) time.

Sample Input/Output:
Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.

Input: nums = [1,2,1,3,5,6,4]
Output: 5
Explanation: Your function can return either index 1 where the peak element is 2, 
             or index 5 where the peak element is 6.

Input: nums = [1]
Output: 0
"""

from typing import List

class Solution:
    def Find_Peak_Linear(self, nums: List[int]) -> int:
        """
        Linear Search Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(nums)
        
        if n == 1:
            return 0
        
        if nums[0] > nums[1]:
            return 0
        
        if nums[n - 1] > nums[n - 2]:
            return n - 1
        
        for i in range(1, n - 1):
            if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
                return i
        
        return 0
    
    def Find_Peak_Binary_Search(self, nums: List[int]) -> int:
        """
        Binary Search Approach - Optimal solution
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = (left + right) // 2
            
            if nums[mid] > nums[mid + 1]:
                right = mid
            else:
                left = mid + 1
        
        return left
    
    def Find_Peak_Recursive(self, nums: List[int]) -> int:
        """
        Recursive Binary Search
        Time Complexity: O(log n)
        Space Complexity: O(log n)
        """
        def Search(left: int, right: int) -> int:
            if left == right:
                return left
            
            mid = (left + right) // 2
            
            if nums[mid] > nums[mid + 1]:
                return Search(left, mid)
            else:
                return Search(mid + 1, right)
        
        return Search(0, len(nums) - 1)
    
    def Find_Peak_Ascending(self, nums: List[int]) -> int:
        """
        Find First Descending Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        for i in range(len(nums) - 1):
            if nums[i] > nums[i + 1]:
                return i
        
        return len(nums) - 1
    
    def Find_Peak_Max_Element(self, nums: List[int]) -> int:
        """
        Find Maximum Element - Simple approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        return nums.index(max(nums))

def Test_Find_Peak():
    solution = Solution()
    
    test_cases = [
        ([1,2,3,1], [2]),
        ([1,2,1,3,5,6,4], [1, 5]),
        ([1], [0]),
        ([1,2], [1]),
        ([2,1], [0])
    ]
    
    for nums, valid_peaks in test_cases:
        result1 = solution.Find_Peak_Linear(nums.copy())
        result2 = solution.Find_Peak_Binary_Search(nums.copy())
        result3 = solution.Find_Peak_Recursive(nums.copy())
        result4 = solution.Find_Peak_Ascending(nums.copy())
        result5 = solution.Find_Peak_Max_Element(nums.copy())
        
        print(f"Array: {nums}")
        print(f"Valid Peaks: {valid_peaks}")
        print(f"Linear: {result1} (Valid: {result1 in valid_peaks})")
        print(f"Binary Search: {result2} (Valid: {result2 in valid_peaks})")
        print(f"Recursive: {result3} (Valid: {result3 in valid_peaks})")
        print(f"Ascending: {result4} (Valid: {result4 in valid_peaks})")
        print(f"Max Element: {result5} (Valid: {result5 in valid_peaks})")
        print("-" * 50)

if __name__ == "__main__":
    Test_Find_Peak()

