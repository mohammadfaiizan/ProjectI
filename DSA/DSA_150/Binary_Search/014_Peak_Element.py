"""
Problem: Peak Element
URL: https://leetcode.com/problems/find-peak-element/description/

Problem Statement:
A peak element is an element that is strictly greater than its neighbors. Given an integer array nums, find a peak element, and return its index.

Sample Input/Output:
Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return index 2.

Input: nums = [1,2,1,3,5,6,4]
Output: 5
Explanation: Your function can return either index 1 or 5 where nums[1] = 2 or nums[5] = 6 are peak elements.
"""

from typing import List

class Solution:
    def Find_Peak_Element_Linear_Search(self, nums: List[int]) -> int:
        """
        Linear Search Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(nums)
        
        for i in range(n):
            left_ok = i == 0 or nums[i] > nums[i-1]
            right_ok = i == n-1 or nums[i] > nums[i+1]
            
            if left_ok and right_ok:
                return i
        
        return -1
    
    def Find_Peak_Element_Binary_Search_Optimal(self, nums: List[int]) -> int:
        """
        Binary Search Optimal Approach
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            if nums[mid] > nums[mid + 1]:
                right = mid
            else:
                left = mid + 1
        
        return left

def Test_Find_Peak_Element():
    solution = Solution()
    
    test_cases = [
        [1,2,3,1],
        [1,2,1,3,5,6,4],
        [1],
        [1,2]
    ]
    
    for nums in test_cases:
        result1 = solution.Find_Peak_Element_Linear_Search(nums.copy())
        result2 = solution.Find_Peak_Element_Binary_Search_Optimal(nums.copy())
        
        print(f"Array: {nums}")
        print(f"Linear Search: {result1}")
        print(f"Binary Search Optimal: {result2}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Find_Peak_Element()
