"""
Problem: Minimum Difference Element in a Sorted Array
URL: https://leetcode.com/problems/minimum-absolute-difference/

Problem Statement:
Given a sorted array, find the element in the array such that its difference with the given number is minimum.

Sample Input/Output:
Input: arr = [1, 3, 8, 10, 15], target = 12
Output: 10
Explanation: |12-10| = 2 is minimum among all differences

Input: arr = [4, 6, 10], target = 7
Output: 6
Explanation: |7-6| = 1 is minimum
"""

from typing import List

class Solution:
    def Find_Min_Diff_Element_Linear(self, arr: List[int], target: int) -> int:
        """
        Linear Search Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        min_diff = float('inf')
        result = arr[0]
        
        for num in arr:
            diff = abs(num - target)
            if diff < min_diff:
                min_diff = diff
                result = num
        
        return result
    
    def Find_Min_Diff_Element_Binary_Search_Optimal(self, arr: List[int], target: int) -> int:
        """
        Binary Search Optimal Approach
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] == target:
                return arr[mid]
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        if left >= len(arr):
            return arr[right]
        if right < 0:
            return arr[left]
        
        if abs(arr[left] - target) < abs(arr[right] - target):
            return arr[left]
        else:
            return arr[right]

def Test_Min_Diff_Element():
    solution = Solution()
    
    test_cases = [
        ([1, 3, 8, 10, 15], 12, 10),
        ([4, 6, 10], 7, 6),
        ([1, 2, 3, 4, 5], 3, 3),
        ([2, 5, 10, 12, 15], 6, 5)
    ]
    
    for arr, target, expected in test_cases:
        result1 = solution.Find_Min_Diff_Element_Linear(arr.copy(), target)
        result2 = solution.Find_Min_Diff_Element_Binary_Search_Optimal(arr.copy(), target)
        
        print(f"Array: {arr}, Target: {target}")
        print(f"Expected: {expected}")
        print(f"Linear: {result1}")
        print(f"Binary Search: {result2}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Min_Diff_Element()
