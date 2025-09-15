"""
Problem: Find Position of an Element in an Infinite Sorted Array
URL: https://takeuforward.org/data-structure/search-in-an-infinite-sorted-array/

Problem Statement:
Given an infinite sorted array and a target, find the position of the target in the array.

Sample Input/Output:
Input: arr = [3, 5, 7, 9, 10, 90, 100, 130, 140, 160, 170], target = 10
Output: 4
Explanation: Target 10 is at index 4
"""

from typing import List

class Solution:
    def Search_In_Infinite_Array_Exponential_Search(self, arr: List[int], target: int) -> int:
        """
        Exponential Search + Binary Search
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if arr[0] == target:
            return 0
        
        bound = 1
        while bound < len(arr) and arr[bound] < target:
            bound *= 2
        
        left = bound // 2
        right = min(bound, len(arr) - 1)
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1

def Test_Search_Infinite_Array():
    solution = Solution()
    
    test_cases = [
        ([3, 5, 7, 9, 10, 90, 100, 130, 140, 160, 170], 10, 4),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 8, 7),
        ([1, 3, 5, 7, 9, 11, 13, 15], 13, 6)
    ]
    
    for arr, target, expected in test_cases:
        result = solution.Search_In_Infinite_Array_Exponential_Search(arr, target)
        
        print(f"Array: {arr}, Target: {target}")
        print(f"Expected: {expected}")
        print(f"Exponential Search: {result}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Search_Infinite_Array()
