"""
Problem: Find Ceil of an Element in a Sorted Array
URL: https://www.naukri.com/code360/problems/ceiling-in-a-sorted-array_1825401

Problem Statement:
Given a sorted array and a value x, find the ceil of x in the array. Ceil of x is the smallest element in array greater than or equal to x.

Sample Input/Output:
Input: arr = [1, 2, 8, 10, 10, 12, 19], x = 5
Output: 8
Explanation: Ceil of 5 is 8 (smallest element >= 5)

Input: arr = [1, 2, 8, 10, 10, 12, 19], x = 1
Output: 1
Explanation: Ceil of 1 is 1
"""

from typing import List

class Solution:
    def Find_Ceil_Linear_Search(self, arr: List[int], x: int) -> int:
        """
        Linear Search Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        for num in arr:
            if num >= x:
                return num
        return -1
    
    def Find_Ceil_Binary_Search_Optimal(self, arr: List[int], x: int) -> int:
        """
        Binary Search Optimal Approach
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        left, right = 0, len(arr) - 1
        ceil = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] >= x:
                ceil = arr[mid]
                right = mid - 1
            else:
                left = mid + 1
        
        return ceil

def Test_Find_Ceil():
    solution = Solution()
    
    test_cases = [
        ([1, 2, 8, 10, 10, 12, 19], 5, 8),
        ([1, 2, 8, 10, 10, 12, 19], 1, 1),
        ([1, 2, 8, 10, 10, 12, 19], 20, -1),
        ([1, 2, 8, 10, 10, 12, 19], 10, 10)
    ]
    
    for arr, x, expected in test_cases:
        result1 = solution.Find_Ceil_Linear_Search(arr.copy(), x)
        result2 = solution.Find_Ceil_Binary_Search_Optimal(arr.copy(), x)
        
        print(f"Array: {arr}, x: {x}")
        print(f"Expected: {expected}")
        print(f"Linear Search: {result1}")
        print(f"Binary Search Optimal: {result2}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Find_Ceil()
