"""
Problem: Find Floor of an Element in a Sorted Array
URL: https://www.geeksforgeeks.org/problems/floor-in-a-sorted-array-1587115620/1

Problem Statement:
Given a sorted array and a value x, find the floor of x in the array. Floor of x is the largest element in array smaller than or equal to x.

Sample Input/Output:
Input: arr = [1, 2, 8, 10, 10, 12, 19], x = 5
Output: 2
Explanation: Floor of 5 is 2 (largest element <= 5)

Input: arr = [1, 2, 8, 10, 10, 12, 19], x = 20
Output: 19
Explanation: Floor of 20 is 19
"""

from typing import List

class Solution:
    def Find_Floor_Linear_Search(self, arr: List[int], x: int) -> int:
        """
        Linear Search Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        floor = -1
        for num in arr:
            if num <= x:
                floor = num
            else:
                break
        return floor
    
    def Find_Floor_Binary_Search_Optimal(self, arr: List[int], x: int) -> int:
        """
        Binary Search Optimal Approach
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        left, right = 0, len(arr) - 1
        floor = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] <= x:
                floor = arr[mid]
                left = mid + 1
            else:
                right = mid - 1
        
        return floor

def Test_Find_Floor():
    solution = Solution()
    
    test_cases = [
        ([1, 2, 8, 10, 10, 12, 19], 5, 2),
        ([1, 2, 8, 10, 10, 12, 19], 20, 19),
        ([1, 2, 8, 10, 10, 12, 19], 0, -1),
        ([1, 2, 8, 10, 10, 12, 19], 10, 10)
    ]
    
    for arr, x, expected in test_cases:
        result1 = solution.Find_Floor_Linear_Search(arr.copy(), x)
        result2 = solution.Find_Floor_Binary_Search_Optimal(arr.copy(), x)
        
        print(f"Array: {arr}, x: {x}")
        print(f"Expected: {expected}")
        print(f"Linear Search: {result1}")
        print(f"Binary Search Optimal: {result2}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Find_Floor()
