"""
Problem: Index of First 1 in a Sorted Infinite Array
URL: https://www.geeksforgeeks.org/problems/index-of-first-1-in-a-sorted-array-of-0s-and-1s4048/1

Problem Statement:
Given an infinite sorted array consisting of 0s and 1s, find the index of first 1.

Sample Input/Output:
Input: arr = [0, 0, 1, 1, 1, 1, 1, 1, 1]
Output: 2
Explanation: First 1 appears at index 2
"""

from typing import List

class Solution:
    def First_One_In_Infinite_Array(self, arr: List[int]) -> int:
        """
        Exponential Search + Binary Search for First 1
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if arr[0] == 1:
            return 0
        
        bound = 1
        while bound < len(arr) and arr[bound] == 0:
            bound *= 2
        
        left = bound // 2
        right = min(bound, len(arr) - 1)
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] == 1:
                if mid == 0 or arr[mid - 1] == 0:
                    return mid
                right = mid - 1
            else:
                left = mid + 1
        
        return -1

def Test_First_One_Infinite():
    solution = Solution()
    
    test_cases = [
        ([0, 0, 1, 1, 1, 1, 1, 1, 1], 2),
        ([0, 1, 1, 1, 1], 1),
        ([1, 1, 1, 1, 1], 0)
    ]
    
    for arr, expected in test_cases:
        result = solution.First_One_In_Infinite_Array(arr)
        
        print(f"Array: {arr}")
        print(f"Expected: {expected}")
        print(f"First One Index: {result}")
        print("-" * 50)

if __name__ == "__main__":
    Test_First_One_Infinite()
