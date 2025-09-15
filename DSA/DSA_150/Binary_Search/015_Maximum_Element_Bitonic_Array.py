"""
Problem: Find Maximum Element in a Bitonic Array
URL: https://leetcode.com/problems/peak-index-in-a-mountain-array/description

Problem Statement:
A bitonic array is an array that is first increasing and then decreasing. Find the maximum element in the array.

Sample Input/Output:
Input: arr = [1, 3, 8, 12, 4, 2]
Output: 12
Explanation: Array increases till 12 then decreases

Input: arr = [3, 4, 5, 1]
Output: 5
Explanation: Maximum element is 5
"""

from typing import List

class Solution:
    def Find_Max_In_Bitonic_Linear(self, arr: List[int]) -> int:
        """
        Linear Search Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        return max(arr)
    
    def Find_Max_In_Bitonic_Binary_Search_Optimal(self, arr: List[int]) -> int:
        """
        Binary Search Optimal Approach
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        left, right = 0, len(arr) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            if arr[mid] > arr[mid + 1]:
                right = mid
            else:
                left = mid + 1
        
        return arr[left]

def Test_Max_Bitonic():
    solution = Solution()
    
    test_cases = [
        ([1, 3, 8, 12, 4, 2], 12),
        ([3, 4, 5, 1], 5),
        ([1, 2, 3, 4, 5, 3, 1], 5),
        ([10, 20, 15, 2, 23, 90, 67], 90)
    ]
    
    for arr, expected in test_cases:
        result1 = solution.Find_Max_In_Bitonic_Linear(arr.copy())
        result2 = solution.Find_Max_In_Bitonic_Binary_Search_Optimal(arr.copy())
        
        print(f"Array: {arr}")
        print(f"Expected: {expected}")
        print(f"Linear: {result1}")
        print(f"Binary Search: {result2}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Max_Bitonic()
