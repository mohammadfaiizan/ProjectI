"""
Problem: How many times is a Sorted Array Rotated
URL: https://www.geeksforgeeks.org/problems/rotation4723/1

Problem Statement:
Given a sorted array that has been rotated some number of times, find how many times the array was rotated.
The rotation count is equal to the index of the minimum element.

Sample Input/Output:
Input: arr = [15, 18, 2, 3, 6, 12]
Output: 2
Explanation: Array was rotated 2 times. Original array: [2, 3, 6, 12, 15, 18]

Input: arr = [7, 9, 11, 12, 5]
Output: 4
Explanation: Array was rotated 4 times. Original array: [5, 7, 9, 11, 12]
"""

from typing import List

class Solution:
    def Find_Rotation_Count_Linear_Search(self, arr: List[int]) -> int:
        """
        Linear Search Approach - Find minimum element linearly
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not arr:
            return 0
        
        min_index = 0
        for i in range(1, len(arr)):
            if arr[i] < arr[min_index]:
                min_index = i
        
        return min_index
    
    def Find_Rotation_Count_Compare_Adjacent(self, arr: List[int]) -> int:
        """
        Compare Adjacent Approach - Find break point
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not arr or len(arr) == 1:
            return 0
        
        n = len(arr)
        
        for i in range(n):
            next_index = (i + 1) % n
            if arr[i] > arr[next_index]:
                return next_index
        
        return 0
    
    def Find_Rotation_Count_Binary_Search_Optimal(self, arr: List[int]) -> int:
        """
        Binary Search Optimal Approach - Find minimum element
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if not arr:
            return 0
        
        left, right = 0, len(arr) - 1
        
        while left <= right:
            if arr[left] <= arr[right]:
                return left
            
            mid = left + (right - left) // 2
            next_mid = (mid + 1) % len(arr)
            prev_mid = (mid + len(arr) - 1) % len(arr)
            
            if arr[mid] <= arr[next_mid] and arr[mid] <= arr[prev_mid]:
                return mid
            
            if arr[mid] >= arr[left]:
                left = mid + 1
            else:
                right = mid - 1
        
        return 0
    
    def Find_Rotation_Count_Modified_Binary_Search(self, arr: List[int]) -> int:
        """
        Modified Binary Search - Compare with boundaries
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if not arr:
            return 0
        
        n = len(arr)
        left, right = 0, n - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            if arr[mid] > arr[right]:
                left = mid + 1
            elif arr[mid] < arr[right]:
                right = mid
            else:
                right -= 1
        
        return left
    
    def Find_Rotation_Count_Pivot_Search(self, arr: List[int]) -> int:
        """
        Pivot Search Approach - Find pivot element
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if not arr or len(arr) == 1:
            return 0
        
        n = len(arr)
        left, right = 0, n - 1
        
        while left <= right:
            if arr[left] <= arr[right]:
                return left
            
            mid = (left + right) // 2
            
            if mid < n - 1 and arr[mid] > arr[mid + 1]:
                return mid + 1
            
            if mid > 0 and arr[mid] < arr[mid - 1]:
                return mid
            
            if arr[left] <= arr[mid]:
                left = mid + 1
            else:
                right = mid - 1
        
        return 0
    
    def Find_Rotation_Count_Minimum_Element(self, arr: List[int]) -> int:
        """
        Minimum Element Approach - Direct minimum search
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if not arr:
            return 0
        
        left, right = 0, len(arr) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            if arr[mid] > arr[right]:
                left = mid + 1
            else:
                right = mid
        
        return left

def Test_Rotation_Count():
    solution = Solution()
    
    test_cases = [
        ([15, 18, 2, 3, 6, 12], 2),
        ([7, 9, 11, 12, 5], 4),
        ([1, 2, 3, 4, 5], 0),
        ([2, 1], 1),
        ([1], 0),
        ([3, 4, 5, 1, 2], 3),
        ([4, 5, 6, 7, 0, 1, 2], 4)
    ]
    
    for arr, expected in test_cases:
        result1 = solution.Find_Rotation_Count_Linear_Search(arr.copy())
        result2 = solution.Find_Rotation_Count_Compare_Adjacent(arr.copy())
        result3 = solution.Find_Rotation_Count_Binary_Search_Optimal(arr.copy())
        result4 = solution.Find_Rotation_Count_Modified_Binary_Search(arr.copy())
        result5 = solution.Find_Rotation_Count_Pivot_Search(arr.copy())
        result6 = solution.Find_Rotation_Count_Minimum_Element(arr.copy())
        
        print(f"Array: {arr}")
        print(f"Expected: {expected}")
        print(f"Linear Search: {result1}")
        print(f"Compare Adjacent: {result2}")
        print(f"Binary Search Optimal: {result3}")
        print(f"Modified Binary Search: {result4}")
        print(f"Pivot Search: {result5}")
        print(f"Minimum Element: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Rotation_Count()
