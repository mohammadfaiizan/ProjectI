"""
Problem: Find an Element in a Nearly Sorted Array
URL: https://www.geeksforgeeks.org/dsa/search-almost-sorted-array/

Problem Statement:
Given a sorted array where every element is at most k positions away from its target position in a sorted array, search for a given element.

Sample Input/Output:
Input: arr = [10, 3, 40, 20, 50, 80, 70], target = 40, k = 1
Output: 2
Explanation: Element 40 is at index 2

Input: arr = [2, 1, 3, 5, 4, 7, 6, 8, 9], target = 4, k = 1  
Output: 4
Explanation: Element 4 is at index 4
"""

from typing import List

class Solution:
    def Search_Nearly_Sorted_Linear(self, arr: List[int], target: int, k: int) -> int:
        """
        Linear Search Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1
    
    def Search_Nearly_Sorted_Binary_Search_Optimal(self, arr: List[int], target: int, k: int) -> int:
        """
        Modified Binary Search for Nearly Sorted Array
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] == target:
                return mid
            
            if mid - 1 >= left and arr[mid - 1] == target:
                return mid - 1
            
            if mid + 1 <= right and arr[mid + 1] == target:
                return mid + 1
            
            if arr[mid] < target:
                left = mid + 2
            else:
                right = mid - 2
        
        return -1

def Test_Search_Nearly_Sorted():
    solution = Solution()
    
    test_cases = [
        ([10, 3, 40, 20, 50, 80, 70], 40, 1, 2),
        ([2, 1, 3, 5, 4, 7, 6, 8, 9], 4, 1, 4),
        ([3, 2, 10, 4, 40], 4, 1, 3),
        ([1, 2, 3, 4, 5], 3, 0, 2)
    ]
    
    for arr, target, k, expected in test_cases:
        result1 = solution.Search_Nearly_Sorted_Linear(arr.copy(), target, k)
        result2 = solution.Search_Nearly_Sorted_Binary_Search_Optimal(arr.copy(), target, k)
        
        print(f"Array: {arr}, Target: {target}, k: {k}")
        print(f"Expected: {expected}")
        print(f"Linear: {result1}")
        print(f"Binary Search Optimal: {result2}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Search_Nearly_Sorted()
