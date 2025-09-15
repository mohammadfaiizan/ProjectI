"""
Problem: Order Agnostic Binary Search
URL: No link found

Problem Statement:
Given a sorted array that can be sorted in either ascending or descending order, and a target value,
find the index of the target. If not found, return -1.

Sample Input/Output:
Input: arr = [1, 2, 3, 4, 5, 6, 7], target = 5
Output: 4
Explanation: Array is sorted in ascending order, target 5 is at index 4

Input: arr = [7, 6, 5, 4, 3, 2, 1], target = 5
Output: 2
Explanation: Array is sorted in descending order, target 5 is at index 2
"""

from typing import List

class Solution:
    def Order_Agnostic_Binary_Search_Linear(self, arr: List[int], target: int) -> int:
        """
        Linear Search Approach - Sequential search regardless of order
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1
    
    def Order_Agnostic_Binary_Search_Two_Phase(self, arr: List[int], target: int) -> int:
        """
        Two Phase Approach - First determine order, then binary search
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if not arr:
            return -1
        
        is_ascending = arr[0] <= arr[-1]
        
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] == target:
                return mid
            
            if is_ascending:
                if arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                if arr[mid] < target:
                    right = mid - 1
                else:
                    left = mid + 1
        
        return -1
    
    def Order_Agnostic_Binary_Search_Optimal(self, arr: List[int], target: int) -> int:
        """
        Order Agnostic Binary Search - Optimal solution
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if not arr:
            return -1
        
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] == target:
                return mid
            
            if arr[left] <= arr[right]:
                if arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                if arr[mid] > target:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1
    
    def Order_Agnostic_Binary_Search_Single_Check(self, arr: List[int], target: int) -> int:
        """
        Single Check Approach - Check order once at beginning
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if not arr:
            return -1
        
        n = len(arr)
        if n == 1:
            return 0 if arr[0] == target else -1
        
        is_ascending = arr[0] < arr[n - 1]
        left, right = 0, n - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if arr[mid] == target:
                return mid
            
            if (is_ascending and arr[mid] < target) or (not is_ascending and arr[mid] > target):
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    
    def Order_Agnostic_Binary_Search_Recursive(self, arr: List[int], target: int) -> int:
        """
        Recursive Approach - Order agnostic recursive binary search
        Time Complexity: O(log n)
        Space Complexity: O(log n)
        """
        def Binary_Search_Helper(left: int, right: int, is_ascending: bool) -> int:
            if left > right:
                return -1
            
            mid = left + (right - left) // 2
            
            if arr[mid] == target:
                return mid
            
            if is_ascending:
                if arr[mid] < target:
                    return Binary_Search_Helper(mid + 1, right, is_ascending)
                else:
                    return Binary_Search_Helper(left, mid - 1, is_ascending)
            else:
                if arr[mid] > target:
                    return Binary_Search_Helper(mid + 1, right, is_ascending)
                else:
                    return Binary_Search_Helper(left, mid - 1, is_ascending)
        
        if not arr:
            return -1
        
        is_ascending = len(arr) == 1 or arr[0] <= arr[-1]
        return Binary_Search_Helper(0, len(arr) - 1, is_ascending)
    
    def Order_Agnostic_Binary_Search_Generic(self, arr: List[int], target: int) -> int:
        """
        Generic Template - Works for any monotonic array
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if not arr:
            return -1
        
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] == target:
                return mid
            
            if len(arr) == 1:
                break
            
            ascending = arr[0] <= arr[-1]
            
            if ascending:
                if arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                if arr[mid] > target:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1

def Test_Order_Agnostic_Binary_Search():
    solution = Solution()
    
    test_cases = [
        ([1, 2, 3, 4, 5, 6, 7], 5, 4),
        ([7, 6, 5, 4, 3, 2, 1], 5, 2),
        ([1, 2, 3, 4, 5], 1, 0),
        ([5, 4, 3, 2, 1], 1, 4),
        ([1, 3, 5, 7, 9], 8, -1),
        ([9, 7, 5, 3, 1], 8, -1),
        ([42], 42, 0)
    ]
    
    for arr, target, expected in test_cases:
        result1 = solution.Order_Agnostic_Binary_Search_Linear(arr.copy(), target)
        result2 = solution.Order_Agnostic_Binary_Search_Two_Phase(arr.copy(), target)
        result3 = solution.Order_Agnostic_Binary_Search_Optimal(arr.copy(), target)
        result4 = solution.Order_Agnostic_Binary_Search_Single_Check(arr.copy(), target)
        result5 = solution.Order_Agnostic_Binary_Search_Recursive(arr.copy(), target)
        result6 = solution.Order_Agnostic_Binary_Search_Generic(arr.copy(), target)
        
        print(f"Array: {arr}, Target: {target}")
        print(f"Expected: {expected}")
        print(f"Linear: {result1}")
        print(f"Two Phase: {result2}")
        print(f"Optimal: {result3}")
        print(f"Single Check: {result4}")
        print(f"Recursive: {result5}")
        print(f"Generic: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Order_Agnostic_Binary_Search()
