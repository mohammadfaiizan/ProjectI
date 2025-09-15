"""
Problem: Count of Elements in a Sorted Array
URL: https://takeuforward.org/data-structure/count-occurrences-in-sorted-array/

Problem Statement:
Given a sorted array and a target value, count the number of occurrences of the target in the array.

Sample Input/Output:
Input: arr = [2, 2, 3, 3, 3, 3, 4], target = 3
Output: 4
Explanation: Target 3 appears 4 times in the array

Input: arr = [1, 1, 2, 2, 2, 2, 3], target = 4
Output: 0
Explanation: Target 4 does not appear in the array
"""

from typing import List

class Solution:
    def Count_Elements_Linear_Search(self, arr: List[int], target: int) -> int:
        """
        Linear Search Approach - Count while iterating
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        count = 0
        for num in arr:
            if num == target:
                count += 1
        return count
    
    def Count_Elements_Built_In_Count(self, arr: List[int], target: int) -> int:
        """
        Built-in Count Approach - Using list.count()
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        return arr.count(target)
    
    def Count_Elements_First_Last_Position(self, arr: List[int], target: int) -> int:
        """
        First Last Position Approach - Find range then calculate count
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        def Find_First():
            left, right = 0, len(arr) - 1
            first = -1
            
            while left <= right:
                mid = left + (right - left) // 2
                
                if arr[mid] == target:
                    first = mid
                    right = mid - 1
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return first
        
        def Find_Last():
            left, right = 0, len(arr) - 1
            last = -1
            
            while left <= right:
                mid = left + (right - left) // 2
                
                if arr[mid] == target:
                    last = mid
                    left = mid + 1
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return last
        
        first = Find_First()
        if first == -1:
            return 0
        
        last = Find_Last()
        return last - first + 1
    
    def Count_Elements_Binary_Search_Optimal(self, arr: List[int], target: int) -> int:
        """
        Binary Search Optimal Approach - Lower and upper bound
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        def Lower_Bound(target: int) -> int:
            left, right = 0, len(arr)
            
            while left < right:
                mid = (left + right) // 2
                if arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid
            
            return left
        
        def Upper_Bound(target: int) -> int:
            left, right = 0, len(arr)
            
            while left < right:
                mid = (left + right) // 2
                if arr[mid] <= target:
                    left = mid + 1
                else:
                    right = mid
            
            return left
        
        lower = Lower_Bound(target)
        upper = Upper_Bound(target)
        
        return upper - lower
    
    def Count_Elements_Bisect_Module(self, arr: List[int], target: int) -> int:
        """
        Bisect Module Approach - Using Python's bisect
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        import bisect
        
        left_index = bisect.bisect_left(arr, target)
        right_index = bisect.bisect_right(arr, target)
        
        return right_index - left_index
    
    def Count_Elements_Single_Binary_Search(self, arr: List[int], target: int) -> int:
        """
        Single Binary Search Approach - Find any occurrence then expand
        Time Complexity: O(log n + k) where k is count
        Space Complexity: O(1)
        """
        def Find_Any_Occurrence() -> int:
            left, right = 0, len(arr) - 1
            
            while left <= right:
                mid = left + (right - left) // 2
                
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return -1
        
        index = Find_Any_Occurrence()
        if index == -1:
            return 0
        
        count = 1
        
        left = index - 1
        while left >= 0 and arr[left] == target:
            count += 1
            left -= 1
        
        right = index + 1
        while right < len(arr) and arr[right] == target:
            count += 1
            right += 1
        
        return count

def Test_Count_Elements():
    solution = Solution()
    
    test_cases = [
        ([2, 2, 3, 3, 3, 3, 4], 3, 4),
        ([1, 1, 2, 2, 2, 2, 3], 4, 0),
        ([5, 5, 5, 5, 5], 5, 5),
        ([1, 2, 3, 4, 5], 3, 1),
        ([], 1, 0),
        ([1], 1, 1),
        ([1], 2, 0)
    ]
    
    for arr, target, expected in test_cases:
        result1 = solution.Count_Elements_Linear_Search(arr.copy(), target)
        result2 = solution.Count_Elements_Built_In_Count(arr.copy(), target)
        result3 = solution.Count_Elements_First_Last_Position(arr.copy(), target)
        result4 = solution.Count_Elements_Binary_Search_Optimal(arr.copy(), target)
        result5 = solution.Count_Elements_Bisect_Module(arr.copy(), target)
        result6 = solution.Count_Elements_Single_Binary_Search(arr.copy(), target)
        
        print(f"Array: {arr}, Target: {target}")
        print(f"Expected: {expected}")
        print(f"Linear Search: {result1}")
        print(f"Built-in Count: {result2}")
        print(f"First Last Position: {result3}")
        print(f"Binary Search Optimal: {result4}")
        print(f"Bisect Module: {result5}")
        print(f"Single Binary Search: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Count_Elements()
