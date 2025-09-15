"""
Problem: Sort an Array
URL: https://leetcode.com/problems/sort-an-array/description/

Problem Statement:
Given an array of integers nums, sort the array in ascending order and return it.
You must solve the problem without using the built-in sort function.

Sample Input/Output:
Input: nums = [5,2,3,1]
Output: [1,2,3,5]
Explanation: After sorting the array, the positions of some numbers are not changed

Input: nums = [5,1,1,2,0,0]
Output: [0,0,1,1,2,5]
Explanation: Note that the values of nums are not necessarily unique.
"""

from typing import List
import random

class Solution:
    def Sort_Array_Built_In(self, nums: List[int]) -> List[int]:
        """
        Built-in Sort - Using Python's built-in sort
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        nums.sort()
        return nums
    
    def Sort_Array_Bubble_Sort(self, nums: List[int]) -> List[int]:
        """
        Bubble Sort - Simple sorting algorithm
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        n = len(nums)
        for i in range(n):
            for j in range(0, n - i - 1):
                if nums[j] > nums[j + 1]:
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
        return nums
    
    def Sort_Array_Merge_Sort_Recursive(self, nums: List[int]) -> List[int]:
        """
        Merge Sort - Recursive divide and conquer
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        def Merge_Sort(arr: List[int]) -> List[int]:
            if len(arr) <= 1:
                return arr
            
            mid = len(arr) // 2
            left = Merge_Sort(arr[:mid])
            right = Merge_Sort(arr[mid:])
            
            return Merge(left, right)
        
        def Merge(left: List[int], right: List[int]) -> List[int]:
            result = []
            i = j = 0
            
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
            
            result.extend(left[i:])
            result.extend(right[j:])
            return result
        
        return Merge_Sort(nums)
    
    def Sort_Array_Quick_Sort_Recursive(self, nums: List[int]) -> List[int]:
        """
        Quick Sort - Recursive partitioning
        Time Complexity: O(n log n) average, O(n²) worst
        Space Complexity: O(log n) average
        """
        def Quick_Sort(arr: List[int], low: int, high: int) -> None:
            if low < high:
                pivot_index = Partition(arr, low, high)
                Quick_Sort(arr, low, pivot_index - 1)
                Quick_Sort(arr, pivot_index + 1, high)
        
        def Partition(arr: List[int], low: int, high: int) -> int:
            pivot = arr[high]
            i = low - 1
            
            for j in range(low, high):
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
            
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            return i + 1
        
        Quick_Sort(nums, 0, len(nums) - 1)
        return nums
    
    def Sort_Array_Heap_Sort_Recursive(self, nums: List[int]) -> List[int]:
        """
        Heap Sort - Using max heap property recursively
        Time Complexity: O(n log n)
        Space Complexity: O(log n)
        """
        def Heapify(arr: List[int], n: int, i: int) -> None:
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n and arr[left] > arr[largest]:
                largest = left
            
            if right < n and arr[right] > arr[largest]:
                largest = right
            
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                Heapify(arr, n, largest)
        
        n = len(nums)
        
        for i in range(n // 2 - 1, -1, -1):
            Heapify(nums, n, i)
        
        for i in range(n - 1, 0, -1):
            nums[0], nums[i] = nums[i], nums[0]
            Heapify(nums, i, 0)
        
        return nums
    
    def Sort_Array_Insertion_Sort_Recursive(self, nums: List[int]) -> List[int]:
        """
        Recursive Insertion Sort - Insertion sort using recursion
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        def Insertion_Sort_Recursive(arr: List[int], n: int) -> None:
            if n <= 1:
                return
            
            Insertion_Sort_Recursive(arr, n - 1)
            
            last = arr[n - 1]
            j = n - 2
            
            while j >= 0 and arr[j] > last:
                arr[j + 1] = arr[j]
                j -= 1
            
            arr[j + 1] = last
        
        Insertion_Sort_Recursive(nums, len(nums))
        return nums

def Test_Sort_Array():
    solution = Solution()
    
    test_cases = [
        ([5,2,3,1], [1,2,3,5]),
        ([5,1,1,2,0,0], [0,0,1,1,2,5]),
        ([1], [1]),
        ([2,1], [1,2]),
        ([3,2,1,5,6,4], [1,2,3,4,5,6])
    ]
    
    for nums, expected in test_cases:
        result1 = solution.Sort_Array_Built_In(nums.copy())
        result2 = solution.Sort_Array_Bubble_Sort(nums.copy())
        result3 = solution.Sort_Array_Merge_Sort_Recursive(nums.copy())
        result4 = solution.Sort_Array_Quick_Sort_Recursive(nums.copy())
        result5 = solution.Sort_Array_Heap_Sort_Recursive(nums.copy())
        result6 = solution.Sort_Array_Insertion_Sort_Recursive(nums.copy())
        
        print(f"Array: {nums}")
        print(f"Expected: {expected}")
        print(f"Built-in: {result1}")
        print(f"Bubble Sort: {result2}")
        print(f"Merge Sort Recursive: {result3}")
        print(f"Quick Sort Recursive: {result4}")
        print(f"Heap Sort Recursive: {result5}")
        print(f"Insertion Sort Recursive: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Sort_Array()
