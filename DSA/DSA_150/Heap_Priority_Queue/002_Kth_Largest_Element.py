"""
Problem: Kth Largest Element
URL: https://leetcode.com/problems/kth-largest-element-in-an-array/

Problem Statement:
Given an integer array nums and an integer k, return the kth largest element in the array.
Note that it is the kth largest element in the sorted order, not the kth distinct element.

Sample Input/Output:
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5
Explanation: The 2nd largest element is 5

Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4
Explanation: The 4th largest element is 4
"""

import heapq
import random
from typing import List

class Solution:
    def Find_Kth_Largest_Brute_Force(self, nums: List[int], k: int) -> int:
        """
        Brute Force Approach - Sort and return kth largest
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        nums.sort(reverse=True)
        return nums[k-1]
    
    def Find_Kth_Largest_Bubble_Sort(self, nums: List[int], k: int) -> int:
        """
        Bubble Sort Approach - Sort only k elements
        Time Complexity: O(n*k)
        Space Complexity: O(1)
        """
        n = len(nums)
        for i in range(k):
            for j in range(0, n-i-1):
                if nums[j] < nums[j+1]:
                    nums[j], nums[j+1] = nums[j+1], nums[j]
        return nums[k-1]
    
    def Find_Kth_Largest_Min_Heap(self, nums: List[int], k: int) -> int:
        """
        Min Heap Approach - Maintain heap of size k
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        """
        min_heap = []
        
        for num in nums:
            if len(min_heap) < k:
                heapq.heappush(min_heap, num)
            elif num > min_heap[0]:
                heapq.heappop(min_heap)
                heapq.heappush(min_heap, num)
        
        return min_heap[0]
    
    def Find_Kth_Largest_Max_Heap(self, nums: List[int], k: int) -> int:
        """
        Max Heap Approach - Build heap and extract k times
        Time Complexity: O(n + k log n)
        Space Complexity: O(n)
        """
        max_heap = [-num for num in nums]
        heapq.heapify(max_heap)
        
        for _ in range(k-1):
            heapq.heappop(max_heap)
        
        return -max_heap[0]
    
    def Find_Kth_Largest_Quick_Select(self, nums: List[int], k: int) -> int:
        """
        Quick Select Approach - Partition based algorithm
        Time Complexity: O(n) average, O(n²) worst
        Space Complexity: O(1)
        """
        def Quick_Select(left: int, right: int, k_smallest: int) -> int:
            if left == right:
                return nums[left]
            
            pivot_index = random.randint(left, right)
            pivot_index = Partition(left, right, pivot_index)
            
            if k_smallest == pivot_index:
                return nums[k_smallest]
            elif k_smallest < pivot_index:
                return Quick_Select(left, pivot_index - 1, k_smallest)
            else:
                return Quick_Select(pivot_index + 1, right, k_smallest)
        
        def Partition(left: int, right: int, pivot_index: int) -> int:
            pivot = nums[pivot_index]
            nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
            
            store_index = left
            for i in range(left, right):
                if nums[i] > pivot:
                    nums[store_index], nums[i] = nums[i], nums[store_index]
                    store_index += 1
            
            nums[right], nums[store_index] = nums[store_index], nums[right]
            return store_index
        
        return Quick_Select(0, len(nums) - 1, k - 1)

def Test_Kth_Largest_Element():
    solution = Solution()
    
    test_cases = [
        ([3,2,1,5,6,4], 2, 5),
        ([3,2,3,1,2,4,5,5,6], 4, 4),
        ([1], 1, 1),
        ([1,2,3,4,5], 3, 3)
    ]
    
    for nums, k, expected in test_cases:
        nums_copy1 = nums.copy()
        nums_copy2 = nums.copy()
        nums_copy3 = nums.copy()
        nums_copy4 = nums.copy()
        nums_copy5 = nums.copy()
        
        result1 = solution.Find_Kth_Largest_Brute_Force(nums_copy1, k)
        result2 = solution.Find_Kth_Largest_Bubble_Sort(nums_copy2, k)
        result3 = solution.Find_Kth_Largest_Min_Heap(nums_copy3, k)
        result4 = solution.Find_Kth_Largest_Max_Heap(nums_copy4, k)
        result5 = solution.Find_Kth_Largest_Quick_Select(nums_copy5, k)
        
        print(f"Array: {nums}, k: {k}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Bubble Sort: {result2}")
        print(f"Min Heap: {result3}")
        print(f"Max Heap: {result4}")
        print(f"Quick Select: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Kth_Largest_Element()
