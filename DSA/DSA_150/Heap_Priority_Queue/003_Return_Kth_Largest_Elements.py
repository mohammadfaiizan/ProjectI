"""
Problem: Return Kth Largest Elements in an Array
URL: https://leetcode.com/problems/kth-largest-element-in-an-array/

Problem Statement:
Given an integer array nums and an integer k, return the kth largest element in the array.
You can assume that there is always a valid answer.

Sample Input/Output:
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5
Explanation: The 2nd largest element is 5

Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4
Explanation: The 4th largest element is 4
"""

import heapq
from typing import List

class Solution:
    def Find_Kth_Largest_Brute_Force(self, nums: List[int], k: int) -> int:
        """
        Brute Force Approach - Sort array in descending order
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        nums.sort(reverse=True)
        return nums[k-1]
    
    def Find_Kth_Largest_Selection_Sort(self, nums: List[int], k: int) -> int:
        """
        Selection Sort Approach - Find k largest elements
        Time Complexity: O(n*k)
        Space Complexity: O(1)
        """
        n = len(nums)
        for i in range(k):
            max_idx = i
            for j in range(i+1, n):
                if nums[j] > nums[max_idx]:
                    max_idx = j
            nums[i], nums[max_idx] = nums[max_idx], nums[i]
        return nums[k-1]
    
    def Find_Kth_Largest_Min_Heap_Optimal(self, nums: List[int], k: int) -> int:
        """
        Min Heap Approach - Maintain heap of k largest elements
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        """
        min_heap = []
        
        for num in nums:
            heapq.heappush(min_heap, num)
            if len(min_heap) > k:
                heapq.heappop(min_heap)
        
        return min_heap[0]
    
    def Find_Kth_Largest_Max_Heap(self, nums: List[int], k: int) -> int:
        """
        Max Heap Approach - Extract k times from max heap
        Time Complexity: O(n + k log n)
        Space Complexity: O(n)
        """
        max_heap = [-num for num in nums]
        heapq.heapify(max_heap)
        
        result = 0
        for _ in range(k):
            result = -heapq.heappop(max_heap)
        
        return result
    
    def Find_Kth_Largest_Heap_Select(self, nums: List[int], k: int) -> int:
        """
        Heap Select Approach - Optimized heap usage
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        """
        return heapq.nlargest(k, nums)[-1]

def Test_Kth_Largest_Elements():
    solution = Solution()
    
    test_cases = [
        ([3,2,1,5,6,4], 2, 5),
        ([3,2,3,1,2,4,5,5,6], 4, 4),
        ([1], 1, 1),
        ([1,2,3,4,5], 3, 3),
        ([7,10,4,3,20,15], 2, 15)
    ]
    
    for nums, k, expected in test_cases:
        nums_copy1 = nums.copy()
        nums_copy2 = nums.copy()
        nums_copy3 = nums.copy()
        nums_copy4 = nums.copy()
        nums_copy5 = nums.copy()
        
        result1 = solution.Find_Kth_Largest_Brute_Force(nums_copy1, k)
        result2 = solution.Find_Kth_Largest_Selection_Sort(nums_copy2, k)
        result3 = solution.Find_Kth_Largest_Min_Heap_Optimal(nums_copy3, k)
        result4 = solution.Find_Kth_Largest_Max_Heap(nums_copy4, k)
        result5 = solution.Find_Kth_Largest_Heap_Select(nums_copy5, k)
        
        print(f"Array: {nums}, k: {k}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Selection Sort: {result2}")
        print(f"Min Heap Optimal: {result3}")
        print(f"Max Heap: {result4}")
        print(f"Heap Select: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Kth_Largest_Elements()
