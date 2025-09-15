"""
Problem: Sort a Nearly/K Sorted Array
URL: https://www.geeksforgeeks.org/problems/nearly-sorted-1587115620/1

Problem Statement:
Given an array of n elements, where each element is at most k away from its target position 
in a sorted array. The task is to sort the array efficiently.

Sample Input/Output:
Input: arr = [3, 2, 1, 5, 4, 7, 6, 5], k = 3
Output: [1, 2, 3, 4, 5, 5, 6, 7]
Explanation: Each element is at most 3 positions away from its sorted position

Input: arr = [2, 6, 3, 12, 56, 8], k = 3
Output: [2, 3, 6, 8, 12, 56]
Explanation: Array sorted with k=3 constraint
"""

import heapq
from typing import List

class Solution:
    def Sort_K_Sorted_Brute_Force(self, arr: List[int], k: int) -> List[int]:
        """
        Brute Force Approach - Use built-in sort
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        arr.sort()
        return arr
    
    def Sort_K_Sorted_Insertion_Sort(self, arr: List[int], k: int) -> List[int]:
        """
        Insertion Sort Approach - Optimized for k-sorted arrays
        Time Complexity: O(n*k)
        Space Complexity: O(1)
        """
        n = len(arr)
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            while j >= 0 and j >= i - k - 1 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr
    
    def Sort_K_Sorted_Min_Heap_Optimal(self, arr: List[int], k: int) -> List[int]:
        """
        Min Heap Approach - Maintain heap of size k+1
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        """
        if not arr:
            return arr
        
        min_heap = []
        result = []
        n = len(arr)
        
        for i in range(min(n, k + 1)):
            heapq.heappush(min_heap, arr[i])
        
        for i in range(k + 1, n):
            result.append(heapq.heappop(min_heap))
            heapq.heappush(min_heap, arr[i])
        
        while min_heap:
            result.append(heapq.heappop(min_heap))
        
        return result
    
    def Sort_K_Sorted_In_Place_Heap(self, arr: List[int], k: int) -> List[int]:
        """
        In-place Min Heap Approach
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        """
        if not arr or k == 0:
            return arr
        
        min_heap = []
        result_index = 0
        
        for i in range(len(arr)):
            heapq.heappush(min_heap, arr[i])
            
            if len(min_heap) > k:
                arr[result_index] = heapq.heappop(min_heap)
                result_index += 1
        
        while min_heap:
            arr[result_index] = heapq.heappop(min_heap)
            result_index += 1
        
        return arr
    
    def Sort_K_Sorted_Priority_Queue(self, arr: List[int], k: int) -> List[int]:
        """
        Priority Queue Approach using heap operations
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        """
        n = len(arr)
        if n <= 1:
            return arr
        
        priority_queue = []
        result = []
        
        for i in range(min(k + 1, n)):
            heapq.heappush(priority_queue, arr[i])
        
        next_index = k + 1
        
        while priority_queue:
            result.append(heapq.heappop(priority_queue))
            
            if next_index < n:
                heapq.heappush(priority_queue, arr[next_index])
                next_index += 1
        
        return result

def Test_Sort_K_Sorted_Array():
    solution = Solution()
    
    test_cases = [
        ([3, 2, 1, 5, 4, 7, 6, 5], 3, [1, 2, 3, 4, 5, 5, 6, 7]),
        ([2, 6, 3, 12, 56, 8], 3, [2, 3, 6, 8, 12, 56]),
        ([10, 9, 8, 7, 4, 70, 60, 50], 4, [4, 7, 8, 9, 10, 50, 60, 70]),
        ([1, 4, 5, 2, 3, 7, 8, 6, 10, 9], 2, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ]
    
    for arr, k, expected in test_cases:
        arr_copy1 = arr.copy()
        arr_copy2 = arr.copy()
        arr_copy3 = arr.copy()
        arr_copy4 = arr.copy()
        arr_copy5 = arr.copy()
        
        result1 = solution.Sort_K_Sorted_Brute_Force(arr_copy1, k)
        result2 = solution.Sort_K_Sorted_Insertion_Sort(arr_copy2, k)
        result3 = solution.Sort_K_Sorted_Min_Heap_Optimal(arr_copy3, k)
        result4 = solution.Sort_K_Sorted_In_Place_Heap(arr_copy4, k)
        result5 = solution.Sort_K_Sorted_Priority_Queue(arr_copy5, k)
        
        print(f"Array: {arr}, k: {k}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Insertion Sort: {result2}")
        print(f"Min Heap Optimal: {result3}")
        print(f"In-place Heap: {result4}")
        print(f"Priority Queue: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Sort_K_Sorted_Array()
