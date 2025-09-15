"""
Problem: Kth Smallest Element
URL: https://www.geeksforgeeks.org/problems/kth-smallest-element5635/1

Problem Statement:
Given an array arr[] and an integer k where k is smaller than the size of the array, 
we need to find the kth smallest element in the given array. It is given that all array elements are distinct.

Sample Input/Output:
Input: arr[] = [7, 10, 4, 3, 20, 15], k = 3
Output: 7
Explanation: 3rd smallest element is 7

Input: arr[] = [2, 3, 1, 20, 15], k = 4  
Output: 15
Explanation: 4th smallest element is 15
"""

import heapq
from typing import List

class Solution:
    def Kth_Smallest_Element_Brute_Force(self, arr: List[int], k: int) -> int:
        """
        Brute Force Approach - Sort and return kth element
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        arr.sort()
        return arr[k-1]
    
    def Kth_Smallest_Element_Selection_Sort(self, arr: List[int], k: int) -> int:
        """
        Selection Sort Approach - Sort only k elements
        Time Complexity: O(n*k)
        Space Complexity: O(1)
        """
        n = len(arr)
        for i in range(k):
            min_idx = i
            for j in range(i+1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return arr[k-1]
    
    def Kth_Smallest_Element_Max_Heap(self, arr: List[int], k: int) -> int:
        """
        Max Heap Approach - Maintain heap of size k
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        """
        max_heap = []
        
        for num in arr:
            if len(max_heap) < k:
                heapq.heappush(max_heap, -num)
            elif num < -max_heap[0]:
                heapq.heappop(max_heap)
                heapq.heappush(max_heap, -num)
        
        return -max_heap[0]
    
    def Kth_Smallest_Element_Min_Heap(self, arr: List[int], k: int) -> int:
        """
        Min Heap Approach - Build heap and extract k times
        Time Complexity: O(n + k log n)
        Space Complexity: O(n)
        """
        heapq.heapify(arr)
        
        for _ in range(k-1):
            heapq.heappop(arr)
        
        return arr[0]

def Test_Kth_Smallest_Element():
    solution = Solution()
    
    test_cases = [
        ([7, 10, 4, 3, 20, 15], 3, 7),
        ([2, 3, 1, 20, 15], 4, 15),
        ([1, 2, 3, 4, 5], 1, 1),
        ([5, 4, 3, 2, 1], 5, 5)
    ]
    
    for arr, k, expected in test_cases:
        arr_copy1 = arr.copy()
        arr_copy2 = arr.copy()
        arr_copy3 = arr.copy()
        arr_copy4 = arr.copy()
        
        result1 = solution.Kth_Smallest_Element_Brute_Force(arr_copy1, k)
        result2 = solution.Kth_Smallest_Element_Selection_Sort(arr_copy2, k)
        result3 = solution.Kth_Smallest_Element_Max_Heap(arr_copy3, k)
        result4 = solution.Kth_Smallest_Element_Min_Heap(arr_copy4, k)
        
        print(f"Array: {arr}, k: {k}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Selection Sort: {result2}")
        print(f"Max Heap: {result3}")
        print(f"Min Heap: {result4}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Kth_Smallest_Element()
