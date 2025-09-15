"""
Problem: Sum of elements between K1st Smallest Element and K2nd Smallest Element
URL: https://www.geeksforgeeks.org/problems/sum-of-elements-between-k1th-and-k2th-smallest-elements3133/

Problem Statement:
Given an array of integers and two numbers k1 and k2. Find the sum of all elements between the K1'th and K2'th smallest elements of the array.
It may be assumed that all elements of the array are distinct.

Sample Input/Output:
Input: arr = [1, 3, 12, 5, 15, 11], k1 = 3, k2 = 6
Output: 23
Explanation: 3rd smallest is 5, 6th smallest is 15, elements between them are 11, 12. Sum = 11 + 12 = 23

Input: arr = [1, 2, 3, 4, 7, 9], k1 = 2, k2 = 5
Output: 10
Explanation: 2nd smallest is 2, 5th smallest is 7, elements between them are 3, 4. Sum = 3 + 4 = 7
"""

import heapq
from typing import List

class Solution:
    def Sum_Between_K1_K2_Brute_Force(self, arr: List[int], k1: int, k2: int) -> int:
        """
        Brute Force Approach - Sort and sum elements
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        arr.sort()
        total_sum = 0
        for i in range(k1, k2 - 1):
            total_sum += arr[i]
        return total_sum
    
    def Sum_Between_K1_K2_Selection_Sort(self, arr: List[int], k1: int, k2: int) -> int:
        """
        Selection Sort Approach - Find k1 and k2 smallest elements
        Time Complexity: O(n * k2)
        Space Complexity: O(1)
        """
        n = len(arr)
        
        for i in range(k2):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        
        total_sum = 0
        for i in range(k1, k2 - 1):
            total_sum += arr[i]
        return total_sum
    
    def Sum_Between_K1_K2_Min_Heap_Optimal(self, arr: List[int], k1: int, k2: int) -> int:
        """
        Min Heap Approach - Extract k2-1 smallest elements
        Time Complexity: O(n + k2 log n)
        Space Complexity: O(n)
        """
        min_heap = arr.copy()
        heapq.heapify(min_heap)
        
        for _ in range(k1):
            heapq.heappop(min_heap)
        
        total_sum = 0
        for _ in range(k2 - k1 - 1):
            total_sum += heapq.heappop(min_heap)
        
        return total_sum
    
    def Sum_Between_K1_K2_Max_Heap(self, arr: List[int], k1: int, k2: int) -> int:
        """
        Max Heap Approach - Find k1 and k2 smallest using max heap
        Time Complexity: O(n log k2)
        Space Complexity: O(k2)
        """
        def Find_Kth_Smallest(arr: List[int], k: int) -> int:
            max_heap = []
            for num in arr:
                if len(max_heap) < k:
                    heapq.heappush(max_heap, -num)
                elif num < -max_heap[0]:
                    heapq.heappop(max_heap)
                    heapq.heappush(max_heap, -num)
            return -max_heap[0]
        
        k1_smallest = Find_Kth_Smallest(arr, k1)
        k2_smallest = Find_Kth_Smallest(arr, k2)
        
        total_sum = 0
        for num in arr:
            if k1_smallest < num < k2_smallest:
                total_sum += num
        
        return total_sum
    
    def Sum_Between_K1_K2_Priority_Queue(self, arr: List[int], k1: int, k2: int) -> int:
        """
        Priority Queue Approach - Extract elements in order
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        priority_queue = arr.copy()
        heapq.heapify(priority_queue)
        
        extracted_elements = []
        for _ in range(k2 - 1):
            extracted_elements.append(heapq.heappop(priority_queue))
        
        total_sum = 0
        for i in range(k1, k2 - 1):
            total_sum += extracted_elements[i]
        
        return total_sum
    
    def Sum_Between_K1_K2_Quick_Select(self, arr: List[int], k1: int, k2: int) -> int:
        """
        Quick Select Approach - Find k1 and k2 smallest elements
        Time Complexity: O(n) average, O(n²) worst
        Space Complexity: O(1)
        """
        def Quick_Select(arr: List[int], left: int, right: int, k: int) -> int:
            if left == right:
                return arr[left]
            
            pivot_index = Partition(arr, left, right)
            
            if k == pivot_index:
                return arr[k]
            elif k < pivot_index:
                return Quick_Select(arr, left, pivot_index - 1, k)
            else:
                return Quick_Select(arr, pivot_index + 1, right, k)
        
        def Partition(arr: List[int], left: int, right: int) -> int:
            pivot = arr[right]
            i = left
            
            for j in range(left, right):
                if arr[j] <= pivot:
                    arr[i], arr[j] = arr[j], arr[i]
                    i += 1
            
            arr[i], arr[right] = arr[right], arr[i]
            return i
        
        arr_copy = arr.copy()
        k1_smallest = Quick_Select(arr_copy, 0, len(arr_copy) - 1, k1 - 1)
        
        arr_copy = arr.copy()
        k2_smallest = Quick_Select(arr_copy, 0, len(arr_copy) - 1, k2 - 1)
        
        total_sum = 0
        for num in arr:
            if k1_smallest < num < k2_smallest:
                total_sum += num
        
        return total_sum

def Test_Sum_Between_K1_K2():
    solution = Solution()
    
    test_cases = [
        ([1, 3, 12, 5, 15, 11], 3, 6, 23),
        ([1, 2, 3, 4, 7, 9], 2, 5, 7),
        ([7, 10, 4, 3, 20, 15], 2, 5, 17),
        ([1, 2, 3, 4, 5], 1, 5, 9)
    ]
    
    for arr, k1, k2, expected in test_cases:
        result1 = solution.Sum_Between_K1_K2_Brute_Force(arr.copy(), k1, k2)
        result2 = solution.Sum_Between_K1_K2_Selection_Sort(arr.copy(), k1, k2)
        result3 = solution.Sum_Between_K1_K2_Min_Heap_Optimal(arr.copy(), k1, k2)
        result4 = solution.Sum_Between_K1_K2_Max_Heap(arr.copy(), k1, k2)
        result5 = solution.Sum_Between_K1_K2_Priority_Queue(arr.copy(), k1, k2)
        result6 = solution.Sum_Between_K1_K2_Quick_Select(arr.copy(), k1, k2)
        
        print(f"Array: {arr}, k1: {k1}, k2: {k2}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Selection Sort: {result2}")
        print(f"Min Heap Optimal: {result3}")
        print(f"Max Heap: {result4}")
        print(f"Priority Queue: {result5}")
        print(f"Quick Select: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Sum_Between_K1_K2()
