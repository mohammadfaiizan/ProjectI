"""
Problem: Kth Smallest Element
URL: https://www.naukri.com/code360/contests/weekly-contest-201/20769517/problems/42563

Problem Statement:
You are given an unsorted array of integers arr of size N and an integer k. 
Your task is to find the k-th smallest element in the array.

Sample Input/Output:
Input: arr = [7, 10, 4, 3, 20, 15], k = 3
Output: 7
Explanation: The 3rd smallest element is 7 (sorted: [3, 4, 7, 10, 15, 20])

Input: arr = [5, 2, 8, 1], k = 2
Output: 2
Explanation: The 2nd smallest element is 2 (sorted: [1, 2, 5, 8])
"""

from typing import List
import heapq
import random

class Solution:
    def Kth_Smallest_Sorting(self, arr: List[int], k: int) -> int:
        """
        Sorting Approach - Sort and return kth element
        Time Complexity: O(n log n)
        Space Complexity: O(1) or O(n) depending on sort implementation
        """
        arr.sort()
        return arr[k - 1]
    
    def Kth_Smallest_Min_Heap(self, arr: List[int], k: int) -> int:
        """
        Min Heap Approach - Build heap and pop k times
        Time Complexity: O(n + k log n)
        Space Complexity: O(n)
        """
        heapq.heapify(arr)
        
        result = 0
        for _ in range(k):
            result = heapq.heappop(arr)
        
        return result
    
    def Kth_Smallest_Max_Heap(self, arr: List[int], k: int) -> int:
        """
        Max Heap Approach - Maintain heap of size k
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        """
        max_heap = []
        
        for num in arr:
            heapq.heappush(max_heap, -num)
            if len(max_heap) > k:
                heapq.heappop(max_heap)
        
        return -max_heap[0]
    
    def Kth_Smallest_QuickSelect_Optimal(self, arr: List[int], k: int) -> int:
        """
        QuickSelect Approach - Optimal solution using partition
        Time Complexity: O(n) average, O(n^2) worst case
        Space Complexity: O(1)
        """
        def Partition(left: int, right: int, pivot_index: int) -> int:
            pivot = arr[pivot_index]
            arr[pivot_index], arr[right] = arr[right], arr[pivot_index]
            
            store_index = left
            for i in range(left, right):
                if arr[i] < pivot:
                    arr[store_index], arr[i] = arr[i], arr[store_index]
                    store_index += 1
            
            arr[right], arr[store_index] = arr[store_index], arr[right]
            return store_index
        
        def Select(left: int, right: int, k_smallest: int) -> int:
            if left == right:
                return arr[left]
            
            pivot_index = random.randint(left, right)
            pivot_index = Partition(left, right, pivot_index)
            
            if k_smallest == pivot_index:
                return arr[k_smallest]
            elif k_smallest < pivot_index:
                return Select(left, pivot_index - 1, k_smallest)
            else:
                return Select(pivot_index + 1, right, k_smallest)
        
        return Select(0, len(arr) - 1, k - 1)
    
    def Kth_Smallest_QuickSelect_Iterative(self, arr: List[int], k: int) -> int:
        """
        QuickSelect Iterative Approach
        Time Complexity: O(n) average, O(n^2) worst case
        Space Complexity: O(1)
        """
        def Partition(left: int, right: int) -> int:
            pivot = arr[right]
            i = left - 1
            
            for j in range(left, right):
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
            
            arr[i + 1], arr[right] = arr[right], arr[i + 1]
            return i + 1
        
        left, right = 0, len(arr) - 1
        k_index = k - 1
        
        while left <= right:
            pivot_index = Partition(left, right)
            
            if pivot_index == k_index:
                return arr[pivot_index]
            elif pivot_index < k_index:
                left = pivot_index + 1
            else:
                right = pivot_index - 1
        
        return -1
    
    def Kth_Smallest_Built_In(self, arr: List[int], k: int) -> int:
        """
        Built-in heapq.nsmallest Approach
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        """
        return heapq.nsmallest(k, arr)[-1]

def Test_Kth_Smallest():
    solution = Solution()
    
    test_cases = [
        ([7, 10, 4, 3, 20, 15], 3, 7),
        ([5, 2, 8, 1], 2, 2),
        ([1], 1, 1),
        ([1, 2, 3, 4, 5], 1, 1),
        ([1, 2, 3, 4, 5], 5, 5),
        ([3, 2, 1, 5, 6, 4], 2, 2),
        ([3, 2, 3, 1, 2, 4, 5, 5, 6], 4, 3)
    ]
    
    for arr, k, expected in test_cases:
        result1 = solution.Kth_Smallest_Sorting(arr.copy(), k)
        result2 = solution.Kth_Smallest_Min_Heap(arr.copy(), k)
        result3 = solution.Kth_Smallest_Max_Heap(arr.copy(), k)
        result4 = solution.Kth_Smallest_QuickSelect_Optimal(arr.copy(), k)
        result5 = solution.Kth_Smallest_QuickSelect_Iterative(arr.copy(), k)
        result6 = solution.Kth_Smallest_Built_In(arr.copy(), k)
        
        print(f"Array: {arr}, k: {k}")
        print(f"Expected: {expected}")
        print(f"Sorting: {result1}")
        print(f"Min Heap: {result2}")
        print(f"Max Heap: {result3}")
        print(f"QuickSelect Optimal: {result4}")
        print(f"QuickSelect Iterative: {result5}")
        print(f"Built-in: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Kth_Smallest()

