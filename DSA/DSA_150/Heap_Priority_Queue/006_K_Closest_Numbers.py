"""
Problem: K Closest Numbers
URL: https://leetcode.com/problems/find-k-closest-elements/description/

Problem Statement:
Given a sorted integer array arr, two integers k and x, return the k closest integers to x in the array.
The result should also be sorted in ascending order.

Sample Input/Output:
Input: arr = [1,2,3,4,5], k = 4, x = 3
Output: [1,2,3,4]
Explanation: The 4 closest elements to 3 are [1,2,3,4]

Input: arr = [1,2,3,4,5], k = 4, x = -1
Output: [1,2,3,4]
Explanation: The 4 closest elements to -1 are [1,2,3,4]
"""

import heapq
from typing import List

class Solution:
    def Find_Closest_Elements_Brute_Force(self, arr: List[int], k: int, x: int) -> List[int]:
        """
        Brute Force Approach - Sort by distance
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        distances = [(abs(num - x), num) for num in arr]
        distances.sort()
        result = [distances[i][1] for i in range(k)]
        return sorted(result)
    
    def Find_Closest_Elements_Two_Pointers(self, arr: List[int], k: int, x: int) -> List[int]:
        """
        Two Pointers Approach - Expand from center
        Time Complexity: O(n + k)
        Space Complexity: O(1)
        """
        n = len(arr)
        left, right = 0, n - k
        
        while left < right:
            if x - arr[left] > arr[right + k - 1] - x:
                left += 1
            else:
                right -= 1
        
        return arr[left:left + k]
    
    def Find_Closest_Elements_Max_Heap_Optimal(self, arr: List[int], k: int, x: int) -> List[int]:
        """
        Max Heap Approach - Maintain k closest elements
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        """
        max_heap = []
        
        for num in arr:
            distance = abs(num - x)
            if len(max_heap) < k:
                heapq.heappush(max_heap, (-distance, -num))
            else:
                if distance < -max_heap[0][0]:
                    heapq.heappop(max_heap)
                    heapq.heappush(max_heap, (-distance, -num))
        
        result = [-item[1] for item in max_heap]
        return sorted(result)
    
    def Find_Closest_Elements_Binary_Search(self, arr: List[int], k: int, x: int) -> List[int]:
        """
        Binary Search Approach - Find optimal window
        Time Complexity: O(log n + k)
        Space Complexity: O(1)
        """
        left, right = 0, len(arr) - k
        
        while left < right:
            mid = (left + right) // 2
            if x - arr[mid] > arr[mid + k] - x:
                left = mid + 1
            else:
                right = mid
        
        return arr[left:left + k]
    
    def Find_Closest_Elements_Min_Heap(self, arr: List[int], k: int, x: int) -> List[int]:
        """
        Min Heap Approach - All elements then extract k smallest distances
        Time Complexity: O(n log n + k log n)
        Space Complexity: O(n)
        """
        min_heap = [(abs(num - x), num) for num in arr]
        heapq.heapify(min_heap)
        
        result = []
        for _ in range(k):
            distance, num = heapq.heappop(min_heap)
            result.append(num)
        
        return sorted(result)
    
    def Find_Closest_Elements_Priority_Queue(self, arr: List[int], k: int, x: int) -> List[int]:
        """
        Priority Queue with custom comparator
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        """
        priority_queue = []
        
        for num in arr:
            distance = abs(num - x)
            if len(priority_queue) < k:
                heapq.heappush(priority_queue, (-distance, -num, num))
            elif distance < -priority_queue[0][0] or (distance == -priority_queue[0][0] and num < -priority_queue[0][1]):
                heapq.heappop(priority_queue)
                heapq.heappush(priority_queue, (-distance, -num, num))
        
        result = [item[2] for item in priority_queue]
        return sorted(result)

def Test_K_Closest_Numbers():
    solution = Solution()
    
    test_cases = [
        ([1,2,3,4,5], 4, 3, [1,2,3,4]),
        ([1,2,3,4,5], 4, -1, [1,2,3,4]),
        ([1,1,1,10,10,10], 1, 9, [10]),
        ([0,0,1,2,3,3,4,7,7,8], 3, 5, [3,3,4])
    ]
    
    for arr, k, x, expected in test_cases:
        result1 = solution.Find_Closest_Elements_Brute_Force(arr.copy(), k, x)
        result2 = solution.Find_Closest_Elements_Two_Pointers(arr.copy(), k, x)
        result3 = solution.Find_Closest_Elements_Max_Heap_Optimal(arr.copy(), k, x)
        result4 = solution.Find_Closest_Elements_Binary_Search(arr.copy(), k, x)
        result5 = solution.Find_Closest_Elements_Min_Heap(arr.copy(), k, x)
        result6 = solution.Find_Closest_Elements_Priority_Queue(arr.copy(), k, x)
        
        print(f"Array: {arr}, k: {k}, x: {x}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Two Pointers: {result2}")
        print(f"Max Heap Optimal: {result3}")
        print(f"Binary Search: {result4}")
        print(f"Min Heap: {result5}")
        print(f"Priority Queue: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_K_Closest_Numbers()
