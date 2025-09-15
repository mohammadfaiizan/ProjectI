"""
Problem: Maximum of all subarrays of size K
URL: https://www.geeksforgeeks.org/problems/maximum-of-all-subarrays-of-size-k3101/1

Problem Statement:
Given an array arr[] of size N and an integer K. Find the maximum for each and every contiguous subarray of size K.

Sample Input/Output:
Input: arr = [1, 2, 3, 1, 4, 5, 2, 3, 6], k = 3
Output: [3, 3, 4, 5, 5, 5, 6]
Explanation: Maximum of [1,2,3] = 3, [2,3,1] = 3, etc.

Input: arr = [8, 5, 10, 7, 9, 4, 15, 12, 90, 13], k = 4
Output: [10, 10, 10, 15, 15, 90, 90]
"""

from typing import List
from collections import deque
import heapq

class Solution:
    def Max_Sliding_Window_Brute_Force(self, arr: List[int], k: int) -> List[int]:
        """
        Brute Force - Find max in each window
        Time Complexity: O(n*k)
        Space Complexity: O(1)
        """
        result = []
        n = len(arr)
        
        for i in range(n - k + 1):
            window_max = max(arr[i:i+k])
            result.append(window_max)
        
        return result
    
    def Max_Sliding_Window_Deque_Optimal(self, arr: List[int], k: int) -> List[int]:
        """
        Deque Approach - Optimal sliding window maximum
        Time Complexity: O(n)
        Space Complexity: O(k)
        """
        dq = deque()
        result = []
        
        for i in range(len(arr)):
            while dq and dq[0] <= i - k:
                dq.popleft()
            
            while dq and arr[dq[-1]] <= arr[i]:
                dq.pop()
            
            dq.append(i)
            
            if i >= k - 1:
                result.append(arr[dq[0]])
        
        return result

def Test_Max_Sliding_Window():
    solution = Solution()
    
    test_cases = [
        ([1, 2, 3, 1, 4, 5, 2, 3, 6], 3, [3, 3, 4, 5, 5, 5, 6]),
        ([8, 5, 10, 7, 9, 4, 15, 12, 90, 13], 4, [10, 10, 10, 15, 15, 90, 90]),
        ([1, 3, -1, -3, 5, 3, 6, 7], 3, [3, 3, 5, 5, 6, 7])
    ]
    
    for arr, k, expected in test_cases:
        result1 = solution.Max_Sliding_Window_Brute_Force(arr.copy(), k)
        result2 = solution.Max_Sliding_Window_Deque_Optimal(arr.copy(), k)
        
        print(f"Array: {arr}, k: {k}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Deque Optimal: {result2}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Max_Sliding_Window()
