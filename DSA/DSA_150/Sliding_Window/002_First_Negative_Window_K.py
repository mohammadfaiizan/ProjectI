"""
Problem: First Negative Number in every subarray size K
URL: https://www.geeksforgeeks.org/problems/first-negative-integer-in-every-window-of-size-k3345/1

Problem Statement:
Given an array and a positive integer k, find the first negative number for each window of size k.
If a window does not contain a negative number, output 0 for that window.

Sample Input/Output:
Input: arr = [12, -1, -7, 8, -15, 30, 16, 28], k = 3
Output: [-1, -1, -7, -15, -15, 0]
Explanation: First negative in each window of size 3

Input: arr = [-8, 2, 3, -6, 10], k = 2
Output: [-8, 0, -6, -6]
Explanation: First negative in each window of size 2
"""

from typing import List
from collections import deque

class Solution:
    def First_Negative_Brute_Force(self, arr: List[int], k: int) -> List[int]:
        """
        Brute Force Approach - Check each window separately
        Time Complexity: O(n*k)
        Space Complexity: O(1)
        """
        n = len(arr)
        result = []
        
        for i in range(n - k + 1):
            first_negative = 0
            for j in range(i, i + k):
                if arr[j] < 0:
                    first_negative = arr[j]
                    break
            result.append(first_negative)
        
        return result
    
    def First_Negative_Nested_Loop(self, arr: List[int], k: int) -> List[int]:
        """
        Nested Loop Approach - Linear search in each window
        Time Complexity: O(n*k)
        Space Complexity: O(1)
        """
        result = []
        n = len(arr)
        
        for i in range(n - k + 1):
            window = arr[i:i+k]
            first_negative = 0
            for num in window:
                if num < 0:
                    first_negative = num
                    break
            result.append(first_negative)
        
        return result
    
    def First_Negative_Sliding_Window_Optimal(self, arr: List[int], k: int) -> List[int]:
        """
        Sliding Window with Deque - Optimal approach
        Time Complexity: O(n)
        Space Complexity: O(k)
        """
        n = len(arr)
        result = []
        dq = deque()
        
        for i in range(n):
            while dq and dq[0] <= i - k:
                dq.popleft()
            
            if arr[i] < 0:
                dq.append(i)
            
            if i >= k - 1:
                if dq:
                    result.append(arr[dq[0]])
                else:
                    result.append(0)
        
        return result
    
    def First_Negative_Two_Pointers(self, arr: List[int], k: int) -> List[int]:
        """
        Two Pointers Approach - Track negative indices
        Time Complexity: O(n)
        Space Complexity: O(k)
        """
        n = len(arr)
        result = []
        negative_indices = []
        left = 0
        
        for right in range(n):
            if arr[right] < 0:
                negative_indices.append(right)
            
            if right - left + 1 == k:
                while negative_indices and negative_indices[0] < left:
                    negative_indices.pop(0)
                
                if negative_indices:
                    result.append(arr[negative_indices[0]])
                else:
                    result.append(0)
                
                left += 1
        
        return result
    
    def First_Negative_Queue_Approach(self, arr: List[int], k: int) -> List[int]:
        """
        Queue Approach - Store negative numbers
        Time Complexity: O(n)
        Space Complexity: O(k)
        """
        from collections import deque
        
        n = len(arr)
        result = []
        queue = deque()
        
        i = 0
        while i < k:
            if arr[i] < 0:
                queue.append(i)
            i += 1
        
        for window_start in range(n - k + 1):
            if queue:
                result.append(arr[queue[0]])
            else:
                result.append(0)
            
            while queue and queue[0] <= window_start:
                queue.popleft()
            
            if window_start + k < n and arr[window_start + k] < 0:
                queue.append(window_start + k)
        
        return result

def Test_First_Negative():
    solution = Solution()
    
    test_cases = [
        ([12, -1, -7, 8, -15, 30, 16, 28], 3, [-1, -1, -7, -15, -15, 0]),
        ([-8, 2, 3, -6, 10], 2, [-8, 0, -6, -6]),
        ([1, 2, 3, 4, 5], 3, [0, 0, 0]),
        ([-1, -2, -3, -4], 2, [-1, -2, -3]),
        ([1, -1, 2, -2, 3], 2, [-1, -1, -2, -2])
    ]
    
    for arr, k, expected in test_cases:
        result1 = solution.First_Negative_Brute_Force(arr.copy(), k)
        result2 = solution.First_Negative_Nested_Loop(arr.copy(), k)
        result3 = solution.First_Negative_Sliding_Window_Optimal(arr.copy(), k)
        result4 = solution.First_Negative_Two_Pointers(arr.copy(), k)
        result5 = solution.First_Negative_Queue_Approach(arr.copy(), k)
        
        print(f"Array: {arr}, k: {k}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Nested Loop: {result2}")
        print(f"Sliding Window Optimal: {result3}")
        print(f"Two Pointers: {result4}")
        print(f"Queue Approach: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_First_Negative()
