"""
Problem: Maximum Sum Subarray of Size K
URL: https://www.geeksforgeeks.org/problems/max-sum-subarray-of-size-k5313/1

Problem Statement:
Given an array of integers and a number k, find the maximum sum of a subarray of size k.

Sample Input/Output:
Input: arr = [2, 1, 5, 1, 3, 2], k = 3
Output: 9
Explanation: Subarray [5, 1, 3] has maximum sum 9

Input: arr = [2, 3, 4, 1, 5], k = 2
Output: 7
Explanation: Subarray [3, 4] has maximum sum 7
"""

from typing import List

class Solution:
    def Max_Sum_Subarray_Brute_Force(self, arr: List[int], k: int) -> int:
        """
        Brute Force Approach - Check all subarrays of size k
        Time Complexity: O(n*k)
        Space Complexity: O(1)
        """
        n = len(arr)
        if n < k:
            return 0
        
        max_sum = float('-inf')
        
        for i in range(n - k + 1):
            current_sum = 0
            for j in range(i, i + k):
                current_sum += arr[j]
            max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    def Max_Sum_Subarray_Nested_Loop(self, arr: List[int], k: int) -> int:
        """
        Nested Loop Approach - Calculate sum for each window
        Time Complexity: O(n*k)
        Space Complexity: O(1)
        """
        n = len(arr)
        max_sum = float('-inf')
        
        for i in range(n - k + 1):
            window_sum = sum(arr[i:i+k])
            max_sum = max(max_sum, window_sum)
        
        return max_sum
    
    def Max_Sum_Subarray_Sliding_Window_Optimal(self, arr: List[int], k: int) -> int:
        """
        Sliding Window Approach - Optimal solution
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(arr)
        if n < k:
            return 0
        
        window_sum = sum(arr[:k])
        max_sum = window_sum
        
        for i in range(k, n):
            window_sum = window_sum - arr[i - k] + arr[i]
            max_sum = max(max_sum, window_sum)
        
        return max_sum
    
    def Max_Sum_Subarray_Two_Pointers(self, arr: List[int], k: int) -> int:
        """
        Two Pointers Approach - Alternative sliding window
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(arr)
        if n < k:
            return 0
        
        left, right = 0, 0
        current_sum = 0
        max_sum = float('-inf')
        
        while right < n:
            current_sum += arr[right]
            
            if right - left + 1 == k:
                max_sum = max(max_sum, current_sum)
                current_sum -= arr[left]
                left += 1
            
            right += 1
        
        return max_sum
    
    def Max_Sum_Subarray_Deque_Approach(self, arr: List[int], k: int) -> int:
        """
        Deque Approach - Using collections.deque
        Time Complexity: O(n)
        Space Complexity: O(k)
        """
        from collections import deque
        
        n = len(arr)
        if n < k:
            return 0
        
        window = deque()
        current_sum = 0
        max_sum = float('-inf')
        
        for i in range(n):
            window.append(arr[i])
            current_sum += arr[i]
            
            if len(window) == k:
                max_sum = max(max_sum, current_sum)
                removed = window.popleft()
                current_sum -= removed
        
        return max_sum

def Test_Max_Sum_Subarray():
    solution = Solution()
    
    test_cases = [
        ([2, 1, 5, 1, 3, 2], 3, 9),
        ([2, 3, 4, 1, 5], 2, 7),
        ([1, 4, 2, 9, 2], 3, 15),
        ([1, 2, 3, 4, 5, 6], 3, 15),
        ([100, 200, 300, 400], 2, 700)
    ]
    
    for arr, k, expected in test_cases:
        result1 = solution.Max_Sum_Subarray_Brute_Force(arr.copy(), k)
        result2 = solution.Max_Sum_Subarray_Nested_Loop(arr.copy(), k)
        result3 = solution.Max_Sum_Subarray_Sliding_Window_Optimal(arr.copy(), k)
        result4 = solution.Max_Sum_Subarray_Two_Pointers(arr.copy(), k)
        result5 = solution.Max_Sum_Subarray_Deque_Approach(arr.copy(), k)
        
        print(f"Array: {arr}, k: {k}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Nested Loop: {result2}")
        print(f"Sliding Window Optimal: {result3}")
        print(f"Two Pointers: {result4}")
        print(f"Deque Approach: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Max_Sum_Subarray()
