"""
Problem: First Negative Integer in Every Window of Size K
URL: https://practice.geeksforgeeks.org/problems/first-negative-integer-in-every-window-of-size-k3345/1

Problem Statement:
Given an array and a positive integer k, find the first negative integer for each and every contiguous subarray of size k.
If a window does not contain a negative integer, then return 0 for that window.

Sample Input/Output:
Input: arr[] = {-8,2,3,-6,10}, k = 2
Output: [-8,0,-6,-6]
"""

from collections import deque


class Solution:
    def First_Negative_Window_K_Deque(self, arr, k):
        """
        Find first negative in each window using deque-based sliding window.
        Time Complexity: O(n)
        Space Complexity: O(k)
        """
        n = len(arr)
        result = []
        dq = deque()
        
        for i in range(k):
            if arr[i] < 0:
                dq.append(i)
        
        if dq:
            result.append(arr[dq[0]])
        else:
            result.append(0)
        
        for i in range(k, n):
            if dq and dq[0] == i - k:
                dq.popleft()
            
            if arr[i] < 0:
                dq.append(i)
            
            if dq:
                result.append(arr[dq[0]])
            else:
                result.append(0)
        
        return result
    
    def First_Negative_Window_K_Brute_Force(self, arr, k):
        """
        Find first negative in each window using brute force.
        Time Complexity: O(n*k)
        Space Complexity: O(1)
        """
        n = len(arr)
        result = []
        
        for i in range(n - k + 1):
            firstNeg = 0
            for j in range(i, i + k):
                if arr[j] < 0:
                    firstNeg = arr[j]
                    break
            result.append(firstNeg)
        
        return result


def Test_First_Negative_Window_K():
    solution = Solution()
    
    arr1 = [-8, 2, 3, -6, 10]
    k1 = 2
    result1 = solution.First_Negative_Window_K_Deque(arr1, k1)
    print(f"Test 1 - Deque: {result1}")
    
    arr2 = [12, -1, -7, 8, -15, 30, 16, 28]
    k2 = 3
    result2 = solution.First_Negative_Window_K_Deque(arr2, k2)
    print(f"Test 2 - Deque: {result2}")
    
    arr3 = [-1, -2, -3, -4, -5]
    k3 = 2
    result3 = solution.First_Negative_Window_K_Deque(arr3, k3)
    print(f"Test 3 - Deque: {result3}")


if __name__ == "__main__":
    Test_First_Negative_Window_K()
