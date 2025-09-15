"""
Problem: Largest Subarray of sum K
URL: https://www.geeksforgeeks.org/problems/longest-sub-array-with-sum-k0809/1

Problem Statement:
Given an array containing N positive integers and an integer K. Your task is to find the length of the longest Sub-Array with sum of the elements equal to the given value K.

Sample Input/Output:
Input: arr = [10, 5, 2, 7, 1, 9], k = 15
Output: 4
Explanation: The sub-array is {5, 2, 7, 1}

Input: arr = [4, 1, 1, 1, 2, 3, 5], k = 5
Output: 4
Explanation: The sub-array is {1, 1, 1, 2}
"""

from typing import List

class Solution:
    def Longest_Subarray_Sum_K_Brute_Force(self, arr: List[int], k: int) -> int:
        """
        Brute Force - Check all subarrays
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        n = len(arr)
        max_length = 0
        
        for i in range(n):
            current_sum = 0
            for j in range(i, n):
                current_sum += arr[j]
                if current_sum == k:
                    max_length = max(max_length, j - i + 1)
        
        return max_length
    
    def Longest_Subarray_Sum_K_Sliding_Window_Optimal(self, arr: List[int], k: int) -> int:
        """
        Sliding Window - For positive integers only
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left = 0
        current_sum = 0
        max_length = 0
        
        for right in range(len(arr)):
            current_sum += arr[right]
            
            while current_sum > k and left <= right:
                current_sum -= arr[left]
                left += 1
            
            if current_sum == k:
                max_length = max(max_length, right - left + 1)
        
        return max_length

def Test_Longest_Subarray_Sum_K():
    solution = Solution()
    
    test_cases = [
        ([10, 5, 2, 7, 1, 9], 15, 4),
        ([4, 1, 1, 1, 2, 3, 5], 5, 4),
        ([1, 2, 3], 3, 1),
        ([1, 1, 1, 1], 2, 2)
    ]
    
    for arr, k, expected in test_cases:
        result1 = solution.Longest_Subarray_Sum_K_Brute_Force(arr.copy(), k)
        result2 = solution.Longest_Subarray_Sum_K_Sliding_Window_Optimal(arr.copy(), k)
        
        print(f"Array: {arr}, k: {k}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Sliding Window Optimal: {result2}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Longest_Subarray_Sum_K()
