"""
Problem: Minimum size subarray sum
URL: https://leetcode.com/problems/minimum-size-subarray-sum/

Problem Statement:
Given an array of positive integers nums and a positive integer target, return the minimal length of a contiguous subarray 
of which the sum is greater than or equal to target. If there is no such subarray, return 0 instead.

Sample Input/Output:
Input: target = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: The subarray [4,3] has the minimal length under the problem constraint.

Input: target = 4, nums = [1,4,4]
Output: 1
Explanation: The subarray [4] has the minimal length under the problem constraint.
"""

from typing import List

class Solution:
    def Min_Subarray_Length_Brute_Force(self, target: int, nums: List[int]) -> int:
        """
        Brute Force - Check all subarrays
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        n = len(nums)
        min_length = float('inf')
        
        for i in range(n):
            current_sum = 0
            for j in range(i, n):
                current_sum += nums[j]
                if current_sum >= target:
                    min_length = min(min_length, j - i + 1)
                    break
        
        return min_length if min_length != float('inf') else 0
    
    def Min_Subarray_Length_Sliding_Window_Optimal(self, target: int, nums: List[int]) -> int:
        """
        Sliding Window - Optimal approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left = 0
        current_sum = 0
        min_length = float('inf')
        
        for right in range(len(nums)):
            current_sum += nums[right]
            
            while current_sum >= target:
                min_length = min(min_length, right - left + 1)
                current_sum -= nums[left]
                left += 1
        
        return min_length if min_length != float('inf') else 0

def Test_Min_Subarray_Length():
    solution = Solution()
    
    test_cases = [
        (7, [2,3,1,2,4,3], 2),
        (4, [1,4,4], 1),
        (11, [1,1,1,1,1,1,1,1], 0),
        (15, [1,2,3,4,5], 5)
    ]
    
    for target, nums, expected in test_cases:
        result1 = solution.Min_Subarray_Length_Brute_Force(target, nums.copy())
        result2 = solution.Min_Subarray_Length_Sliding_Window_Optimal(target, nums.copy())
        
        print(f"Target: {target}, Nums: {nums}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Sliding Window Optimal: {result2}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Min_Subarray_Length()
