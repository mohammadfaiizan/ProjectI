"""
Problem: Single Number
URL: https://leetcode.com/problems/single-number/

Problem Statement:
Given a non-empty array of integers nums, every element appears twice except for one. 
Find that single one.

You must implement a solution with a linear runtime complexity and use only constant extra space.

Sample Input/Output:
Input: nums = [2,2,1]
Output: 1

Input: nums = [4,1,2,1,2]
Output: 4

Input: nums = [1]
Output: 1
"""

from typing import List

class Solution:
    def Single_Number_Brute_Force(self, nums: List[int]) -> int:
        """
        Brute Force Approach - Count occurrences
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        for num in nums:
            count = 0
            for n in nums:
                if n == num:
                    count += 1
            if count == 1:
                return num
        
        return -1
    
    def Single_Number_Hash_Map(self, nums: List[int]) -> int:
        """
        Hash Map Approach - Count with dictionary
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        count_map = {}
        
        for num in nums:
            count_map[num] = count_map.get(num, 0) + 1
        
        for num, count in count_map.items():
            if count == 1:
                return num
        
        return -1
    
    def Single_Number_Set(self, nums: List[int]) -> int:
        """
        Set Approach - Mathematical approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        return 2 * sum(set(nums)) - sum(nums)
    
    def Single_Number_XOR_Optimal(self, nums: List[int]) -> int:
        """
        XOR Approach - Optimal solution using bit manipulation
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        result = 0
        
        for num in nums:
            result ^= num
        
        return result
    
    def Single_Number_XOR_Reduce(self, nums: List[int]) -> int:
        """
        XOR with Reduce - Functional approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        from functools import reduce
        return reduce(lambda x, y: x ^ y, nums)
    
    def Single_Number_Sorting(self, nums: List[int]) -> int:
        """
        Sorting Approach - Sort and compare adjacent
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        if len(nums) == 1:
            return nums[0]
        
        nums.sort()
        
        for i in range(0, len(nums) - 1, 2):
            if nums[i] != nums[i + 1]:
                return nums[i]
        
        return nums[-1]

def Test_Single_Number():
    solution = Solution()
    
    test_cases = [
        ([2,2,1], 1),
        ([4,1,2,1,2], 4),
        ([1], 1),
        ([1,3,1,3,5], 5),
        ([0,1,0,1,99], 99)
    ]
    
    for nums, expected in test_cases:
        result1 = solution.Single_Number_Brute_Force(nums.copy())
        result2 = solution.Single_Number_Hash_Map(nums.copy())
        result3 = solution.Single_Number_Set(nums.copy())
        result4 = solution.Single_Number_XOR_Optimal(nums.copy())
        result5 = solution.Single_Number_XOR_Reduce(nums.copy())
        result6 = solution.Single_Number_Sorting(nums.copy())
        
        print(f"Array: {nums}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Hash Map: {result2}")
        print(f"Set: {result3}")
        print(f"XOR Optimal: {result4}")
        print(f"XOR Reduce: {result5}")
        print(f"Sorting: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Single_Number()

