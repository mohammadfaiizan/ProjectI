"""
Problem: Two Sum
URL: https://leetcode.com/problems/two-sum/

Problem Statement:
Given an array of integers nums and an integer target, return indices of the two numbers 
such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the 
same element twice.

You can return the answer in any order.

Sample Input/Output:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: nums[0] + nums[1] = 2 + 7 = 9

Input: nums = [3,2,4], target = 6
Output: [1,2]

Input: nums = [3,3], target = 6
Output: [0,1]
"""

from typing import List

class Solution:
    def Two_Sum_Brute_Force(self, nums: List[int], target: int) -> List[int]:
        """
        Brute Force Approach - Check all pairs
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        n = len(nums)
        
        for i in range(n):
            for j in range(i + 1, n):
                if nums[i] + nums[j] == target:
                    return [i, j]
        
        return []
    
    def Two_Sum_Two_Pointer(self, nums: List[int], target: int) -> List[int]:
        """
        Two Pointer Approach - Sort and use two pointers
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        indexed_nums = [(num, i) for i, num in enumerate(nums)]
        indexed_nums.sort()
        
        left, right = 0, len(nums) - 1
        
        while left < right:
            current_sum = indexed_nums[left][0] + indexed_nums[right][0]
            
            if current_sum == target:
                return [indexed_nums[left][1], indexed_nums[right][1]]
            elif current_sum < target:
                left += 1
            else:
                right -= 1
        
        return []
    
    def Two_Sum_Hash_Map_Optimal(self, nums: List[int], target: int) -> List[int]:
        """
        Hash Map Approach - One pass with hash map
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        num_map = {}
        
        for i, num in enumerate(nums):
            complement = target - num
            
            if complement in num_map:
                return [num_map[complement], i]
            
            num_map[num] = i
        
        return []
    
    def Two_Sum_Hash_Map_Two_Pass(self, nums: List[int], target: int) -> List[int]:
        """
        Hash Map Two Pass Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        num_map = {}
        
        for i, num in enumerate(nums):
            num_map[num] = i
        
        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_map and num_map[complement] != i:
                return [i, num_map[complement]]
        
        return []
    
    def Two_Sum_Enumerate(self, nums: List[int], target: int) -> List[int]:
        """
        Hash Map with Enumerate
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        seen = {}
        
        for idx, val in enumerate(nums):
            diff = target - val
            if diff in seen:
                return [seen[diff], idx]
            seen[val] = idx
        
        return []

def Test_Two_Sum():
    solution = Solution()
    
    test_cases = [
        ([2,7,11,15], 9, [0,1]),
        ([3,2,4], 6, [1,2]),
        ([3,3], 6, [0,1]),
        ([1,2,3,4,5], 9, [3,4]),
        ([0,4,3,0], 0, [0,3]),
        ([-1,-2,-3,-4,-5], -8, [2,4])
    ]
    
    for nums, target, expected in test_cases:
        result1 = solution.Two_Sum_Brute_Force(nums.copy(), target)
        result2 = solution.Two_Sum_Two_Pointer(nums.copy(), target)
        result3 = solution.Two_Sum_Hash_Map_Optimal(nums.copy(), target)
        result4 = solution.Two_Sum_Hash_Map_Two_Pass(nums.copy(), target)
        result5 = solution.Two_Sum_Enumerate(nums.copy(), target)
        
        print(f"Array: {nums}, Target: {target}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Two Pointer: {sorted(result2)}")
        print(f"Hash Map Optimal: {result3}")
        print(f"Hash Map Two Pass: {sorted(result4)}")
        print(f"Enumerate: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Two_Sum()

