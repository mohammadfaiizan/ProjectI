"""
Problem: Subset (Partition Equal Subset Sum)
URL: https://leetcode.com/problems/partition-equal-subset-sum/description/

Problem Statement:
Given a non-empty array nums containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.

Sample Input/Output:
Input: nums = [1,5,11,5]
Output: true
Explanation: The array can be partitioned as [1, 5, 5] and [11].

Input: nums = [1,2,3,5]
Output: false
Explanation: The array cannot be partitioned into equal sum subsets.
"""

from typing import List

class Solution:
    def Can_Partition_Brute_Force(self, nums: List[int]) -> bool:
        """
        Brute Force - Check all possible subsets
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        total_sum = sum(nums)
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        
        def Generate_Subsets(index: int, current_sum: int) -> bool:
            if current_sum == target:
                return True
            if index >= len(nums) or current_sum > target:
                return False
            
            include = Generate_Subsets(index + 1, current_sum + nums[index])
            exclude = Generate_Subsets(index + 1, current_sum)
            
            return include or exclude
        
        return Generate_Subsets(0, 0)
    
    def Can_Partition_Recursive_Optimal(self, nums: List[int]) -> bool:
        """
        Recursive with Memoization - Optimal recursive solution
        Time Complexity: O(n * sum)
        Space Complexity: O(n * sum)
        """
        total_sum = sum(nums)
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        memo = {}
        
        def Can_Make_Sum(index: int, remaining_sum: int) -> bool:
            if remaining_sum == 0:
                return True
            if index >= len(nums) or remaining_sum < 0:
                return False
            
            if (index, remaining_sum) in memo:
                return memo[(index, remaining_sum)]
            
            include = Can_Make_Sum(index + 1, remaining_sum - nums[index])
            exclude = Can_Make_Sum(index + 1, remaining_sum)
            
            memo[(index, remaining_sum)] = include or exclude
            return memo[(index, remaining_sum)]
        
        return Can_Make_Sum(0, target)
    
    def Can_Partition_Dynamic_Programming(self, nums: List[int]) -> bool:
        """
        Dynamic Programming - Bottom-up approach
        Time Complexity: O(n * sum)
        Space Complexity: O(sum)
        """
        total_sum = sum(nums)
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        dp = [False] * (target + 1)
        dp[0] = True
        
        for num in nums:
            for j in range(target, num - 1, -1):
                dp[j] = dp[j] or dp[j - num]
        
        return dp[target]

def Test_Can_Partition():
    solution = Solution()
    
    test_cases = [
        ([1,5,11,5], True),
        ([1,2,3,5], False),
        ([1,2,5], False),
        ([1,1], True),
        ([2,2,1,1], True)
    ]
    
    for nums, expected in test_cases:
        result1 = solution.Can_Partition_Brute_Force(nums.copy())
        result2 = solution.Can_Partition_Recursive_Optimal(nums.copy())
        result3 = solution.Can_Partition_Dynamic_Programming(nums.copy())
        
        print(f"Array: {nums}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Recursive Optimal: {result2}")
        print(f"Dynamic Programming: {result3}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Can_Partition()
