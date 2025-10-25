"""
Problem: Minimum Limit of Balls in a Bag
URL: https://leetcode.com/problems/minimum-limit-of-balls-in-a-bag/

Problem Statement:
You are given an integer array nums where the ith bag contains nums[i] balls. 
You are also given an integer maxOperations.

You can perform the following operation at most maxOperations times:
- Take any bag of balls and divide it into two new bags with a positive number of balls.

Your penalty is the maximum number of balls in a bag. You want to minimize your penalty 
after the operations.

Return the minimum possible penalty after performing the operations.

Sample Input/Output:
Input: nums = [9], maxOperations = 2
Output: 3
Explanation: Split bag with 9 balls into (6, 3). Split bag with 6 balls into (3, 3).
             Bags are [3,3,3]. Maximum = 3.

Input: nums = [2,4,8,2], maxOperations = 4
Output: 2
Explanation: Split [8] into [4,4], [4,4,4,2,2,2]. Maximum = 2.

Input: nums = [7,17], maxOperations = 2
Output: 7
"""

from typing import List
import math

class Solution:
    def Minimum_Limit_Brute_Force(self, nums: List[int], maxOperations: int) -> int:
        """
        Brute Force Approach - Try all possible penalties
        Time Complexity: O(max(nums) * n)
        Space Complexity: O(1)
        """
        def Can_Achieve(penalty: int) -> bool:
            operations = 0
            for balls in nums:
                if balls > penalty:
                    operations += (balls - 1) // penalty
                    if operations > maxOperations:
                        return False
            return True
        
        max_balls = max(nums)
        
        for penalty in range(1, max_balls + 1):
            if Can_Achieve(penalty):
                return penalty
        
        return max_balls
    
    def Minimum_Limit_Binary_Search(self, nums: List[int], maxOperations: int) -> int:
        """
        Binary Search Approach
        Time Complexity: O(n * log(max(nums)))
        Space Complexity: O(1)
        """
        def Can_Achieve(penalty: int) -> bool:
            operations = 0
            for balls in nums:
                if balls > penalty:
                    operations += (balls - 1) // penalty
            return operations <= maxOperations
        
        left, right = 1, max(nums)
        result = right
        
        while left <= right:
            mid = (left + right) // 2
            
            if Can_Achieve(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result
    
    def Minimum_Limit_Binary_Search_Optimal(self, nums: List[int], maxOperations: int) -> int:
        """
        Binary Search with Math Optimization - Optimal solution
        Time Complexity: O(n * log(max(nums)))
        Space Complexity: O(1)
        """
        def Operations_Needed(penalty: int) -> int:
            operations = 0
            for balls in nums:
                operations += (balls - 1) // penalty
            return operations
        
        left, right = 1, max(nums)
        
        while left < right:
            mid = (left + right) // 2
            
            if Operations_Needed(mid) <= maxOperations:
                right = mid
            else:
                left = mid + 1
        
        return left
    
    def Minimum_Limit_Binary_Search_Ceil(self, nums: List[int], maxOperations: int) -> int:
        """
        Binary Search with Ceiling Formula
        Time Complexity: O(n * log(max(nums)))
        Space Complexity: O(1)
        """
        def Check_Valid(penalty: int) -> bool:
            total_ops = 0
            for balls in nums:
                if balls > penalty:
                    total_ops += math.ceil(balls / penalty) - 1
            return total_ops <= maxOperations
        
        lo, hi = 1, max(nums)
        
        while lo < hi:
            mid = (lo + hi) // 2
            if Check_Valid(mid):
                hi = mid
            else:
                lo = mid + 1
        
        return lo
    
    def Minimum_Limit_Binary_Search_Compact(self, nums: List[int], maxOperations: int) -> int:
        """
        Compact Binary Search Implementation
        Time Complexity: O(n * log(max(nums)))
        Space Complexity: O(1)
        """
        def Valid(penalty: int) -> bool:
            return sum((balls - 1) // penalty for balls in nums) <= maxOperations
        
        left, right = 1, max(nums)
        
        while left < right:
            mid = (left + right) // 2
            if Valid(mid):
                right = mid
            else:
                left = mid + 1
        
        return left

def Test_Minimum_Limit():
    solution = Solution()
    
    test_cases = [
        ([9], 2, 3),
        ([2,4,8,2], 4, 2),
        ([7,17], 2, 7),
        ([1], 0, 1),
        ([10], 1, 5),
        ([3,3,3], 3, 1),
        ([100], 10, 9)
    ]
    
    for nums, maxOps, expected in test_cases:
        result1 = solution.Minimum_Limit_Brute_Force(nums.copy(), maxOps)
        result2 = solution.Minimum_Limit_Binary_Search(nums.copy(), maxOps)
        result3 = solution.Minimum_Limit_Binary_Search_Optimal(nums.copy(), maxOps)
        result4 = solution.Minimum_Limit_Binary_Search_Ceil(nums.copy(), maxOps)
        result5 = solution.Minimum_Limit_Binary_Search_Compact(nums.copy(), maxOps)
        
        print(f"Bags: {nums}, MaxOps: {maxOps}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Binary Search: {result2}")
        print(f"Binary Search Optimal: {result3}")
        print(f"Binary Search Ceil: {result4}")
        print(f"Binary Search Compact: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Minimum_Limit()

