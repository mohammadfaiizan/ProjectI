"""
Problem: Equal Sum Partition
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
    def Can_Partition_Recursive(self, nums: List[int]) -> bool:
        """
        Recursive Brute Force - Check all partitions
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        total_sum = sum(nums)
        
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        
        def Can_Make_Sum(index: int, current_sum: int) -> bool:
            if current_sum == target:
                return True
            
            if index >= len(nums) or current_sum > target:
                return False
            
            include = Can_Make_Sum(index + 1, current_sum + nums[index])
            exclude = Can_Make_Sum(index + 1, current_sum)
            
            return include or exclude
        
        return Can_Make_Sum(0, 0)
    
    def Can_Partition_Memoized(self, nums: List[int]) -> bool:
        """
        Memoized DP - Top-down with caching
        Time Complexity: O(n * sum)
        Space Complexity: O(n * sum)
        """
        total_sum = sum(nums)
        
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        memo = {}
        
        def Can_Make_Sum(index: int, remaining: int) -> bool:
            if remaining == 0:
                return True
            
            if index >= len(nums) or remaining < 0:
                return False
            
            if (index, remaining) in memo:
                return memo[(index, remaining)]
            
            include = Can_Make_Sum(index + 1, remaining - nums[index])
            exclude = Can_Make_Sum(index + 1, remaining)
            
            memo[(index, remaining)] = include or exclude
            return memo[(index, remaining)]
        
        return Can_Make_Sum(0, target)
    
    def Can_Partition_Tabulation_Optimal(self, nums: List[int]) -> bool:
        """
        Tabulation DP Optimal - Bottom-up approach
        Time Complexity: O(n * sum)
        Space Complexity: O(n * sum)
        """
        total_sum = sum(nums)
        
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        n = len(nums)
        
        dp = [[False for _ in range(target + 1)] for _ in range(n + 1)]
        
        for i in range(n + 1):
            dp[i][0] = True
        
        for i in range(1, n + 1):
            for j in range(1, target + 1):
                if nums[i-1] <= j:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j - nums[i-1]]
                else:
                    dp[i][j] = dp[i-1][j]
        
        return dp[n][target]
    
    def Can_Partition_Space_Optimized(self, nums: List[int]) -> bool:
        """
        Space Optimized DP - Use 1D array
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
    
    def Can_Partition_Bitset(self, nums: List[int]) -> bool:
        """
        Bitset Approach - Use bitwise operations
        Time Complexity: O(n * sum / 32)
        Space Complexity: O(sum / 32)
        """
        total_sum = sum(nums)
        
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        bits = 1
        
        for num in nums:
            bits |= bits << num
        
        return bool(bits & (1 << target))

def Test_Can_Partition():
    solution = Solution()
    
    test_cases = [
        ([1,5,11,5], True),
        ([1,2,3,5], False),
        ([1,1,3,4,7], True),
        ([2,3,4,6], False),
        ([1,2,5], False),
        ([1,1], True)
    ]
    
    methods = [
        ("Recursive", solution.Can_Partition_Recursive),
        ("Memoized", solution.Can_Partition_Memoized),
        ("Tabulation Optimal", solution.Can_Partition_Tabulation_Optimal),
        ("Space Optimized", solution.Can_Partition_Space_Optimized),
        ("Bitset", solution.Can_Partition_Bitset)
    ]
    
    for nums, expected in test_cases:
        print(f"Array: {nums}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            try:
                if method_name == "Recursive" and len(nums) > 15:
                    print(f"{method_name}: Skipped (too slow)")
                    continue
                
                result = method(nums.copy())
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Can_Partition()
