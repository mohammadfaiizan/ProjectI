"""
Problem: Target Sum
URL: https://leetcode.com/problems/target-sum/description/

Problem Statement:
You are given an integer array nums and an integer target.
You want to build an expression out of nums by adding one of the symbols '+' and '-' before each integer in nums and then concatenate all the integers.
Return the number of different expressions that you can build, which evaluates to target.

Sample Input/Output:
Input: nums = [1,1,1,1,1], target = 3
Output: 5
Explanation: There are 5 ways to assign symbols to make the sum of nums be target 3.
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3

Input: nums = [1], target = 1
Output: 1
"""

from typing import List

class Solution:
    def Find_Target_Sum_Ways_Recursive(self, nums: List[int], target: int) -> int:
        """
        Recursive Brute Force - Try all sign combinations
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        def Count_Ways(index: int, current_sum: int) -> int:
            if index == len(nums):
                return 1 if current_sum == target else 0
            
            positive = Count_Ways(index + 1, current_sum + nums[index])
            negative = Count_Ways(index + 1, current_sum - nums[index])
            
            return positive + negative
        
        return Count_Ways(0, 0)
    
    def Find_Target_Sum_Ways_Memoized(self, nums: List[int], target: int) -> int:
        """
        Memoized DP - Top-down with caching
        Time Complexity: O(n * sum)
        Space Complexity: O(n * sum)
        """
        memo = {}
        
        def Count_Ways(index: int, current_sum: int) -> int:
            if index == len(nums):
                return 1 if current_sum == target else 0
            
            if (index, current_sum) in memo:
                return memo[(index, current_sum)]
            
            positive = Count_Ways(index + 1, current_sum + nums[index])
            negative = Count_Ways(index + 1, current_sum - nums[index])
            
            memo[(index, current_sum)] = positive + negative
            return memo[(index, current_sum)]
        
        return Count_Ways(0, 0)
    
    def Find_Target_Sum_Ways_Transform_Optimal(self, nums: List[int], target: int) -> int:
        """
        Transform to Subset Sum Optimal - Convert to subset sum problem
        Time Complexity: O(n * sum)
        Space Complexity: O(sum)
        """
        total_sum = sum(nums)
        
        if target > total_sum or target < -total_sum or (target + total_sum) % 2 != 0:
            return 0
        
        subset_sum = (target + total_sum) // 2
        
        dp = [0] * (subset_sum + 1)
        dp[0] = 1
        
        for num in nums:
            for j in range(subset_sum, num - 1, -1):
                dp[j] += dp[j - num]
        
        return dp[subset_sum]
    
    def Find_Target_Sum_Ways_2D_DP(self, nums: List[int], target: int) -> int:
        """
        2D DP - Traditional DP table approach
        Time Complexity: O(n * sum)
        Space Complexity: O(n * sum)
        """
        total_sum = sum(nums)
        
        if target > total_sum or target < -total_sum:
            return 0
        
        offset = total_sum
        dp = [[0 for _ in range(2 * total_sum + 1)] for _ in range(len(nums) + 1)]
        dp[0][offset] = 1
        
        for i in range(1, len(nums) + 1):
            for j in range(2 * total_sum + 1):
                if dp[i-1][j] > 0:
                    if j + nums[i-1] <= 2 * total_sum:
                        dp[i][j + nums[i-1]] += dp[i-1][j]
                    if j - nums[i-1] >= 0:
                        dp[i][j - nums[i-1]] += dp[i-1][j]
        
        return dp[len(nums)][target + offset]
    
    def Find_Target_Sum_Ways_Optimized_Transform(self, nums: List[int], target: int) -> int:
        """
        Optimized Transform - Handle edge cases efficiently
        Time Complexity: O(n * sum)
        Space Complexity: O(sum)
        """
        total_sum = sum(nums)
        
        if abs(target) > total_sum or (target + total_sum) % 2 != 0:
            return 0
        
        subset_sum = (target + total_sum) // 2
        
        if subset_sum < 0:
            return 0
        
        dp = [0] * (subset_sum + 1)
        dp[0] = 1
        
        for num in nums:
            for j in range(subset_sum, num - 1, -1):
                dp[j] += dp[j - num]
        
        return dp[subset_sum]
    
    def Find_Target_Sum_Ways_With_Expressions(self, nums: List[int], target: int) -> tuple:
        """
        With Expressions Tracking - Return count and sample expressions
        Time Complexity: O(2^n)
        Space Complexity: O(2^n)
        """
        expressions = []
        
        def Find_Expressions(index: int, current_sum: int, expression: str) -> None:
            if index == len(nums):
                if current_sum == target:
                    expressions.append(expression)
                return
            
            if index == 0:
                Find_Expressions(index + 1, nums[index], str(nums[index]))
                Find_Expressions(index + 1, -nums[index], f"-{nums[index]}")
            else:
                Find_Expressions(index + 1, current_sum + nums[index], f"{expression}+{nums[index]}")
                Find_Expressions(index + 1, current_sum - nums[index], f"{expression}-{nums[index]}")
        
        if len(nums) <= 10:
            Find_Expressions(0, 0, "")
        
        return len(expressions), expressions

def Test_Find_Target_Sum_Ways():
    solution = Solution()
    
    test_cases = [
        ([1,1,1,1,1], 3, 5),
        ([1], 1, 1),
        ([1], 2, 0),
        ([1,0], 1, 2),
        ([100], -200, 0)
    ]
    
    methods = [
        ("Recursive", solution.Find_Target_Sum_Ways_Recursive),
        ("Memoized", solution.Find_Target_Sum_Ways_Memoized),
        ("Transform Optimal", solution.Find_Target_Sum_Ways_Transform_Optimal),
        ("2D DP", solution.Find_Target_Sum_Ways_2D_DP),
        ("Optimized Transform", solution.Find_Target_Sum_Ways_Optimized_Transform)
    ]
    
    for nums, target, expected in test_cases:
        print(f"Array: {nums}, Target: {target}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            try:
                if method_name == "Recursive" and len(nums) > 15:
                    print(f"{method_name}: Skipped (too slow)")
                    continue
                
                result = method(nums.copy(), target)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        if len(nums) <= 5:
            count, expressions = solution.Find_Target_Sum_Ways_With_Expressions(nums.copy(), target)
            print(f"With Expressions: Count={count}")
            for expr in expressions[:5]:
                print(f"  {expr}")
            if len(expressions) > 5:
                print(f"  ... and {len(expressions) - 5} more")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Find_Target_Sum_Ways()
