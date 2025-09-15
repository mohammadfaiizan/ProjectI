"""
Problem: House Robber
URL: https://leetcode.com/problems/house-robber/

Problem Statement:
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, 
the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected 
and it will automatically contact the police if two adjacent houses were broken into on the same night.
Given an integer array nums representing the amount of money of each house, 
return the maximum amount of money you can rob tonight without alerting the police.

Sample Input/Output:
Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3). Total amount you can rob = 1 + 3 = 4.

Input: nums = [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1). Total amount you can rob = 2 + 9 + 1 = 12.
"""

from typing import List

class Solution:
    def Rob_Recursive(self, nums: List[int]) -> int:
        """
        Recursive - Try both rob and not rob at each house
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        def Rob_From_Index(index: int) -> int:
            if index >= len(nums):
                return 0
            
            rob_current = nums[index] + Rob_From_Index(index + 2)
            skip_current = Rob_From_Index(index + 1)
            
            return max(rob_current, skip_current)
        
        return Rob_From_Index(0)
    
    def Rob_Memoized(self, nums: List[int]) -> int:
        """
        Memoized - Top-down DP with memoization
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        memo = {}
        
        def Rob_Memo(index: int) -> int:
            if index >= len(nums):
                return 0
            
            if index in memo:
                return memo[index]
            
            rob_current = nums[index] + Rob_Memo(index + 2)
            skip_current = Rob_Memo(index + 1)
            
            memo[index] = max(rob_current, skip_current)
            return memo[index]
        
        return Rob_Memo(0)
    
    def Rob_Tabulation_2D(self, nums: List[int]) -> int:
        """
        Tabulation 2D - Bottom-up DP with 2D state
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not nums:
            return 0
        
        n = len(nums)
        
        dp = [[0, 0] for _ in range(n)]
        dp[0][0] = 0
        dp[0][1] = nums[0]
        
        for i in range(1, n):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1])
            dp[i][1] = dp[i-1][0] + nums[i]
        
        return max(dp[n-1][0], dp[n-1][1])
    
    def Rob_Tabulation_1D_Optimal(self, nums: List[int]) -> int:
        """
        Tabulation 1D Optimal - Bottom-up DP with 1D array
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        
        for i in range(2, n):
            dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        
        return dp[n-1]
    
    def Rob_Space_Optimized(self, nums: List[int]) -> int:
        """
        Space Optimized - Use only two variables
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        
        prev2 = nums[0]
        prev1 = max(nums[0], nums[1])
        
        for i in range(2, len(nums)):
            current = max(prev1, prev2 + nums[i])
            prev2 = prev1
            prev1 = current
        
        return prev1
    
    def Rob_State_Machine(self, nums: List[int]) -> int:
        """
        State Machine - Track rob and not_rob states
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        rob = 0
        not_rob = 0
        
        for money in nums:
            new_rob = not_rob + money
            new_not_rob = max(rob, not_rob)
            
            rob = new_rob
            not_rob = new_not_rob
        
        return max(rob, not_rob)
    
    def Rob_With_Houses_Robbed(self, nums: List[int]) -> tuple:
        """
        With Houses Robbed - Return max money and indices of robbed houses
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not nums:
            return 0, []
        if len(nums) == 1:
            return nums[0], [0]
        
        n = len(nums)
        dp = [0] * n
        choice = [False] * n
        
        dp[0] = nums[0]
        choice[0] = True
        
        if nums[1] > nums[0]:
            dp[1] = nums[1]
            choice[1] = True
        else:
            dp[1] = nums[0]
            choice[1] = False
        
        for i in range(2, n):
            if dp[i-2] + nums[i] > dp[i-1]:
                dp[i] = dp[i-2] + nums[i]
                choice[i] = True
            else:
                dp[i] = dp[i-1]
                choice[i] = False
        
        robbed_houses = []
        i = n - 1
        
        while i >= 0:
            if choice[i]:
                robbed_houses.append(i)
                i -= 2
            else:
                i -= 1
        
        return dp[n-1], robbed_houses[::-1]
    
    def Rob_Greedy_Comparison(self, nums: List[int]) -> int:
        """
        Greedy Comparison - Compare with greedy approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        def Greedy_Rob(houses: List[int]) -> int:
            total = 0
            i = 0
            
            while i < len(houses):
                if i + 1 < len(houses) and houses[i + 1] > houses[i]:
                    total += houses[i + 1]
                    i += 2
                else:
                    total += houses[i]
                    i += 2
            
            return total
        
        greedy_result = Greedy_Rob(nums)
        optimal_result = self.Rob_Space_Optimized(nums)
        
        return optimal_result
    
    def Rob_Bottom_Up_Alternative(self, nums: List[int]) -> int:
        """
        Bottom Up Alternative - Different approach to bottom-up
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not nums:
            return 0
        
        include = 0
        exclude = 0
        
        for money in nums:
            new_exclude = max(include, exclude)
            include = exclude + money
            exclude = new_exclude
        
        return max(include, exclude)

def Test_Rob():
    solution = Solution()
    
    test_cases = [
        ([1,2,3,1], 4),
        ([2,7,9,3,1], 12),
        ([2,1,1,2], 4),
        ([5,1,3,9], 14),
        ([1], 1),
        ([2,3], 3),
        ([1,2,3,4,5], 9),
        ([10,5,2,7,8], 18)
    ]
    
    methods = [
        ("Memoized", solution.Rob_Memoized),
        ("Tabulation 2D", solution.Rob_Tabulation_2D),
        ("Tabulation 1D Optimal", solution.Rob_Tabulation_1D_Optimal),
        ("Space Optimized", solution.Rob_Space_Optimized),
        ("State Machine", solution.Rob_State_Machine),
        ("Greedy Comparison", solution.Rob_Greedy_Comparison),
        ("Bottom Up Alternative", solution.Rob_Bottom_Up_Alternative)
    ]
    
    for nums, expected in test_cases:
        print(f"Houses: {nums}")
        print(f"Expected: {expected}")
        
        if len(nums) <= 8:
            result_rec = solution.Rob_Recursive(nums.copy())
            print(f"Recursive: {result_rec}")
        
        for method_name, method in methods:
            try:
                result = method(nums.copy())
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        max_money, robbed_houses = solution.Rob_With_Houses_Robbed(nums.copy())
        print(f"With Houses: Money={max_money}, Robbed={robbed_houses}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Rob()
