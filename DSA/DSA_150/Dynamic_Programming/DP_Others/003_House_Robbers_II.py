"""
Problem: House Robber II
URL: https://leetcode.com/problems/house-robber-ii/

Problem Statement:
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. 
All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. 
Meanwhile, adjacent houses have security systems connected and it will automatically contact the police 
if two adjacent houses were broken into on the same night.
Given an integer array nums representing the amount of money of each house, 
return the maximum amount of money you can rob tonight without alerting the police.

Sample Input/Output:
Input: nums = [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.

Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3). Total amount you can rob = 1 + 3 = 4.
"""

from typing import List

class Solution:
    def Rob_II_Brute_Force(self, nums: List[int]) -> int:
        """
        Brute Force - Try all valid combinations
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])
        
        def Is_Valid_Combination(indices: List[int]) -> bool:
            if not indices:
                return True
            
            indices_set = set(indices)
            n = len(nums)
            
            for i in indices:
                if (i + 1) % n in indices_set or (i - 1) % n in indices_set:
                    return False
            
            return True
        
        def Generate_All_Combinations(index: int, current: List[int]) -> int:
            if index >= len(nums):
                if Is_Valid_Combination(current):
                    return sum(nums[i] for i in current)
                return 0
            
            exclude = Generate_All_Combinations(index + 1, current)
            
            current.append(index)
            include = Generate_All_Combinations(index + 1, current)
            current.pop()
            
            return max(exclude, include)
        
        return Generate_All_Combinations(0, [])
    
    def Rob_II_Two_Cases_Optimal(self, nums: List[int]) -> int:
        """
        Two Cases Optimal - Rob first or not rob first
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])
        
        def Rob_Linear(houses: List[int]) -> int:
            if not houses:
                return 0
            if len(houses) == 1:
                return houses[0]
            
            prev2 = houses[0]
            prev1 = max(houses[0], houses[1])
            
            for i in range(2, len(houses)):
                current = max(prev1, prev2 + houses[i])
                prev2 = prev1
                prev1 = current
            
            return prev1
        
        case1 = Rob_Linear(nums[:-1])
        case2 = Rob_Linear(nums[1:])
        
        return max(case1, case2)
    
    def Rob_II_DP_With_States(self, nums: List[int]) -> int:
        """
        DP With States - Track rob_first and not_rob_first states
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])
        
        n = len(nums)
        
        rob_first = [0] * n
        not_rob_first = [0] * n
        
        rob_first[0] = nums[0]
        rob_first[1] = nums[0]
        
        not_rob_first[0] = 0
        not_rob_first[1] = nums[1]
        
        for i in range(2, n):
            if i == n - 1:
                rob_first[i] = rob_first[i-1]
            else:
                rob_first[i] = max(rob_first[i-1], rob_first[i-2] + nums[i])
            
            not_rob_first[i] = max(not_rob_first[i-1], not_rob_first[i-2] + nums[i])
        
        return max(rob_first[n-1], not_rob_first[n-1])
    
    def Rob_II_Memoized(self, nums: List[int]) -> int:
        """
        Memoized - Top-down DP with memoization
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])
        
        memo = {}
        
        def Rob_Memo(index: int, can_rob_last: bool) -> int:
            if index >= len(nums):
                return 0
            
            if index == len(nums) - 1 and not can_rob_last:
                return 0
            
            if (index, can_rob_last) in memo:
                return memo[(index, can_rob_last)]
            
            if index == 0:
                rob_current = nums[index] + Rob_Memo(index + 2, False)
                skip_current = Rob_Memo(index + 1, True)
            else:
                rob_current = nums[index] + Rob_Memo(index + 2, can_rob_last)
                skip_current = Rob_Memo(index + 1, can_rob_last)
            
            memo[(index, can_rob_last)] = max(rob_current, skip_current)
            return memo[(index, can_rob_last)]
        
        return Rob_Memo(0, True)
    
    def Rob_II_State_Machine(self, nums: List[int]) -> int:
        """
        State Machine - Use state machine approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])
        
        def Rob_With_State_Machine(houses: List[int]) -> int:
            rob = 0
            not_rob = 0
            
            for money in houses:
                new_rob = not_rob + money
                new_not_rob = max(rob, not_rob)
                
                rob = new_rob
                not_rob = new_not_rob
            
            return max(rob, not_rob)
        
        case1 = Rob_With_State_Machine(nums[:-1])
        case2 = Rob_With_State_Machine(nums[1:])
        
        return max(case1, case2)
    
    def Rob_II_Rolling_Array(self, nums: List[int]) -> int:
        """
        Rolling Array - Use rolling array technique
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])
        
        def Rob_Range(start: int, end: int) -> int:
            prev_rob = 0
            prev_not_rob = 0
            
            for i in range(start, end + 1):
                current_rob = prev_not_rob + nums[i]
                current_not_rob = max(prev_rob, prev_not_rob)
                
                prev_rob = current_rob
                prev_not_rob = current_not_rob
            
            return max(prev_rob, prev_not_rob)
        
        return max(Rob_Range(0, len(nums) - 2), Rob_Range(1, len(nums) - 1))
    
    def Rob_II_With_Houses_Robbed(self, nums: List[int]) -> tuple:
        """
        With Houses Robbed - Return max money and indices of robbed houses
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not nums:
            return 0, []
        if len(nums) == 1:
            return nums[0], [0]
        if len(nums) == 2:
            return max(nums[0], nums[1]), [0 if nums[0] > nums[1] else 1]
        
        def Rob_Linear_With_Indices(houses: List[int], offset: int) -> tuple:
            if not houses:
                return 0, []
            if len(houses) == 1:
                return houses[0], [offset]
            
            n = len(houses)
            dp = [0] * n
            choice = [False] * n
            
            dp[0] = houses[0]
            choice[0] = True
            
            if houses[1] > houses[0]:
                dp[1] = houses[1]
                choice[1] = True
            else:
                dp[1] = houses[0]
                choice[1] = False
            
            for i in range(2, n):
                if dp[i-2] + houses[i] > dp[i-1]:
                    dp[i] = dp[i-2] + houses[i]
                    choice[i] = True
                else:
                    dp[i] = dp[i-1]
                    choice[i] = False
            
            robbed_houses = []
            i = n - 1
            
            while i >= 0:
                if choice[i]:
                    robbed_houses.append(i + offset)
                    i -= 2
                else:
                    i -= 1
            
            return dp[n-1], robbed_houses[::-1]
        
        case1_money, case1_houses = Rob_Linear_With_Indices(nums[:-1], 0)
        case2_money, case2_houses = Rob_Linear_With_Indices(nums[1:], 1)
        
        if case1_money >= case2_money:
            return case1_money, case1_houses
        else:
            return case2_money, case2_houses
    
    def Rob_II_Comparison_With_Linear(self, nums: List[int]) -> tuple:
        """
        Comparison With Linear - Compare circular vs linear results
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        def Rob_Linear(houses: List[int]) -> int:
            if not houses:
                return 0
            if len(houses) == 1:
                return houses[0]
            
            prev2 = houses[0]
            prev1 = max(houses[0], houses[1])
            
            for i in range(2, len(houses)):
                current = max(prev1, prev2 + houses[i])
                prev2 = prev1
                prev1 = current
            
            return prev1
        
        linear_result = Rob_Linear(nums)
        circular_result = self.Rob_II_Two_Cases_Optimal(nums)
        
        return circular_result, linear_result

def Test_Rob_II():
    solution = Solution()
    
    test_cases = [
        ([2,3,2], 3),
        ([1,2,3,1], 4),
        ([1,2,3], 3),
        ([2,7,9,3,1], 11),
        ([5], 5),
        ([1,2], 2),
        ([2,1,1,2], 3),
        ([4,1,2,9], 10)
    ]
    
    methods = [
        ("Two Cases Optimal", solution.Rob_II_Two_Cases_Optimal),
        ("DP With States", solution.Rob_II_DP_With_States),
        ("Memoized", solution.Rob_II_Memoized),
        ("State Machine", solution.Rob_II_State_Machine),
        ("Rolling Array", solution.Rob_II_Rolling_Array)
    ]
    
    for nums, expected in test_cases:
        print(f"Houses: {nums}")
        print(f"Expected: {expected}")
        
        if len(nums) <= 6:
            result_bf = solution.Rob_II_Brute_Force(nums.copy())
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method(nums.copy())
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        max_money, robbed_houses = solution.Rob_II_With_Houses_Robbed(nums.copy())
        print(f"With Houses: Money={max_money}, Robbed={robbed_houses}")
        
        circular_result, linear_result = solution.Rob_II_Comparison_With_Linear(nums.copy())
        print(f"Comparison: Circular={circular_result}, Linear={linear_result}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Rob_II()
