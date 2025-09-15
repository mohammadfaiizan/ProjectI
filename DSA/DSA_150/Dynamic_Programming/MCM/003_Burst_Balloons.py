"""
Problem: Burst Balloons
URL: https://leetcode.com/problems/burst-balloons/description/

Problem Statement:
You are given n balloons, indexed from 0 to n - 1. Each balloon is painted with a number on it represented by an array nums.
You are asked to burst all the balloons.
If you burst the ith balloon, you will get nums[i - 1] * nums[i] * nums[i + 1] coins.
If i - 1 or i + 1 goes out of bounds of the array, then treat it as if there is a balloon with a 1 painted on it.
Return the maximum coins you can collect by bursting the balloons wisely.

Sample Input/Output:
Input: nums = [3,1,5,8]
Output: 167
Explanation:
nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 15 + 120 + 24 + 8 = 167

Input: nums = [1,5]
Output: 10
"""

from typing import List

class Solution:
    def Max_Coins_Brute_Force(self, nums: List[int]) -> int:
        """
        Brute Force - Try all possible bursting orders
        Time Complexity: O(n!)
        Space Complexity: O(n)
        """
        def Calculate_Coins(balloons: List[int], path: List[int]) -> int:
            if not balloons:
                return 0
            
            max_coins = 0
            
            for i in range(len(balloons)):
                left = balloons[i - 1] if i > 0 else 1
                right = balloons[i + 1] if i < len(balloons) - 1 else 1
                coins = left * balloons[i] * right
                
                new_balloons = balloons[:i] + balloons[i + 1:]
                new_path = path + [balloons[i]]
                
                total_coins = coins + Calculate_Coins(new_balloons, new_path)
                max_coins = max(max_coins, total_coins)
            
            return max_coins
        
        return Calculate_Coins(nums, [])
    
    def Max_Coins_Recursive(self, nums: List[int]) -> int:
        """
        Recursive - MCM pattern thinking last balloon to burst
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        nums_with_bounds = [1] + nums + [1]
        
        def MCM_Burst(left: int, right: int) -> int:
            if left + 1 >= right:
                return 0
            
            max_coins = 0
            
            for k in range(left + 1, right):
                coins = (MCM_Burst(left, k) + 
                        MCM_Burst(k, right) + 
                        nums_with_bounds[left] * nums_with_bounds[k] * nums_with_bounds[right])
                max_coins = max(max_coins, coins)
            
            return max_coins
        
        return MCM_Burst(0, len(nums_with_bounds) - 1)
    
    def Max_Coins_Memoized(self, nums: List[int]) -> int:
        """
        Memoized - Top-down DP with memoization
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        """
        nums_with_bounds = [1] + nums + [1]
        memo = {}
        
        def MCM_Burst_Memo(left: int, right: int) -> int:
            if left + 1 >= right:
                return 0
            
            if (left, right) in memo:
                return memo[(left, right)]
            
            max_coins = 0
            
            for k in range(left + 1, right):
                coins = (MCM_Burst_Memo(left, k) + 
                        MCM_Burst_Memo(k, right) + 
                        nums_with_bounds[left] * nums_with_bounds[k] * nums_with_bounds[right])
                max_coins = max(max_coins, coins)
            
            memo[(left, right)] = max_coins
            return max_coins
        
        return MCM_Burst_Memo(0, len(nums_with_bounds) - 1)
    
    def Max_Coins_Tabulation_Optimal(self, nums: List[int]) -> int:
        """
        Tabulation Optimal - Bottom-up DP
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        """
        nums_with_bounds = [1] + nums + [1]
        n = len(nums_with_bounds)
        
        dp = [[0] * n for _ in range(n)]
        
        for length in range(2, n):
            for left in range(n - length):
                right = left + length
                
                for k in range(left + 1, right):
                    coins = (dp[left][k] + 
                            dp[k][right] + 
                            nums_with_bounds[left] * nums_with_bounds[k] * nums_with_bounds[right])
                    dp[left][right] = max(dp[left][right], coins)
        
        return dp[0][n - 1]
    
    def Max_Coins_With_Sequence(self, nums: List[int]) -> tuple:
        """
        With Sequence - Return max coins and bursting sequence
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        """
        nums_with_bounds = [1] + nums + [1]
        n = len(nums_with_bounds)
        
        dp = [[0] * n for _ in range(n)]
        choice = [[0] * n for _ in range(n)]
        
        for length in range(2, n):
            for left in range(n - length):
                right = left + length
                
                for k in range(left + 1, right):
                    coins = (dp[left][k] + 
                            dp[k][right] + 
                            nums_with_bounds[left] * nums_with_bounds[k] * nums_with_bounds[right])
                    
                    if coins > dp[left][right]:
                        dp[left][right] = coins
                        choice[left][right] = k
        
        def Get_Burst_Sequence(left: int, right: int) -> List[int]:
            if left + 1 >= right:
                return []
            
            k = choice[left][right]
            sequence = []
            
            sequence.extend(Get_Burst_Sequence(left, k))
            sequence.extend(Get_Burst_Sequence(k, right))
            sequence.append(nums_with_bounds[k])
            
            return sequence
        
        burst_sequence = Get_Burst_Sequence(0, n - 1)
        return dp[0][n - 1], burst_sequence
    
    def Max_Coins_Space_Optimized(self, nums: List[int]) -> int:
        """
        Space Optimized - Optimize using gap method
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        """
        nums_with_bounds = [1] + nums + [1]
        n = len(nums_with_bounds)
        
        dp = [[0] * n for _ in range(n)]
        
        for gap in range(2, n):
            for left in range(n - gap):
                right = left + gap
                
                for k in range(left + 1, right):
                    coins = (dp[left][k] + 
                            dp[k][right] + 
                            nums_with_bounds[left] * nums_with_bounds[k] * nums_with_bounds[right])
                    dp[left][right] = max(dp[left][right], coins)
        
        return dp[0][n - 1]
    
    def Max_Coins_Divide_Conquer(self, nums: List[int]) -> int:
        """
        Divide Conquer - Pure divide and conquer approach
        Time Complexity: O(n³) with memoization
        Space Complexity: O(n²)
        """
        cache = {}
        
        def Burst(left: int, right: int, nums_extended: List[int]) -> int:
            if (left, right) in cache:
                return cache[(left, right)]
            
            if left + 1 == right:
                return 0
            
            max_coins = 0
            
            for i in range(left + 1, right):
                left_coins = Burst(left, i, nums_extended)
                right_coins = Burst(i, right, nums_extended)
                current_coins = nums_extended[left] * nums_extended[i] * nums_extended[right]
                
                total = left_coins + right_coins + current_coins
                max_coins = max(max_coins, total)
            
            cache[(left, right)] = max_coins
            return max_coins
        
        nums_extended = [1] + nums + [1]
        return Burst(0, len(nums_extended) - 1, nums_extended)

def Test_Max_Coins():
    solution = Solution()
    
    test_cases = [
        ([3,1,5,8], 167),
        ([1,5], 10),
        ([8,2,6,8,9,8,1,4,1,5,3,0,7,7,0,4,2], 1654),
        ([7,9,8,0,2], 144),
        ([1,2,3], 12)
    ]
    
    methods = [
        ("Recursive", solution.Max_Coins_Recursive),
        ("Memoized", solution.Max_Coins_Memoized),
        ("Tabulation Optimal", solution.Max_Coins_Tabulation_Optimal),
        ("Space Optimized", solution.Max_Coins_Space_Optimized),
        ("Divide Conquer", solution.Max_Coins_Divide_Conquer)
    ]
    
    for nums, expected in test_cases:
        print(f"Balloons: {nums}")
        print(f"Expected: {expected}")
        
        if len(nums) <= 6:
            result_bf = solution.Max_Coins_Brute_Force(nums.copy())
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method(nums.copy())
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        if len(nums) <= 8:
            max_coins, sequence = solution.Max_Coins_With_Sequence(nums.copy())
            print(f"With Sequence: Coins={max_coins}")
            print(f"Burst Order: {sequence}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Max_Coins()
