"""
Problem: Coin Change Problem - Maximum number of ways
URL: https://leetcode.com/problems/coin-change-ii/

Problem Statement:
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
Return the number of combinations that make up that amount. If that amount of money cannot be made up by any combination of the coins, return 0.
You may assume that you have an infinite number of each kind of coin.

Sample Input/Output:
Input: amount = 5, coins = [1,2,5]
Output: 4
Explanation: there are four ways to make up the amount:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1

Input: amount = 3, coins = [2]
Output: 0
Explanation: the amount of 3 cannot be made up just with coins of 2.
"""

from typing import List

class Solution:
    def Change_Recursive(self, amount: int, coins: List[int]) -> int:
        """
        Recursive Brute Force - Try all combinations
        Time Complexity: O(2^(amount + len(coins)))
        Space Complexity: O(amount)
        """
        def Count_Ways(remaining: int, coin_index: int) -> int:
            if remaining == 0:
                return 1
            
            if remaining < 0 or coin_index >= len(coins):
                return 0
            
            include = Count_Ways(remaining - coins[coin_index], coin_index)
            exclude = Count_Ways(remaining, coin_index + 1)
            
            return include + exclude
        
        return Count_Ways(amount, 0)
    
    def Change_Memoized(self, amount: int, coins: List[int]) -> int:
        """
        Memoized DP - Top-down with caching
        Time Complexity: O(amount * len(coins))
        Space Complexity: O(amount * len(coins))
        """
        memo = {}
        
        def Count_Ways(remaining: int, coin_index: int) -> int:
            if remaining == 0:
                return 1
            
            if remaining < 0 or coin_index >= len(coins):
                return 0
            
            if (remaining, coin_index) in memo:
                return memo[(remaining, coin_index)]
            
            include = Count_Ways(remaining - coins[coin_index], coin_index)
            exclude = Count_Ways(remaining, coin_index + 1)
            
            memo[(remaining, coin_index)] = include + exclude
            return memo[(remaining, coin_index)]
        
        return Count_Ways(amount, 0)
    
    def Change_Tabulation_2D(self, amount: int, coins: List[int]) -> int:
        """
        Tabulation 2D DP - Bottom-up with 2D table
        Time Complexity: O(amount * len(coins))
        Space Complexity: O(amount * len(coins))
        """
        n = len(coins)
        dp = [[0 for _ in range(amount + 1)] for _ in range(n + 1)]
        
        for i in range(n + 1):
            dp[i][0] = 1
        
        for i in range(1, n + 1):
            for j in range(1, amount + 1):
                dp[i][j] = dp[i-1][j]
                
                if j >= coins[i-1]:
                    dp[i][j] += dp[i][j - coins[i-1]]
        
        return dp[n][amount]
    
    def Change_Space_Optimized_Optimal(self, amount: int, coins: List[int]) -> int:
        """
        Space Optimized Optimal - Unbounded knapsack pattern
        Time Complexity: O(amount * len(coins))
        Space Complexity: O(amount)
        """
        dp = [0] * (amount + 1)
        dp[0] = 1
        
        for coin in coins:
            for j in range(coin, amount + 1):
                dp[j] += dp[j - coin]
        
        return dp[amount]
    
    def Change_Order_Independent(self, amount: int, coins: List[int]) -> int:
        """
        Order Independent - Different order handling
        Time Complexity: O(amount * len(coins))
        Space Complexity: O(amount)
        """
        dp = [0] * (amount + 1)
        dp[0] = 1
        
        for i in range(1, amount + 1):
            for coin in coins:
                if i >= coin:
                    dp[i] += dp[i - coin]
        
        return dp[amount]
    
    def Change_With_Combinations(self, amount: int, coins: List[int]) -> tuple:
        """
        With Combinations Tracking - Return count and sample combinations
        Time Complexity: O(amount * len(coins) * result_count)
        Space Complexity: O(amount * result_count)
        """
        def Find_All_Combinations(remaining: int, coin_index: int, current_combo: List[int], all_combos: List[List[int]]) -> None:
            if remaining == 0:
                all_combos.append(current_combo[:])
                return
            
            if remaining < 0 or coin_index >= len(coins):
                return
            
            current_combo.append(coins[coin_index])
            Find_All_Combinations(remaining - coins[coin_index], coin_index, current_combo, all_combos)
            current_combo.pop()
            
            Find_All_Combinations(remaining, coin_index + 1, current_combo, all_combos)
        
        if amount <= 20:
            all_combinations = []
            Find_All_Combinations(amount, 0, [], all_combinations)
            return len(all_combinations), all_combinations
        else:
            return self.Change_Space_Optimized_Optimal(amount, coins), []

def Test_Change():
    solution = Solution()
    
    test_cases = [
        (5, [1,2,5], 4),
        (3, [2], 0),
        (10, [10], 1),
        (4, [1,2,3], 4),
        (6, [1,2,3], 7)
    ]
    
    methods = [
        ("Recursive", solution.Change_Recursive),
        ("Memoized", solution.Change_Memoized),
        ("Tabulation 2D", solution.Change_Tabulation_2D),
        ("Space Optimized Optimal", solution.Change_Space_Optimized_Optimal),
        ("Order Independent", solution.Change_Order_Independent)
    ]
    
    for amount, coins, expected in test_cases:
        print(f"Amount: {amount}, Coins: {coins}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            try:
                if method_name == "Recursive" and amount > 20:
                    print(f"{method_name}: Skipped (too slow)")
                    continue
                
                result = method(amount, coins.copy())
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        if amount <= 10:
            count, combinations = solution.Change_With_Combinations(amount, coins.copy())
            print(f"With Combinations: Count={count}")
            for combo in combinations:
                print(f"  {combo}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Change()
