"""
Problem: Coin Change Problem - Minimum number of coins
URL: https://leetcode.com/problems/coin-change/

Problem Statement:
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.
You may assume that you have an infinite number of each kind of coin.

Sample Input/Output:
Input: coins = [1,3,4], amount = 6
Output: 2
Explanation: 6 = 3 + 3

Input: coins = [2], amount = 3
Output: -1

Input: coins = [1], amount = 0
Output: 0
"""

from typing import List
import math

class Solution:
    def Coin_Change_Recursive(self, coins: List[int], amount: int) -> int:
        """
        Recursive Brute Force - Try all combinations
        Time Complexity: O(amount^len(coins))
        Space Complexity: O(amount)
        """
        def Min_Coins(remaining: int) -> int:
            if remaining == 0:
                return 0
            
            if remaining < 0:
                return float('inf')
            
            min_count = float('inf')
            
            for coin in coins:
                result = Min_Coins(remaining - coin)
                if result != float('inf'):
                    min_count = min(min_count, result + 1)
            
            return min_count
        
        result = Min_Coins(amount)
        return result if result != float('inf') else -1
    
    def Coin_Change_Memoized(self, coins: List[int], amount: int) -> int:
        """
        Memoized DP - Top-down with caching
        Time Complexity: O(amount * len(coins))
        Space Complexity: O(amount)
        """
        memo = {}
        
        def Min_Coins(remaining: int) -> int:
            if remaining == 0:
                return 0
            
            if remaining < 0:
                return float('inf')
            
            if remaining in memo:
                return memo[remaining]
            
            min_count = float('inf')
            
            for coin in coins:
                result = Min_Coins(remaining - coin)
                if result != float('inf'):
                    min_count = min(min_count, result + 1)
            
            memo[remaining] = min_count
            return min_count
        
        result = Min_Coins(amount)
        return result if result != float('inf') else -1
    
    def Coin_Change_Tabulation_Optimal(self, coins: List[int], amount: int) -> int:
        """
        Tabulation DP Optimal - Bottom-up approach
        Time Complexity: O(amount * len(coins))
        Space Complexity: O(amount)
        """
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for i in range(1, amount + 1):
            for coin in coins:
                if i >= coin:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1
    
    def Coin_Change_BFS(self, coins: List[int], amount: int) -> int:
        """
        BFS Approach - Level-wise exploration
        Time Complexity: O(amount * len(coins))
        Space Complexity: O(amount)
        """
        if amount == 0:
            return 0
        
        from collections import deque
        
        queue = deque([amount])
        visited = {amount}
        level = 0
        
        while queue:
            level += 1
            
            for _ in range(len(queue)):
                current = queue.popleft()
                
                for coin in coins:
                    next_amount = current - coin
                    
                    if next_amount == 0:
                        return level
                    
                    if next_amount > 0 and next_amount not in visited:
                        visited.add(next_amount)
                        queue.append(next_amount)
        
        return -1
    
    def Coin_Change_Greedy_Plus_DP(self, coins: List[int], amount: int) -> int:
        """
        Greedy + DP - Optimized with greedy pruning
        Time Complexity: O(amount * len(coins))
        Space Complexity: O(amount)
        """
        coins.sort(reverse=True)
        
        def Min_Coins_Helper(remaining: int, coin_index: int, current_count: int, best: List[int]) -> None:
            if coin_index >= len(coins):
                return
            
            coin = coins[coin_index]
            max_use = remaining // coin
            
            for use_count in range(max_use, -1, -1):
                new_remaining = remaining - use_count * coin
                new_count = current_count + use_count
                
                if new_count >= best[0]:
                    break
                
                if new_remaining == 0:
                    best[0] = new_count
                    break
                
                Min_Coins_Helper(new_remaining, coin_index + 1, new_count, best)
        
        best = [float('inf')]
        Min_Coins_Helper(amount, 0, 0, best)
        
        return best[0] if best[0] != float('inf') else -1
    
    def Coin_Change_With_Coins_Used(self, coins: List[int], amount: int) -> tuple:
        """
        With Coins Used - Return minimum count and actual coins
        Time Complexity: O(amount * len(coins))
        Space Complexity: O(amount)
        """
        dp = [float('inf')] * (amount + 1)
        parent = [-1] * (amount + 1)
        dp[0] = 0
        
        for i in range(1, amount + 1):
            for coin in coins:
                if i >= coin and dp[i - coin] + 1 < dp[i]:
                    dp[i] = dp[i - coin] + 1
                    parent[i] = coin
        
        if dp[amount] == float('inf'):
            return -1, []
        
        coins_used = []
        current = amount
        
        while current > 0:
            coin_used = parent[current]
            coins_used.append(coin_used)
            current -= coin_used
        
        return dp[amount], coins_used

def Test_Coin_Change():
    solution = Solution()
    
    test_cases = [
        ([1,3,4], 6, 2),
        ([2], 3, -1),
        ([1], 0, 0),
        ([1,2,5], 11, 3),
        ([2,3,5], 9, 2),
        ([1,4,5], 8, 2)
    ]
    
    methods = [
        ("Recursive", solution.Coin_Change_Recursive),
        ("Memoized", solution.Coin_Change_Memoized),
        ("Tabulation Optimal", solution.Coin_Change_Tabulation_Optimal),
        ("BFS", solution.Coin_Change_BFS),
        ("Greedy Plus DP", solution.Coin_Change_Greedy_Plus_DP)
    ]
    
    for coins, amount, expected in test_cases:
        print(f"Coins: {coins}, Amount: {amount}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            try:
                if method_name == "Recursive" and amount > 20:
                    print(f"{method_name}: Skipped (too slow)")
                    continue
                
                result = method(coins.copy(), amount)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        min_count, coins_used = solution.Coin_Change_With_Coins_Used(coins.copy(), amount)
        print(f"With Coins Used: Count={min_count}, Coins={coins_used}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Coin_Change()
