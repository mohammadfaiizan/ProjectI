"""
Problem: Coin Change
URL: https://practice.geeksforgeeks.org/problems/coin-change2448/1

Problem Statement:
Given a value N, find the number of ways to make change for N cents, if we have infinite supply of each of S = {S1, S2, .. , SM} valued coins.

Sample Input/Output:
Input: coins = [1, 2, 3], amount = 4
Output: 4
Explanation: {1,1,1,1}, {1,1,2}, {2,2}, {1,3}
"""

class Solution:
    def Coin_Change_DP_Tabulation(self, coins, n, amount):
        """
        DP Tabulation approach
        Time Complexity: O(n*amount)
        Space Complexity: O(amount)
        """
        dp = [0] * (amount+1)
        dp[0] = 1
        for i in range(n):
            for j in range(coins[i], amount+1):
                dp[j] += dp[j-coins[i]]
        return dp[amount]

    def Coin_Change_Recursive_Memo(self, coins, n, amount):
        """
        Recursive Memoization approach
        Time Complexity: O(n*amount)
        Space Complexity: O(n*amount)
        """
        memo = [[-1] * (amount+1) for _ in range(n+1)]
        return self.Coin_Change_Memo_Helper(coins, n, amount, memo)

    def Coin_Change_Memo_Helper(self, coins, n, amount, memo):
        if amount == 0:
            return 1
        if n == 0:
            return 0
        if memo[n][amount] != -1:
            return memo[n][amount]
        if coins[n-1] > amount:
            memo[n][amount] = self.Coin_Change_Memo_Helper(coins, n-1, amount, memo)
        else:
            memo[n][amount] = self.Coin_Change_Memo_Helper(coins, n, amount-coins[n-1], memo) + \
                              self.Coin_Change_Memo_Helper(coins, n-1, amount, memo)
        return memo[n][amount]

def Test_Coin_Change():
    solution = Solution()
    coins = [1, 2, 3]
    amount = 4
    
    print("DP Tabulation:", solution.Coin_Change_DP_Tabulation(coins, len(coins), amount))
    print("Recursive Memo:", solution.Coin_Change_Recursive_Memo(coins, len(coins), amount))

if __name__ == "__main__":
    Test_Coin_Change()
