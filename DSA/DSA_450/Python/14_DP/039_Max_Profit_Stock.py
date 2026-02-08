"""
Problem: Maximum Profit Stock
URL: https://practice.geeksforgeeks.org/problems/maximum-profit4657/1

Problem Statement:
In the stock market, a person buys a stock and sells it on some future date. Given the stock prices of N days in an array A[] and a positive integer K, find out the maximum profit a person can make in at most K transactions. A transaction is equivalent to (buying + selling) of a stock and new transaction can start only when the previous transaction has been completed.

Sample Input/Output:
Input: prices=[2,4,7,5,4,3,5], k=2
Output: 7
"""


class Solution:
    def Stock_K_Trans_DP(self, prices: list[int], k: int) -> int:
        """
        Dynamic Programming
        Time Complexity: O(n*k)
        Space Complexity: O(n*k)
        """
        n = len(prices)
        if n <= 1 or k == 0:
            return 0
        
        if k >= n // 2:
            profit = 0
            for i in range(1, n):
                if prices[i] > prices[i - 1]:
                    profit += prices[i] - prices[i - 1]
            return profit
        
        dp = [[0] * n for _ in range(k + 1)]
        
        for t in range(1, k + 1):
            max_diff = -prices[0]
            for i in range(1, n):
                dp[t][i] = max(dp[t][i - 1], prices[i] + max_diff)
                max_diff = max(max_diff, dp[t - 1][i] - prices[i])
        
        return dp[k][n - 1]
    
    def Stock_Two_Trans(self, prices: list[int]) -> int:
        """
        Two Transactions
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(prices)
        if n <= 1:
            return 0
        
        profit = [0] * n
        
        max_price = prices[n - 1]
        for i in range(n - 2, -1, -1):
            max_price = max(max_price, prices[i])
            profit[i] = max(profit[i + 1], max_price - prices[i])
        
        min_price = prices[0]
        for i in range(1, n):
            min_price = min(min_price, prices[i])
            profit[i] = max(profit[i - 1], profit[i] + (prices[i] - min_price))
        
        return profit[n - 1]


def Test_MaxProfitStock():
    solution = Solution()
    prices = [2, 4, 7, 5, 4, 3, 5]
    assert solution.Stock_K_Trans_DP(prices, 2) == 7
    assert solution.Stock_Two_Trans(prices) >= 7


if __name__ == "__main__":
    Test_MaxProfitStock()
