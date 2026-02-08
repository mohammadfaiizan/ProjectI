"""
Problem: Buy and Sell Stock at Most Twice
URL: https://www.geeksforgeeks.org/maximum-profit-by-buying-and-selling-a-share-at-most-twice/

Problem Statement:
Given an array price[] where price[i] is the stock price on day i, find the maximum profit
achievable by buying and selling at most twice. You must sell before buying again.

Sample Input/Output:
Input: prices = [2, 30, 15, 10, 8, 25, 80]
Output: 100
Explanation: Buy at 2, sell at 30 (profit 28). Buy at 8, sell at 80 (profit 72). Total = 100.

Input: prices = [100, 30, 15, 10, 8, 25, 80]
Output: 72
Explanation: Buy at 8, sell at 80. Only one transaction needed.
"""


class Solution:
    def Max_Profit_Two_Pass_DP_Optimal(self, prices):
        """
        Two Pass DP - Forward pass for first transaction, backward for second
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(prices)
        if n < 2:
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

    def Max_Profit_State_Machine(self, prices):
        """
        State Machine - Track 4 states (buy1, sell1, buy2, sell2)
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        buy1 = float('-inf')
        sell1 = 0
        buy2 = float('-inf')
        sell2 = 0
        for price in prices:
            buy1 = max(buy1, -price)
            sell1 = max(sell1, buy1 + price)
            buy2 = max(buy2, sell1 - price)
            sell2 = max(sell2, buy2 + price)
        return sell2

    def Max_Profit_Valley_Peak_Unlimited(self, prices):
        """
        Valley Peak (Unlimited Transactions) - Sum all positive differences
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        profit = 0
        for i in range(1, len(prices)):
            diff = prices[i] - prices[i - 1]
            if diff > 0:
                profit += diff
        return profit


def Test_Buy_Sell_Stock_Twice():
    solution = Solution()

    class TestCase:
        def __init__(self, prices, expected):
            self.prices = prices
            self.expected = expected

    test_cases = [
        TestCase([2, 30, 15, 10, 8, 25, 80], 100),
        TestCase([100, 30, 15, 10, 8, 25, 80], 72),
        TestCase([10, 22, 5, 75, 65, 80], 87),
        TestCase([1, 2, 3, 4, 5], 4)
    ]

    for tc in test_cases:
        print(f"Prices: {tc.prices}, Expected: {tc.expected}")

        print("Two Pass DP:", solution.Max_Profit_Two_Pass_DP_Optimal(tc.prices))
        print("State Machine:", solution.Max_Profit_State_Machine(tc.prices))
        print("Valley Peak (unlimited):", solution.Max_Profit_Valley_Peak_Unlimited(tc.prices))

        print("-" * 50)


if __name__ == "__main__":
    Test_Buy_Sell_Stock_Twice()
