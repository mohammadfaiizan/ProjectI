"""
Problem: Best Time to Buy and Sell Stock
URL: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

Problem Statement:
Given an array prices[] where prices[i] is the price of a stock on the ith day.
Maximize profit by choosing a single day to buy and a different day in the future to sell.
Return max profit, or 0 if no profit is possible.

Sample Input/Output:
Input: prices = [7, 1, 5, 3, 6, 4]
Output: 5
Explanation: Buy on day 2 (price=1), sell on day 5 (price=6), profit = 5.

Input: prices = [7, 6, 4, 3, 1]
Output: 0
Explanation: No profitable transaction possible.
"""


class Solution:
    def Max_Profit_Single_Pass_Optimal(self, prices):
        """
        Single Pass - Track minimum price and maximum profit
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        min_price = float('inf')
        max_profit = 0
        for price in prices:
            min_price = min(min_price, price)
            max_profit = max(max_profit, price - min_price)
        return max_profit

    def Max_Profit_Kadane_Variant(self, prices):
        """
        Kadane's Variant - Max subarray on price differences
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        local = global_max = 0
        for i in range(1, len(prices)):
            local = max(0, local + prices[i] - prices[i - 1])
            global_max = max(local, global_max)
        return global_max

    def Max_Profit_Brute_Force(self, prices):
        """
        Brute Force - Check all pairs of buy and sell days
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        max_profit = 0
        for i in range(len(prices)):
            for j in range(i + 1, len(prices)):
                max_profit = max(max_profit, prices[j] - prices[i])
        return max_profit


def Test_Best_Time_Buy_Sell_Stock():
    solution = Solution()

    class TestCase:
        def __init__(self, prices, expected):
            self.prices = prices
            self.expected = expected

    test_cases = [
        TestCase([7, 1, 5, 3, 6, 4], 5),
        TestCase([7, 6, 4, 3, 1], 0),
        TestCase([2, 4, 1], 2),
        TestCase([1, 2], 1)
    ]

    for tc in test_cases:
        print(f"Prices: {tc.prices}, Expected: {tc.expected}")

        print("Single Pass:", solution.Max_Profit_Single_Pass_Optimal(tc.prices))
        print("Kadane's Variant:", solution.Max_Profit_Kadane_Variant(tc.prices))
        print("Brute Force:", solution.Max_Profit_Brute_Force(tc.prices))

        print("-" * 50)


if __name__ == "__main__":
    Test_Best_Time_Buy_Sell_Stock()
