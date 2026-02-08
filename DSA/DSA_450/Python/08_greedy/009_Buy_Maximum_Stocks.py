"""
Problem: Buy Maximum Stocks
URL: https://www.geeksforgeeks.org/buy-maximum-stocks-stocks-can-bought-th-day/

Problem Statement:
In a stock market, there is a product with its infinite stocks. The stock prices are given for N days, where arr[i] denotes the price of the stock on the ith day. There is a rule that a customer can buy at most i stock on the ith day. If the customer has an amount of k amount of money initially, find out the maximum number of stocks a customer can buy.

Sample Input/Output:
Input: price[] = {10, 7, 19}, k = 45
Output: 4
Explanation: Day 1: Buy 1 stock at 10, Day 2: Buy 2 stocks at 7 each, Day 3: Buy 1 stock at 19. Total = 1 + 2 + 1 = 4 stocks.
"""


class Solution:
    def Buy_Maximum_Stocks_Sort_Greedy(self, price, n, k):
        """
        Sort by price, greedily buy maximum stocks at lowest prices
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        stocks = [(price[i], i + 1) for i in range(n)]
        
        stocks.sort()
        
        stocks_bought = 0
        remaining = k
        
        for stock in stocks:
            max_buy = min(stock[1], remaining // stock[0])
            stocks_bought += max_buy
            remaining -= max_buy * stock[0]
            if remaining < stock[0]:
                break
        
        return stocks_bought


def Test_Buy_Maximum_Stocks():
    solution = Solution()
    price = [10, 7, 19]
    k = 45
    print(f"Maximum stocks: {solution.Buy_Maximum_Stocks_Sort_Greedy(price, len(price), k)}")


if __name__ == "__main__":
    Test_Buy_Maximum_Stocks()
