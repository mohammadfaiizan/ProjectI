"""
Problem: Best time to buy and sell stocks
URL: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/

Problem Statement:
You are given an array prices where prices[i] is the price of a given stock on the ith day.
You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

Sample Input/Output:
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.
"""

from typing import List

class Solution:
    def Max_Profit_Brute_Force(self, prices: List[int]) -> int:
        """
        Brute Force Approach - Check all pairs
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        max_profit = 0
        n = len(prices)
        
        for i in range(n):
            for j in range(i + 1, n):
                profit = prices[j] - prices[i]
                max_profit = max(max_profit, profit)
        
        return max_profit
    
    def Max_Profit_Nested_Loop(self, prices: List[int]) -> int:
        """
        Nested Loop Approach - Find max price after each day
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        max_profit = 0
        
        for i in range(len(prices)):
            for j in range(i + 1, len(prices)):
                if prices[j] > prices[i]:
                    max_profit = max(max_profit, prices[j] - prices[i])
        
        return max_profit
    
    def Max_Profit_Sliding_Window_Optimal(self, prices: List[int]) -> int:
        """
        Sliding Window Approach - Track minimum price seen so far
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not prices:
            return 0
        
        min_price = prices[0]
        max_profit = 0
        
        for price in prices[1:]:
            max_profit = max(max_profit, price - min_price)
            min_price = min(min_price, price)
        
        return max_profit
    
    def Max_Profit_Two_Pointers(self, prices: List[int]) -> int:
        """
        Two Pointers Approach - Buy and sell pointers
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if len(prices) < 2:
            return 0
        
        left = 0  # buy pointer
        right = 1  # sell pointer
        max_profit = 0
        
        while right < len(prices):
            if prices[left] < prices[right]:
                profit = prices[right] - prices[left]
                max_profit = max(max_profit, profit)
            else:
                left = right
            
            right += 1
        
        return max_profit
    
    def Max_Profit_Dynamic_Programming(self, prices: List[int]) -> int:
        """
        Dynamic Programming Approach - Track min price and max profit
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not prices:
            return 0
        
        min_price_so_far = float('inf')
        max_profit_so_far = 0
        
        for price in prices:
            min_price_so_far = min(min_price_so_far, price)
            max_profit_so_far = max(max_profit_so_far, price - min_price_so_far)
        
        return max_profit_so_far
    
    def Max_Profit_Kadane_Algorithm(self, prices: List[int]) -> int:
        """
        Kadane's Algorithm Variant - Maximum subarray sum approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if len(prices) < 2:
            return 0
        
        max_ending_here = 0
        max_so_far = 0
        
        for i in range(1, len(prices)):
            max_ending_here = max(0, max_ending_here + (prices[i] - prices[i-1]))
            max_so_far = max(max_so_far, max_ending_here)
        
        return max_so_far

def Test_Max_Profit():
    solution = Solution()
    
    test_cases = [
        ([7,1,5,3,6,4], 5),
        ([7,6,4,3,1], 0),
        ([1,2,3,4,5], 4),
        ([1], 0),
        ([1,2], 1)
    ]
    
    for prices, expected in test_cases:
        result1 = solution.Max_Profit_Brute_Force(prices.copy())
        result2 = solution.Max_Profit_Nested_Loop(prices.copy())
        result3 = solution.Max_Profit_Sliding_Window_Optimal(prices.copy())
        result4 = solution.Max_Profit_Two_Pointers(prices.copy())
        result5 = solution.Max_Profit_Dynamic_Programming(prices.copy())
        result6 = solution.Max_Profit_Kadane_Algorithm(prices.copy())
        
        print(f"Prices: {prices}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Nested Loop: {result2}")
        print(f"Sliding Window Optimal: {result3}")
        print(f"Two Pointers: {result4}")
        print(f"Dynamic Programming: {result5}")
        print(f"Kadane Algorithm: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Max_Profit()
