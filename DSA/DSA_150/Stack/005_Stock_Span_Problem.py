"""
Problem: Stock Span Problem
URL: https://leetcode.com/problems/online-stock-span/description/

Problem Statement:
Design an algorithm that collects daily price quotes for some stock and returns the span of that stock's price for the current day.
The span of the stock's price today is defined as the maximum number of consecutive days (starting from today and going backward) 
for which the stock price was less than or equal to today's price.

Sample Input/Output:
Input: prices = [100, 80, 60, 70, 60, 75, 85]
Output: [1, 1, 1, 2, 1, 4, 6]
Explanation: On day 0, span=1. On day 3, price=70, span=2 (days 2,3). On day 6, price=85, span=6 (days 1,2,3,4,5,6)

Input: prices = [31, 41, 48, 59, 79]
Output: [1, 2, 3, 4, 5]
Explanation: Each day has higher price than all previous days
"""

from typing import List

class Solution:
    def Stock_Span_Brute_Force(self, prices: List[int]) -> List[int]:
        """
        Brute Force Approach - Check all previous days
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        n = len(prices)
        spans = []
        
        for i in range(n):
            span = 1
            j = i - 1
            while j >= 0 and prices[j] <= prices[i]:
                span += 1
                j -= 1
            spans.append(span)
        
        return spans
    
    def Stock_Span_Nested_Loop(self, prices: List[int]) -> List[int]:
        """
        Nested Loop Approach - Count consecutive smaller prices
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        result = []
        
        for i in range(len(prices)):
            count = 1
            for j in range(i - 1, -1, -1):
                if prices[j] <= prices[i]:
                    count += 1
                else:
                    break
            result.append(count)
        
        return result
    
    def Stock_Span_Stack_Optimal(self, prices: List[int]) -> List[int]:
        """
        Stack Approach - Optimal solution using stack
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(prices)
        spans = [0] * n
        stack = []
        
        for i in range(n):
            while stack and prices[stack[-1]] <= prices[i]:
                stack.pop()
            
            spans[i] = i + 1 if not stack else i - stack[-1]
            stack.append(i)
        
        return spans
    
    def Stock_Span_Monotonic_Stack(self, prices: List[int]) -> List[int]:
        """
        Monotonic Stack Approach - Maintain decreasing stack
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = []
        stack = []
        
        for i, price in enumerate(prices):
            while stack and prices[stack[-1]] <= price:
                stack.pop()
            
            span = i + 1 if not stack else i - stack[-1]
            result.append(span)
            stack.append(i)
        
        return result
    
    def Stock_Span_Stack_With_Values(self, prices: List[int]) -> List[int]:
        """
        Stack with Price Values - Store (price, span) pairs
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = []
        stack = []
        
        for price in prices:
            span = 1
            
            while stack and stack[-1][0] <= price:
                span += stack.pop()[1]
            
            result.append(span)
            stack.append((price, span))
        
        return result

class Stock_Spanner:
    """
    Online Stock Spanner - Processes prices one by one
    """
    def __init__(self):
        self.stack = []
    
    def Next(self, price: int) -> int:
        span = 1
        
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1]
        
        self.stack.append((price, span))
        return span

def Test_Stock_Span():
    solution = Solution()
    
    test_cases = [
        ([100, 80, 60, 70, 60, 75, 85], [1, 1, 1, 2, 1, 4, 6]),
        ([31, 41, 48, 59, 79], [1, 2, 3, 4, 5]),
        ([10, 4, 5, 90, 120, 80], [1, 1, 2, 4, 5, 1]),
        ([1, 1, 1, 1, 1], [1, 2, 3, 4, 5])
    ]
    
    for prices, expected in test_cases:
        result1 = solution.Stock_Span_Brute_Force(prices.copy())
        result2 = solution.Stock_Span_Nested_Loop(prices.copy())
        result3 = solution.Stock_Span_Stack_Optimal(prices.copy())
        result4 = solution.Stock_Span_Monotonic_Stack(prices.copy())
        result5 = solution.Stock_Span_Stack_With_Values(prices.copy())
        
        print(f"Prices: {prices}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Nested Loop: {result2}")
        print(f"Stack Optimal: {result3}")
        print(f"Monotonic Stack: {result4}")
        print(f"Stack with Values: {result5}")
        
        stock_spanner = Stock_Spanner()
        online_result = []
        for price in prices:
            online_result.append(stock_spanner.Next(price))
        print(f"Online Spanner: {online_result}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Stock_Span()
