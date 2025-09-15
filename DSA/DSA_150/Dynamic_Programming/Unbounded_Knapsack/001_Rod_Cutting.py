"""
Problem: Rod Cutting Problem
URL: https://www.geeksforgeeks.org/problems/rod-cutting0840/1

Problem Statement:
Given a rod of length n inches and an array of prices that includes prices of all pieces of size smaller than n. 
Determine the maximum value obtainable by cutting up the rod and selling the pieces.

Sample Input/Output:
Input: length = 8, prices = [1, 5, 8, 9, 10, 17, 17, 20]
Output: 22
Explanation: The maximum obtainable value is 22 (by cutting in two pieces of lengths 2 and 6)

Input: length = 8, prices = [3, 5, 8, 9, 10, 17, 17, 20]
Output: 24
Explanation: The maximum obtainable value is 24 (by cutting in eight pieces of length 1)
"""

from typing import List

class Solution:
    def Cut_Rod_Recursive(self, prices: List[int], n: int) -> int:
        """
        Recursive Brute Force - Try all possible cuts
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        if n <= 0:
            return 0
        
        max_val = 0
        
        for i in range(n):
            max_val = max(max_val, prices[i] + self.Cut_Rod_Recursive(prices, n - i - 1))
        
        return max_val
    
    def Cut_Rod_Memoized(self, prices: List[int], n: int) -> int:
        """
        Memoized DP - Top-down with caching
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        memo = {}
        
        def Cut_Rod_Helper(length: int) -> int:
            if length <= 0:
                return 0
            
            if length in memo:
                return memo[length]
            
            max_val = 0
            
            for i in range(min(length, len(prices))):
                max_val = max(max_val, prices[i] + Cut_Rod_Helper(length - i - 1))
            
            memo[length] = max_val
            return max_val
        
        return Cut_Rod_Helper(n)
    
    def Cut_Rod_Tabulation_Optimal(self, prices: List[int], n: int) -> int:
        """
        Tabulation DP Optimal - Bottom-up approach
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        dp = [0] * (n + 1)
        
        for i in range(1, n + 1):
            for j in range(i):
                if j < len(prices):
                    dp[i] = max(dp[i], prices[j] + dp[i - j - 1])
        
        return dp[n]
    
    def Cut_Rod_Space_Optimized(self, prices: List[int], n: int) -> int:
        """
        Space Optimized - Unbounded knapsack pattern
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        dp = [0] * (n + 1)
        
        for i in range(1, len(prices) + 1):
            for j in range(i, n + 1):
                dp[j] = max(dp[j], dp[j - i] + prices[i - 1])
        
        return dp[n]
    
    def Cut_Rod_With_Cuts_Tracking(self, prices: List[int], n: int) -> tuple:
        """
        With Cuts Tracking - Track optimal cuts
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        dp = [0] * (n + 1)
        cuts = [0] * (n + 1)
        
        for i in range(1, n + 1):
            for j in range(i):
                if j < len(prices) and prices[j] + dp[i - j - 1] > dp[i]:
                    dp[i] = prices[j] + dp[i - j - 1]
                    cuts[i] = j + 1
        
        cut_sizes = []
        length = n
        while length > 0:
            cut_sizes.append(cuts[length])
            length -= cuts[length]
        
        return dp[n], cut_sizes

def Test_Cut_Rod():
    solution = Solution()
    
    test_cases = [
        ([1, 5, 8, 9, 10, 17, 17, 20], 8, 22),
        ([3, 5, 8, 9, 10, 17, 17, 20], 8, 24),
        ([1, 5, 8, 9, 10, 17, 17, 20], 4, 10),
        ([2, 1, 1, 1], 4, 8),
        ([3, 5, 8], 3, 8)
    ]
    
    methods = [
        ("Recursive", solution.Cut_Rod_Recursive),
        ("Memoized", solution.Cut_Rod_Memoized),
        ("Tabulation Optimal", solution.Cut_Rod_Tabulation_Optimal),
        ("Space Optimized", solution.Cut_Rod_Space_Optimized)
    ]
    
    for prices, n, expected in test_cases:
        print(f"Prices: {prices}, Length: {n}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            try:
                if method_name == "Recursive" and n > 15:
                    print(f"{method_name}: Skipped (too slow)")
                    continue
                
                result = method(prices.copy(), n)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        max_value, cut_sizes = solution.Cut_Rod_With_Cuts_Tracking(prices.copy(), n)
        print(f"With Cuts: Value={max_value}, Cuts={cut_sizes}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Cut_Rod()
