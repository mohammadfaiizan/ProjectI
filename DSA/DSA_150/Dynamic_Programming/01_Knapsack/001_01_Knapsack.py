"""
Problem: 0/1 Knapsack
URL: Classic 0/1 Knapsack Problem

Problem Statement:
Given weights and values of n items, put these items in a knapsack of capacity W to get the maximum total value in the knapsack.
In the 0-1 Knapsack problem, we are not allowed to break an item, either pick the complete item or don't pick it (0-1 property).

Sample Input/Output:
Input: W = 50, weights = [10, 20, 30], values = [60, 100, 120]
Output: 220
Explanation: Take items with weights 20 and 30 to get maximum value 220

Input: W = 10, weights = [5, 4, 6, 3], values = [10, 40, 30, 50]
Output: 90
Explanation: Take items with weights 4 and 3 to get maximum value 90
"""

from typing import List

class Solution:
    def Knapsack_Recursive_Brute_Force(self, W: int, weights: List[int], values: List[int], n: int) -> int:
        """
        Recursive Brute Force - Try all combinations
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        if n == 0 or W == 0:
            return 0
        
        if weights[n-1] > W:
            return self.Knapsack_Recursive_Brute_Force(W, weights, values, n-1)
        
        include = values[n-1] + self.Knapsack_Recursive_Brute_Force(W - weights[n-1], weights, values, n-1)
        exclude = self.Knapsack_Recursive_Brute_Force(W, weights, values, n-1)
        
        return max(include, exclude)
    
    def Knapsack_Memoized(self, W: int, weights: List[int], values: List[int], n: int) -> int:
        """
        Memoized DP - Top-down approach with caching
        Time Complexity: O(n * W)
        Space Complexity: O(n * W)
        """
        memo = {}
        
        def Knapsack_Helper(capacity: int, index: int) -> int:
            if index == 0 or capacity == 0:
                return 0
            
            if (capacity, index) in memo:
                return memo[(capacity, index)]
            
            if weights[index-1] > capacity:
                result = Knapsack_Helper(capacity, index-1)
            else:
                include = values[index-1] + Knapsack_Helper(capacity - weights[index-1], index-1)
                exclude = Knapsack_Helper(capacity, index-1)
                result = max(include, exclude)
            
            memo[(capacity, index)] = result
            return result
        
        return Knapsack_Helper(W, n)
    
    def Knapsack_Tabulation_Optimal(self, W: int, weights: List[int], values: List[int], n: int) -> int:
        """
        Tabulation DP Optimal - Bottom-up approach
        Time Complexity: O(n * W)
        Space Complexity: O(n * W)
        """
        dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(1, W + 1):
                if weights[i-1] <= w:
                    include = values[i-1] + dp[i-1][w - weights[i-1]]
                    exclude = dp[i-1][w]
                    dp[i][w] = max(include, exclude)
                else:
                    dp[i][w] = dp[i-1][w]
        
        return dp[n][W]
    
    def Knapsack_Space_Optimized(self, W: int, weights: List[int], values: List[int], n: int) -> int:
        """
        Space Optimized DP - Use 1D array
        Time Complexity: O(n * W)
        Space Complexity: O(W)
        """
        dp = [0 for _ in range(W + 1)]
        
        for i in range(n):
            for w in range(W, weights[i] - 1, -1):
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
        
        return dp[W]
    
    def Knapsack_With_Items_Tracking(self, W: int, weights: List[int], values: List[int], n: int) -> tuple:
        """
        With Items Tracking - Track which items are selected
        Time Complexity: O(n * W)
        Space Complexity: O(n * W)
        """
        dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(1, W + 1):
                if weights[i-1] <= w:
                    include = values[i-1] + dp[i-1][w - weights[i-1]]
                    exclude = dp[i-1][w]
                    dp[i][w] = max(include, exclude)
                else:
                    dp[i][w] = dp[i-1][w]
        
        selected_items = []
        w = W
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected_items.append(i-1)
                w -= weights[i-1]
        
        return dp[n][W], selected_items[::-1]

def Test_Knapsack():
    solution = Solution()
    
    test_cases = [
        (50, [10, 20, 30], [60, 100, 120], 3, 220),
        (10, [5, 4, 6, 3], [10, 40, 30, 50], 4, 90),
        (7, [1, 3, 4, 5], [1, 4, 5, 7], 4, 9),
        (8, [3, 4, 5], [30, 50, 60], 3, 90),
        (0, [1, 2, 3], [10, 20, 30], 3, 0)
    ]
    
    methods = [
        ("Recursive Brute Force", solution.Knapsack_Recursive_Brute_Force),
        ("Memoized", solution.Knapsack_Memoized),
        ("Tabulation Optimal", solution.Knapsack_Tabulation_Optimal),
        ("Space Optimized", solution.Knapsack_Space_Optimized)
    ]
    
    for W, weights, values, n, expected in test_cases:
        print(f"Capacity: {W}, Weights: {weights}, Values: {values}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            try:
                if method_name == "Recursive Brute Force" and n > 10:
                    print(f"{method_name}: Skipped (too slow)")
                    continue
                
                result = method(W, weights.copy(), values.copy(), n)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        max_value, items = solution.Knapsack_With_Items_Tracking(W, weights.copy(), values.copy(), n)
        print(f"With Items Tracking: Value={max_value}, Items={items}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Knapsack()
