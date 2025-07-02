"""
Array Dynamic Programming Problems
==================================

Topics: DP problems using arrays, optimization problems
Companies: Google, Facebook, Amazon, Microsoft, Netflix
Difficulty: Medium to Hard
"""

from typing import List
import sys

class ArrayDynamicProgramming:
    
    # ==========================================
    # 1. CLASSIC DP PROBLEMS WITH ARRAYS
    # ==========================================
    
    def fibonacci(self, n: int) -> int:
        """Basic Fibonacci using array DP
        Time: O(n), Space: O(n) -> O(1) optimized
        """
        if n <= 1:
            return n
        
        # Space optimized version
        prev2, prev1 = 0, 1
        for i in range(2, n + 1):
            current = prev1 + prev2
            prev2, prev1 = prev1, current
        
        return prev1
    
    def climbing_stairs(self, n: int) -> int:
        """LC 70: Climbing Stairs
        Time: O(n), Space: O(1)
        """
        if n <= 2:
            return n
        
        prev2, prev1 = 1, 2
        for i in range(3, n + 1):
            current = prev1 + prev2
            prev2, prev1 = prev1, current
        
        return prev1
    
    def house_robber(self, nums: List[int]) -> int:
        """LC 198: House Robber
        Time: O(n), Space: O(1)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        
        prev2, prev1 = nums[0], max(nums[0], nums[1])
        
        for i in range(2, len(nums)):
            current = max(prev1, prev2 + nums[i])
            prev2, prev1 = prev1, current
        
        return prev1
    
    def house_robber_circular(self, nums: List[int]) -> int:
        """LC 213: House Robber II (circular arrangement)
        Time: O(n), Space: O(1)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums)
        
        def rob_linear(houses):
            prev2, prev1 = 0, 0
            for money in houses:
                current = max(prev1, prev2 + money)
                prev2, prev1 = prev1, current
            return prev1
        
        # Case 1: Rob houses 0 to n-2 (exclude last)
        # Case 2: Rob houses 1 to n-1 (exclude first)
        return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))
    
    # ==========================================
    # 2. SUBARRAY/SUBSEQUENCE DP PROBLEMS
    # ==========================================
    
    def longest_increasing_subsequence(self, nums: List[int]) -> int:
        """LC 300: Longest Increasing Subsequence
        Time: O(n log n), Space: O(n)
        """
        if not nums:
            return 0
        
        from bisect import bisect_left
        tails = []
        
        for num in nums:
            pos = bisect_left(tails, num)
            if pos == len(tails):
                tails.append(num)
            else:
                tails[pos] = num
        
        return len(tails)
    
    def longest_common_subsequence(self, text1: str, text2: str) -> int:
        """LC 1143: Longest Common Subsequence
        Time: O(m*n), Space: O(min(m,n))
        """
        if len(text2) < len(text1):
            text1, text2 = text2, text1
        
        prev = [0] * (len(text1) + 1)
        
        for i in range(1, len(text2) + 1):
            curr = [0] * (len(text1) + 1)
            for j in range(1, len(text1) + 1):
                if text2[i-1] == text1[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(curr[j-1], prev[j])
            prev = curr
        
        return prev[len(text1)]
    
    def maximum_subarray_sum(self, nums: List[int]) -> int:
        """LC 53: Maximum Subarray (Kadane's Algorithm)
        Time: O(n), Space: O(1)
        """
        max_ending_here = max_so_far = nums[0]
        
        for i in range(1, len(nums)):
            max_ending_here = max(nums[i], max_ending_here + nums[i])
            max_so_far = max(max_so_far, max_ending_here)
        
        return max_so_far
    
    def maximum_product_subarray(self, nums: List[int]) -> int:
        """LC 152: Maximum Product Subarray
        Time: O(n), Space: O(1)
        """
        if not nums:
            return 0
        
        max_prod = min_prod = result = nums[0]
        
        for i in range(1, len(nums)):
            if nums[i] < 0:
                max_prod, min_prod = min_prod, max_prod
            
            max_prod = max(nums[i], max_prod * nums[i])
            min_prod = min(nums[i], min_prod * nums[i])
            
            result = max(result, max_prod)
        
        return result
    
    # ==========================================
    # 3. MATRIX DP PROBLEMS
    # ==========================================
    
    def unique_paths(self, m: int, n: int) -> int:
        """LC 62: Unique Paths in grid
        Time: O(m*n), Space: O(n)
        """
        dp = [1] * n
        
        for i in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j-1]
        
        return dp[n-1]
    
    def unique_paths_with_obstacles(self, obstacleGrid: List[List[int]]) -> int:
        """LC 63: Unique Paths II with obstacles
        Time: O(m*n), Space: O(n)
        """
        if not obstacleGrid or obstacleGrid[0][0] == 1:
            return 0
        
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [0] * n
        dp[0] = 1
        
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    dp[j] = 0
                elif j > 0:
                    dp[j] += dp[j-1]
        
        return dp[n-1]
    
    def minimum_path_sum(self, grid: List[List[int]]) -> int:
        """LC 64: Minimum Path Sum
        Time: O(m*n), Space: O(n)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        dp = [float('inf')] * n
        dp[0] = grid[0][0]
        
        # Fill first row
        for j in range(1, n):
            dp[j] = dp[j-1] + grid[0][j]
        
        # Fill remaining rows
        for i in range(1, m):
            dp[0] += grid[i][0]
            for j in range(1, n):
                dp[j] = min(dp[j], dp[j-1]) + grid[i][j]
        
        return dp[n-1]
    
    # ==========================================
    # 4. KNAPSACK VARIATIONS
    # ==========================================
    
    def coin_change(self, coins: List[int], amount: int) -> int:
        """LC 322: Coin Change (Unbounded Knapsack)
        Time: O(amount * coins), Space: O(amount)
        """
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for coin in coins:
            for i in range(coin, amount + 1):
                if dp[i - coin] != float('inf'):
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1
    
    def combination_sum_iv(self, nums: List[int], target: int) -> int:
        """LC 377: Combination Sum IV
        Time: O(target * nums), Space: O(target)
        """
        dp = [0] * (target + 1)
        dp[0] = 1
        
        for i in range(1, target + 1):
            for num in nums:
                if i >= num:
                    dp[i] += dp[i - num]
        
        return dp[target]
    
    def can_partition(self, nums: List[int]) -> bool:
        """LC 416: Partition Equal Subset Sum (0/1 Knapsack)
        Time: O(n * sum), Space: O(sum)
        """
        total_sum = sum(nums)
        
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        dp = [False] * (target + 1)
        dp[0] = True
        
        for num in nums:
            for j in range(target, num - 1, -1):
                dp[j] = dp[j] or dp[j - num]
        
        return dp[target]
    
    # ==========================================
    # 5. ADVANCED ARRAY DP
    # ==========================================
    
    def buy_sell_stock_with_cooldown(self, prices: List[int]) -> int:
        """LC 309: Best Time to Buy and Sell Stock with Cooldown
        Time: O(n), Space: O(1)
        """
        if len(prices) <= 1:
            return 0
        
        # hold, sold, rest
        hold = -prices[0]
        sold = 0
        rest = 0
        
        for i in range(1, len(prices)):
            prev_hold, prev_sold, prev_rest = hold, sold, rest
            
            hold = max(prev_hold, prev_rest - prices[i])
            sold = prev_hold + prices[i]
            rest = max(prev_rest, prev_sold)
        
        return max(sold, rest)
    
    def max_sum_no_adjacent(self, nums: List[int]) -> int:
        """Maximum sum with no two adjacent elements
        Time: O(n), Space: O(1)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        
        incl = nums[0]
        excl = 0
        
        for i in range(1, len(nums)):
            new_excl = max(incl, excl)
            incl = excl + nums[i]
            excl = new_excl
        
        return max(incl, excl)

# Test Examples
def run_examples():
    adp = ArrayDynamicProgramming()
    
    print("=== ARRAY DYNAMIC PROGRAMMING EXAMPLES ===\n")
    
    # Classic DP problems
    print("1. CLASSIC DP PROBLEMS:")
    n = 10
    fib_result = adp.fibonacci(n)
    print(f"Fibonacci({n}): {fib_result}")
    
    stairs = 5
    climb_result = adp.climbing_stairs(stairs)
    print(f"Climbing stairs({stairs}): {climb_result}")
    
    houses = [2, 7, 9, 3, 1]
    rob_result = adp.house_robber(houses)
    print(f"House robber {houses}: {rob_result}")
    
    # Subarray problems
    print("\n2. SUBARRAY/SUBSEQUENCE PROBLEMS:")
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    lis_result = adp.longest_increasing_subsequence(nums)
    print(f"LIS length in {nums}: {lis_result}")
    
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    max_sum = adp.maximum_subarray_sum(nums)
    print(f"Maximum subarray sum: {max_sum}")
    
    # Matrix DP
    print("\n3. MATRIX DP PROBLEMS:")
    m, n = 3, 7
    paths = adp.unique_paths(m, n)
    print(f"Unique paths in {m}x{n} grid: {paths}")
    
    # Knapsack variations
    print("\n4. KNAPSACK PROBLEMS:")
    coins = [1, 3, 4]
    amount = 6
    min_coins = adp.coin_change(coins, amount)
    print(f"Minimum coins for amount {amount}: {min_coins}")
    
    nums = [1, 5, 11, 5]
    can_part = adp.can_partition(nums)
    print(f"Can partition {nums}: {can_part}")

if __name__ == "__main__":
    run_examples() 