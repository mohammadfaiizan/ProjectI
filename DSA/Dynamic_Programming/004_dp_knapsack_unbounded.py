"""
Dynamic Programming - Unbounded Knapsack Pattern
This module implements the Unbounded Knapsack problem and its variants including
rod cutting, coin change problems, ribbon cutting, and integer break with optimizations.
"""

from typing import List, Dict, Tuple, Optional
import time
import math

# ==================== CLASSIC UNBOUNDED KNAPSACK ====================

class UnboundedKnapsack:
    """
    Unbounded Knapsack Problem Implementation
    
    Given items with weights and values, and a knapsack capacity,
    find the maximum value that can be obtained by selecting items
    such that total weight doesn't exceed capacity.
    
    Key difference from 0/1: Each item can be taken unlimited times.
    """
    
    def unbounded_knapsack_recursive(self, weights: List[int], values: List[int], 
                                   capacity: int, n: int) -> int:
        """
        Recursive solution (exponential time - for demonstration)
        
        Time Complexity: O(2^(capacity/min_weight))
        Space Complexity: O(capacity/min_weight) - recursion stack
        
        Args:
            weights: List of item weights
            values: List of item values
            capacity: Knapsack capacity
            n: Number of different item types
        
        Returns:
            Maximum value achievable
        """
        # Base case
        if capacity == 0 or n == 0:
            return 0
        
        # If current item's weight exceeds capacity, skip it
        if weights[n - 1] > capacity:
            return self.unbounded_knapsack_recursive(weights, values, capacity, n - 1)
        
        # Choose maximum of including or excluding current item type
        # Note: if included, we can include it again (unbounded)
        include = values[n - 1] + self.unbounded_knapsack_recursive(
            weights, values, capacity - weights[n - 1], n
        )
        exclude = self.unbounded_knapsack_recursive(weights, values, capacity, n - 1)
        
        return max(include, exclude)
    
    def unbounded_knapsack_memoization(self, weights: List[int], values: List[int], 
                                     capacity: int) -> int:
        """
        Memoization approach (top-down DP)
        
        Time Complexity: O(n * capacity)
        Space Complexity: O(n * capacity)
        
        Args:
            weights: List of item weights
            values: List of item values
            capacity: Knapsack capacity
        
        Returns:
            Maximum value achievable
        """
        n = len(weights)
        memo = {}
        
        def dp(remaining_capacity: int, item_index: int) -> int:
            # Base case
            if remaining_capacity == 0 or item_index == n:
                return 0
            
            # Check memo
            if (remaining_capacity, item_index) in memo:
                return memo[(remaining_capacity, item_index)]
            
            # Skip current item type
            result = dp(remaining_capacity, item_index + 1)
            
            # Include current item type if it fits
            if weights[item_index] <= remaining_capacity:
                include = values[item_index] + dp(
                    remaining_capacity - weights[item_index], item_index
                )
                result = max(result, include)
            
            memo[(remaining_capacity, item_index)] = result
            return result
        
        return dp(capacity, 0)
    
    def unbounded_knapsack_tabulation(self, weights: List[int], values: List[int], 
                                    capacity: int) -> int:
        """
        Tabulation approach (bottom-up DP)
        
        Time Complexity: O(n * capacity)
        Space Complexity: O(capacity)
        
        dp[w] = maximum value achievable with capacity w
        
        Args:
            weights: List of item weights
            values: List of item values
            capacity: Knapsack capacity
        
        Returns:
            Maximum value achievable
        """
        n = len(weights)
        dp = [0 for _ in range(capacity + 1)]
        
        # For each capacity
        for w in range(1, capacity + 1):
            # Try each item type
            for i in range(n):
                if weights[i] <= w:
                    dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
        
        return dp[capacity]
    
    def unbounded_knapsack_with_items(self, weights: List[int], values: List[int], 
                                    capacity: int) -> Tuple[int, List[int]]:
        """
        Return maximum value and count of each item type used
        
        Args:
            weights: List of item weights
            values: List of item values
            capacity: Knapsack capacity
        
        Returns:
            Tuple of (max_value, item_counts)
        """
        n = len(weights)
        dp = [0 for _ in range(capacity + 1)]
        parent = [-1 for _ in range(capacity + 1)]
        
        # Fill DP table and track which item was used
        for w in range(1, capacity + 1):
            for i in range(n):
                if weights[i] <= w:
                    if dp[w - weights[i]] + values[i] > dp[w]:
                        dp[w] = dp[w - weights[i]] + values[i]
                        parent[w] = i
        
        # Backtrack to find item counts
        item_counts = [0] * n
        current_capacity = capacity
        
        while current_capacity > 0 and parent[current_capacity] != -1:
            item_index = parent[current_capacity]
            item_counts[item_index] += 1
            current_capacity -= weights[item_index]
        
        return dp[capacity], item_counts

# ==================== ROD CUTTING PROBLEM ====================

class RodCutting:
    """
    Rod Cutting Problem
    
    Given a rod of length n and prices for pieces of different lengths,
    find the maximum revenue obtainable by cutting the rod and selling the pieces.
    """
    
    def rod_cutting_recursive(self, prices: List[int], length: int) -> int:
        """
        Recursive solution for rod cutting
        
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        
        Args:
            prices: prices[i] is price for rod of length i+1
            length: Length of rod to cut
        
        Returns:
            Maximum revenue obtainable
        """
        if length == 0:
            return 0
        
        max_revenue = 0
        
        # Try all possible first cuts
        for i in range(1, min(length, len(prices)) + 1):
            revenue = prices[i - 1] + self.rod_cutting_recursive(prices, length - i)
            max_revenue = max(max_revenue, revenue)
        
        return max_revenue
    
    def rod_cutting_memoization(self, prices: List[int], length: int) -> int:
        """
        Memoization approach for rod cutting
        
        Time Complexity: O(n²)
        Space Complexity: O(n)
        
        Args:
            prices: prices[i] is price for rod of length i+1
            length: Length of rod to cut
        
        Returns:
            Maximum revenue obtainable
        """
        memo = {}
        
        def dp(remaining_length: int) -> int:
            if remaining_length == 0:
                return 0
            
            if remaining_length in memo:
                return memo[remaining_length]
            
            max_revenue = 0
            for i in range(1, min(remaining_length, len(prices)) + 1):
                revenue = prices[i - 1] + dp(remaining_length - i)
                max_revenue = max(max_revenue, revenue)
            
            memo[remaining_length] = max_revenue
            return max_revenue
        
        return dp(length)
    
    def rod_cutting_tabulation(self, prices: List[int], length: int) -> int:
        """
        Tabulation approach for rod cutting
        
        Time Complexity: O(n²)
        Space Complexity: O(n)
        
        Args:
            prices: prices[i] is price for rod of length i+1
            length: Length of rod to cut
        
        Returns:
            Maximum revenue obtainable
        """
        dp = [0 for _ in range(length + 1)]
        
        for i in range(1, length + 1):
            for j in range(1, min(i, len(prices)) + 1):
                dp[i] = max(dp[i], prices[j - 1] + dp[i - j])
        
        return dp[length]
    
    def rod_cutting_with_cuts(self, prices: List[int], length: int) -> Tuple[int, List[int]]:
        """
        Return maximum revenue and the actual cuts made
        
        Args:
            prices: prices[i] is price for rod of length i+1
            length: Length of rod to cut
        
        Returns:
            Tuple of (max_revenue, cuts_list)
        """
        dp = [0 for _ in range(length + 1)]
        cuts = [0 for _ in range(length + 1)]
        
        for i in range(1, length + 1):
            for j in range(1, min(i, len(prices)) + 1):
                if prices[j - 1] + dp[i - j] > dp[i]:
                    dp[i] = prices[j - 1] + dp[i - j]
                    cuts[i] = j
        
        # Reconstruct the cuts
        cut_list = []
        remaining = length
        while remaining > 0:
            cut_list.append(cuts[remaining])
            remaining -= cuts[remaining]
        
        return dp[length], cut_list
    
    def rod_cutting_with_cost(self, prices: List[int], costs: List[int], length: int) -> int:
        """
        Rod cutting where each cut has an associated cost
        
        Args:
            prices: prices[i] is price for rod of length i+1
            costs: costs[i] is cost for making cut of length i+1
            length: Length of rod to cut
        
        Returns:
            Maximum profit (revenue - costs)
        """
        dp = [0 for _ in range(length + 1)]
        
        for i in range(1, length + 1):
            for j in range(1, min(i, len(prices)) + 1):
                profit = prices[j - 1] - costs[j - 1] + dp[i - j]
                dp[i] = max(dp[i], profit)
        
        return dp[length]

# ==================== COIN CHANGE PROBLEMS ====================

class CoinChange:
    """
    Coin Change Problems
    
    Various problems related to making change with given coin denominations.
    """
    
    def coin_change_min_coins(self, coins: List[int], amount: int) -> int:
        """
        Find minimum number of coins to make given amount
        
        LeetCode 322 - Coin Change
        
        Time Complexity: O(amount * len(coins))
        Space Complexity: O(amount)
        
        Args:
            coins: Available coin denominations
            amount: Target amount to make
        
        Returns:
            Minimum number of coins needed, -1 if impossible
        """
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1
    
    def coin_change_count_ways(self, coins: List[int], amount: int) -> int:
        """
        Count number of ways to make given amount
        
        LeetCode 518 - Coin Change 2
        
        Time Complexity: O(amount * len(coins))
        Space Complexity: O(amount)
        
        Args:
            coins: Available coin denominations
            amount: Target amount to make
        
        Returns:
            Number of ways to make the amount
        """
        dp = [0] * (amount + 1)
        dp[0] = 1  # One way to make amount 0
        
        # Important: iterate coins first to avoid counting permutations
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]
        
        return dp[amount]
    
    def coin_change_count_ways_permutations(self, coins: List[int], amount: int) -> int:
        """
        Count number of ways including different orders (permutations)
        
        Args:
            coins: Available coin denominations
            amount: Target amount to make
        
        Returns:
            Number of ways including permutations
        """
        dp = [0] * (amount + 1)
        dp[0] = 1
        
        # Iterate amount first to count permutations
        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i:
                    dp[i] += dp[i - coin]
        
        return dp[amount]
    
    def coin_change_with_exact_coins(self, coins: List[int], amount: int, 
                                   exact_count: int) -> int:
        """
        Make amount using exactly 'exact_count' coins
        
        Args:
            coins: Available coin denominations
            amount: Target amount to make
            exact_count: Exact number of coins to use
        
        Returns:
            Number of ways to make amount with exact count, 0 if impossible
        """
        # dp[i][j] = ways to make amount i using exactly j coins
        dp = [[0 for _ in range(exact_count + 1)] for _ in range(amount + 1)]
        dp[0][0] = 1  # One way to make 0 with 0 coins
        
        for coin in coins:
            for i in range(coin, amount + 1):
                for j in range(1, exact_count + 1):
                    dp[i][j] += dp[i - coin][j - 1]
        
        return dp[amount][exact_count]
    
    def coin_change_with_limits(self, coins: List[int], limits: List[int], 
                               amount: int) -> int:
        """
        Coin change where each coin type has a usage limit
        
        Args:
            coins: Available coin denominations
            limits: limits[i] is max count for coins[i]
            amount: Target amount to make
        
        Returns:
            Minimum coins needed, -1 if impossible
        """
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for i in range(len(coins)):
            coin = coins[i]
            limit = limits[i]
            
            # Use multiple knapsack approach
            for j in range(amount, -1, -1):
                if dp[j] == float('inf'):
                    continue
                
                for k in range(1, limit + 1):
                    if j + k * coin <= amount:
                        dp[j + k * coin] = min(dp[j + k * coin], dp[j] + k)
        
        return dp[amount] if dp[amount] != float('inf') else -1
    
    def coin_change_with_denomination_cost(self, coins: List[int], costs: List[int], 
                                         amount: int) -> int:
        """
        Coin change where each denomination has different costs
        
        Args:
            coins: Available coin denominations
            costs: costs[i] is cost for using coins[i]
            amount: Target amount to make
        
        Returns:
            Minimum total cost to make amount
        """
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for i in range(1, amount + 1):
            for j in range(len(coins)):
                if coins[j] <= i:
                    dp[i] = min(dp[i], dp[i - coins[j]] + costs[j])
        
        return dp[amount] if dp[amount] != float('inf') else -1

# ==================== MAXIMUM RIBBON CUT ====================

class MaximumRibbonCut:
    """
    Maximum Ribbon Cut Problem
    
    Given a ribbon of length n and sizes of ribbon pieces,
    find the maximum number of pieces that can be cut.
    """
    
    def max_ribbon_cuts(self, ribbon_length: int, cuts: List[int]) -> int:
        """
        Find maximum number of ribbon pieces
        
        Time Complexity: O(ribbon_length * len(cuts))
        Space Complexity: O(ribbon_length)
        
        Args:
            ribbon_length: Total length of ribbon
            cuts: Available cut sizes
        
        Returns:
            Maximum number of pieces, -1 if impossible to cut completely
        """
        dp = [-1] * (ribbon_length + 1)
        dp[0] = 0  # 0 cuts needed for length 0
        
        for i in range(1, ribbon_length + 1):
            for cut in cuts:
                if cut <= i and dp[i - cut] != -1:
                    dp[i] = max(dp[i], dp[i - cut] + 1)
        
        return dp[ribbon_length]
    
    def max_ribbon_cuts_with_min_pieces(self, ribbon_length: int, cuts: List[int], 
                                       min_pieces: int) -> int:
        """
        Maximum ribbon cuts with constraint on minimum pieces
        
        Args:
            ribbon_length: Total length of ribbon
            cuts: Available cut sizes
            min_pieces: Minimum number of pieces required
        
        Returns:
            Maximum number of pieces >= min_pieces, -1 if impossible
        """
        dp = [-1] * (ribbon_length + 1)
        dp[0] = 0
        
        for i in range(1, ribbon_length + 1):
            for cut in cuts:
                if cut <= i and dp[i - cut] != -1:
                    dp[i] = max(dp[i], dp[i - cut] + 1)
        
        if dp[ribbon_length] >= min_pieces:
            return dp[ribbon_length]
        return -1
    
    def max_ribbon_cuts_with_values(self, ribbon_length: int, cuts: List[int], 
                                   values: List[int]) -> int:
        """
        Maximum value from ribbon cuts (each cut size has different value)
        
        Args:
            ribbon_length: Total length of ribbon
            cuts: Available cut sizes
            values: values[i] is value for cuts[i]
        
        Returns:
            Maximum value obtainable
        """
        dp = [0] * (ribbon_length + 1)
        
        for i in range(1, ribbon_length + 1):
            for j in range(len(cuts)):
                if cuts[j] <= i:
                    dp[i] = max(dp[i], dp[i - cuts[j]] + values[j])
        
        return dp[ribbon_length]
    
    def ribbon_cuts_ways(self, ribbon_length: int, cuts: List[int]) -> int:
        """
        Count number of ways to cut ribbon completely
        
        Args:
            ribbon_length: Total length of ribbon
            cuts: Available cut sizes
        
        Returns:
            Number of ways to cut ribbon
        """
        dp = [0] * (ribbon_length + 1)
        dp[0] = 1  # One way to cut length 0
        
        for cut in cuts:
            for i in range(cut, ribbon_length + 1):
                dp[i] += dp[i - cut]
        
        return dp[ribbon_length]

# ==================== INTEGER BREAK ====================

class IntegerBreak:
    """
    Integer Break Problem
    
    Given an integer n, break it into sum of positive integers
    to maximize their product.
    """
    
    def integer_break_basic(self, n: int) -> int:
        """
        Basic integer break to maximize product
        
        LeetCode 343 - Integer Break
        
        Time Complexity: O(n²)
        Space Complexity: O(n)
        
        Args:
            n: Integer to break
        
        Returns:
            Maximum product obtainable
        """
        if n <= 2:
            return 1
        
        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 1
        
        for i in range(3, n + 1):
            for j in range(1, i):
                # Either don't break j further, or use dp[j] (break j further)
                dp[i] = max(dp[i], max(j, dp[j]) * max(i - j, dp[i - j]))
        
        return dp[n]
    
    def integer_break_optimized(self, n: int) -> int:
        """
        Optimized version using mathematical insight
        
        Key insight: To maximize product, use as many 3s as possible,
        with remainder handled specially.
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if n <= 2:
            return 1
        if n == 3:
            return 2
        if n == 4:
            return 4
        
        # Use as many 3s as possible
        if n % 3 == 0:
            return 3 ** (n // 3)
        elif n % 3 == 1:
            # Replace one 3 with two 2s (3+1 = 2+2, but 2*2 > 3*1)
            return 3 ** (n // 3 - 1) * 4
        else:  # n % 3 == 2
            return 3 ** (n // 3) * 2
    
    def integer_break_with_constraint(self, n: int, max_parts: int) -> int:
        """
        Integer break with constraint on maximum number of parts
        
        Args:
            n: Integer to break
            max_parts: Maximum number of parts allowed
        
        Returns:
            Maximum product with at most max_parts parts
        """
        # dp[i][j] = max product for integer i using at most j parts
        dp = [[0 for _ in range(max_parts + 1)] for _ in range(n + 1)]
        
        # Base cases
        for j in range(max_parts + 1):
            dp[0][j] = 1  # Product of empty set is 1
            dp[1][j] = 1  # Can only break 1 into 1
        
        for i in range(2, n + 1):
            for j in range(1, max_parts + 1):
                dp[i][j] = 1  # Don't break at all (use i itself)
                
                # Try breaking into k and (i-k)
                for k in range(1, i):
                    if j > 1:  # Only if we can use more parts
                        dp[i][j] = max(dp[i][j], k * dp[i - k][j - 1])
        
        return dp[n][max_parts]
    
    def integer_break_min_parts(self, n: int, target_product: int) -> int:
        """
        Find minimum parts needed to achieve at least target product
        
        Args:
            n: Integer to break
            target_product: Target product to achieve
        
        Returns:
            Minimum parts needed, -1 if impossible
        """
        if target_product == 1:
            return 1
        
        # dp[i] = minimum parts to achieve product >= target_product using integer i
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            # Try not breaking
            if i >= target_product:
                dp[i] = 1
            
            # Try breaking into j and (i-j)
            for j in range(1, i):
                # Calculate product achievable
                for parts_j in range(1, i):
                    for parts_remaining in range(1, i):
                        if parts_j + parts_remaining <= i:
                            # This is a simplified version - full implementation would be more complex
                            pass
        
        return dp[n] if dp[n] != float('inf') else -1

# ==================== ADVANCED UNBOUNDED KNAPSACK VARIANTS ====================

class AdvancedUnboundedKnapsack:
    """
    Advanced variations of unbounded knapsack problems
    """
    
    def unbounded_knapsack_with_order(self, weights: List[int], values: List[int], 
                                    capacity: int) -> int:
        """
        Unbounded knapsack where order of selection matters
        
        This counts permutations rather than combinations.
        """
        dp = [0] * (capacity + 1)
        
        for w in range(1, capacity + 1):
            for i in range(len(weights)):
                if weights[i] <= w:
                    dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
        
        return dp[capacity]
    
    def unbounded_knapsack_min_items(self, weights: List[int], capacity: int) -> int:
        """
        Find minimum number of items to exactly fill capacity
        
        Args:
            weights: Item weights (values are all 1)
            capacity: Target capacity to fill exactly
        
        Returns:
            Minimum items needed, -1 if impossible
        """
        dp = [float('inf')] * (capacity + 1)
        dp[0] = 0
        
        for w in range(1, capacity + 1):
            for weight in weights:
                if weight <= w:
                    dp[w] = min(dp[w], dp[w - weight] + 1)
        
        return dp[capacity] if dp[capacity] != float('inf') else -1
    
    def unbounded_knapsack_combinations_target(self, nums: List[int], target: int) -> int:
        """
        Count combinations that sum to target (order doesn't matter)
        
        Time Complexity: O(target * len(nums))
        Space Complexity: O(target)
        """
        dp = [0] * (target + 1)
        dp[0] = 1
        
        for num in nums:
            for i in range(num, target + 1):
                dp[i] += dp[i - num]
        
        return dp[target]
    
    def unbounded_knapsack_permutations_target(self, nums: List[int], target: int) -> int:
        """
        Count permutations that sum to target (order matters)
        
        Time Complexity: O(target * len(nums))
        Space Complexity: O(target)
        """
        dp = [0] * (target + 1)
        dp[0] = 1
        
        for i in range(1, target + 1):
            for num in nums:
                if num <= i:
                    dp[i] += dp[i - num]
        
        return dp[i]

# ==================== PERFORMANCE ANALYSIS ====================

def performance_comparison():
    """Compare performance of different unbounded knapsack implementations"""
    print("=== Unbounded Knapsack Performance Comparison ===\n")
    
    # Test data
    weights = [1, 3, 4]
    values = [1, 4, 5]
    capacity = 8
    
    knapsack = UnboundedKnapsack()
    
    print(f"Test case: weights={weights}, values={values}, capacity={capacity}")
    
    # Memoization
    start_time = time.time()
    memo_result = knapsack.unbounded_knapsack_memoization(weights, values, capacity)
    memo_time = time.time() - start_time
    
    # Tabulation
    start_time = time.time()
    tab_result = knapsack.unbounded_knapsack_tabulation(weights, values, capacity)
    tab_time = time.time() - start_time
    
    print(f"Memoization: {memo_result} ({memo_time:.6f}s)")
    print(f"Tabulation:  {tab_result} ({tab_time:.6f}s)")
    print(f"Results match: {memo_result == tab_result}")

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Unbounded Knapsack Pattern Demo ===\n")
    
    # Classic Unbounded Knapsack
    print("1. Classic Unbounded Knapsack:")
    knapsack = UnboundedKnapsack()
    
    weights = [1, 3, 4]
    values = [1, 4, 5]
    capacity = 8
    
    max_value = knapsack.unbounded_knapsack_tabulation(weights, values, capacity)
    max_value_with_items, item_counts = knapsack.unbounded_knapsack_with_items(weights, values, capacity)
    
    print(f"  Items: weights={weights}, values={values}")
    print(f"  Capacity: {capacity}")
    print(f"  Maximum value: {max_value}")
    print(f"  Item counts: {item_counts}")
    
    total_weight = sum(weights[i] * item_counts[i] for i in range(len(weights)))
    total_value = sum(values[i] * item_counts[i] for i in range(len(values)))
    print(f"  Total weight used: {total_weight}")
    print(f"  Total value achieved: {total_value}")
    print()
    
    # Rod Cutting
    print("2. Rod Cutting Problem:")
    rod_cutting = RodCutting()
    
    prices = [1, 5, 8, 9, 10, 17, 17, 20]
    length = 8
    
    max_revenue = rod_cutting.rod_cutting_tabulation(prices, length)
    max_revenue_with_cuts, cuts = rod_cutting.rod_cutting_with_cuts(prices, length)
    
    print(f"  Prices for lengths 1-8: {prices}")
    print(f"  Rod length: {length}")
    print(f"  Maximum revenue: {max_revenue}")
    print(f"  Optimal cuts: {cuts}")
    print(f"  Verification: sum of cuts = {sum(cuts)}")
    
    # Rod cutting with costs
    costs = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    max_profit = rod_cutting.rod_cutting_with_cost(prices, costs, length)
    print(f"  With cutting costs {costs}: max profit = {max_profit}")
    print()
    
    # Coin Change
    print("3. Coin Change Problems:")
    coin_change = CoinChange()
    
    coins = [1, 3, 4]
    amount = 6
    
    min_coins = coin_change.coin_change_min_coins(coins, amount)
    count_ways = coin_change.coin_change_count_ways(coins, amount)
    count_permutations = coin_change.coin_change_count_ways_permutations(coins, amount)
    
    print(f"  Coins: {coins}, Amount: {amount}")
    print(f"  Minimum coins needed: {min_coins}")
    print(f"  Number of ways (combinations): {count_ways}")
    print(f"  Number of ways (permutations): {count_permutations}")
    
    # Exact coins
    exact_count = 3
    exact_ways = coin_change.coin_change_with_exact_coins(coins, amount, exact_count)
    print(f"  Ways to make {amount} with exactly {exact_count} coins: {exact_ways}")
    
    # With limits
    limits = [2, 1, 2]  # max 2 of coin 1, max 1 of coin 3, max 2 of coin 4
    min_with_limits = coin_change.coin_change_with_limits(coins, limits, amount)
    print(f"  Min coins with limits {limits}: {min_with_limits}")
    print()
    
    # Maximum Ribbon Cut
    print("4. Maximum Ribbon Cut:")
    ribbon = MaximumRibbonCut()
    
    ribbon_length = 13
    cuts = [2, 3, 5]
    
    max_pieces = ribbon.max_ribbon_cuts(ribbon_length, cuts)
    ways_to_cut = ribbon.ribbon_cuts_ways(ribbon_length, cuts)
    
    print(f"  Ribbon length: {ribbon_length}, Cut sizes: {cuts}")
    print(f"  Maximum pieces: {max_pieces}")
    print(f"  Number of ways to cut: {ways_to_cut}")
    
    # With values
    values = [1, 2, 3]  # values for cut sizes [2, 3, 5]
    max_value = ribbon.max_ribbon_cuts_with_values(ribbon_length, cuts, values)
    print(f"  Maximum value with cut values {values}: {max_value}")
    print()
    
    # Integer Break
    print("5. Integer Break:")
    int_break = IntegerBreak()
    
    test_numbers = [8, 10, 12]
    
    for n in test_numbers:
        basic_result = int_break.integer_break_basic(n)
        optimized_result = int_break.integer_break_optimized(n)
        
        print(f"  Integer {n}:")
        print(f"    Basic DP: {basic_result}")
        print(f"    Optimized: {optimized_result}")
        print(f"    Results match: {basic_result == optimized_result}")
    
    # With constraints
    n = 10
    max_parts = 3
    constrained_result = int_break.integer_break_with_constraint(n, max_parts)
    print(f"  Integer {n} with max {max_parts} parts: {constrained_result}")
    print()
    
    # Advanced Unbounded Knapsack
    print("6. Advanced Unbounded Knapsack:")
    advanced = AdvancedUnboundedKnapsack()
    
    weights_adv = [2, 3, 5]
    target = 8
    
    min_items = advanced.unbounded_knapsack_min_items(weights_adv, target)
    combinations = advanced.unbounded_knapsack_combinations_target(weights_adv, target)
    permutations = advanced.unbounded_knapsack_permutations_target(weights_adv, target)
    
    print(f"  Weights: {weights_adv}, Target: {target}")
    print(f"  Minimum items to reach target: {min_items}")
    print(f"  Combinations that sum to target: {combinations}")
    print(f"  Permutations that sum to target: {permutations}")
    print()
    
    # Performance comparison
    performance_comparison()
    print()
    
    # Pattern Recognition Guide
    print("=== Unbounded Knapsack Pattern Recognition ===")
    print("Identify Unbounded Knapsack patterns when:")
    print("  1. Items can be used unlimited times (not 0/1 choice)")
    print("  2. Goal is to optimize some objective (max/min)")
    print("  3. Subject to capacity or target constraints")
    print("  4. Order of selection may or may not matter")
    
    print("\nCommon variants:")
    print("  1. Rod Cutting: maximize revenue by cutting rod optimally")
    print("  2. Coin Change: minimize coins or count ways to make amount")
    print("  3. Ribbon Cut: maximize pieces or ways to cut completely")
    print("  4. Integer Break: maximize product of integer parts")
    
    print("\nKey differences from 0/1 Knapsack:")
    print("  1. Can reuse same item type multiple times")
    print("  2. Inner loop goes forward (not backward) in 1D DP")
    print("  3. When including item, stay at same item index")
    print("  4. Often related to change-making problems")
    
    print("\nOptimization techniques:")
    print("  1. 2D → 1D space optimization")
    print("  2. Mathematical insights (like integer break)")
    print("  3. Order matters: permutations vs combinations")
    print("  4. Early termination when target reached")
    
    print("\nReal-world applications:")
    print("  1. Manufacturing: cutting materials optimally")
    print("  2. Finance: making change, currency exchange")
    print("  3. Resource allocation: unlimited resource types")
    print("  4. Combinatorics: counting ways to form sums")
    print("  5. Optimization: breaking problems into subproblems")
    
    print("\n=== Demo Complete ===") 