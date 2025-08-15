"""
LeetCode 322: Coin Change
Difficulty: Medium
Category: Fibonacci & Linear DP (Unbounded Knapsack variant)

PROBLEM DESCRIPTION:
===================
You are given an integer array coins representing coins of different denominations and an 
integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount 
of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

Example 1:
Input: coins = [1,3,4], amount = 6
Output: 2
Explanation: 6 = 3 + 3

Example 2:
Input: coins = [2], amount = 3
Output: -1

Example 3:
Input: coins = [1], amount = 0
Output: 0

Constraints:
- 1 <= coins.length <= 12
- 1 <= coins[i] <= 2^31 - 1
- 0 <= amount <= 10^4
"""

def coin_change_bruteforce(coins, amount):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible combinations using recursion.
    For each amount, try using each coin and recursively solve for remaining amount.
    
    Time Complexity: O(amount^coins) - exponential due to overlapping subproblems
    Space Complexity: O(amount) - recursion stack depth
    """
    def min_coins(remaining):
        if remaining == 0:
            return 0
        if remaining < 0:
            return float('inf')
        
        min_count = float('inf')
        for coin in coins:
            result = min_coins(remaining - coin)
            if result != float('inf'):
                min_count = min(min_count, result + 1)
        
        return min_count
    
    result = min_coins(amount)
    return result if result != float('inf') else -1


def coin_change_memoization(coins, amount):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to avoid recalculating same subproblems.
    
    Time Complexity: O(amount * len(coins)) - each subproblem calculated once
    Space Complexity: O(amount) - memoization table + recursion stack
    """
    memo = {}
    
    def min_coins(remaining):
        if remaining == 0:
            return 0
        if remaining < 0:
            return float('inf')
        
        if remaining in memo:
            return memo[remaining]
        
        min_count = float('inf')
        for coin in coins:
            result = min_coins(remaining - coin)
            if result != float('inf'):
                min_count = min(min_count, result + 1)
        
        memo[remaining] = min_count
        return min_count
    
    result = min_coins(amount)
    return result if result != float('inf') else -1


def coin_change_tabulation(coins, amount):
    """
    DYNAMIC PROGRAMMING - BOTTOM UP (TABULATION):
    =============================================
    Build solution from bottom up using iteration.
    dp[i] = minimum coins needed to make amount i
    
    Time Complexity: O(amount * len(coins)) - nested loops
    Space Complexity: O(amount) - DP table
    """
    # Initialize DP array with amount + 1 (impossible value)
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # Base case: 0 coins needed for amount 0
    
    # For each amount from 1 to target amount
    for i in range(1, amount + 1):
        # Try each coin
        for coin in coins:
            if coin <= i:  # Can use this coin
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1


def coin_change_bfs(coins, amount):
    """
    BFS APPROACH:
    ============
    Use BFS to find minimum number of coins.
    Each level represents using one more coin.
    
    Time Complexity: O(amount * len(coins)) - similar to DP
    Space Complexity: O(amount) - queue and visited set
    """
    if amount == 0:
        return 0
    
    from collections import deque
    
    queue = deque([amount])
    visited = {amount}
    level = 0
    
    while queue:
        level += 1
        size = len(queue)
        
        for _ in range(size):
            current_amount = queue.popleft()
            
            for coin in coins:
                next_amount = current_amount - coin
                
                if next_amount == 0:
                    return level
                
                if next_amount > 0 and next_amount not in visited:
                    visited.add(next_amount)
                    queue.append(next_amount)
    
    return -1


def coin_change_optimized_dp(coins, amount):
    """
    OPTIMIZED DP WITH EARLY TERMINATION:
    ===================================
    Optimized version with early termination and better initialization.
    
    Time Complexity: O(amount * len(coins)) - worst case
    Space Complexity: O(amount) - DP table
    """
    if amount == 0:
        return 0
    
    # Sort coins in descending order for potential early termination
    coins.sort(reverse=True)
    
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin > i:
                continue  # Coin too large
            if dp[i - coin] != float('inf'):
                dp[i] = min(dp[i], dp[i - coin] + 1)
                # Early termination: if we found a solution with 1 coin, it's optimal
                if dp[i] == 1:
                    break
    
    return dp[amount] if dp[amount] != float('inf') else -1


def coin_change_dfs_pruning(coins, amount):
    """
    DFS WITH PRUNING:
    ================
    Use DFS with pruning to find optimal solution.
    
    Time Complexity: O(amount^len(coins)) - worst case, but pruning helps
    Space Complexity: O(amount) - recursion stack
    """
    coins.sort(reverse=True)  # Try larger coins first
    min_coins = [float('inf')]
    
    def dfs(remaining, coin_count, start_idx):
        if remaining == 0:
            min_coins[0] = min(min_coins[0], coin_count)
            return
        
        # Pruning: if current path already uses too many coins
        if coin_count >= min_coins[0]:
            return
        
        for i in range(start_idx, len(coins)):
            coin = coins[i]
            if coin > remaining:
                continue
            
            # Pruning: if even using largest coin we need too many coins
            max_coins_with_current = remaining // coin
            if coin_count + max_coins_with_current >= min_coins[0]:
                continue
            
            # Try using this coin multiple times
            k = remaining // coin
            for j in range(k, 0, -1):
                dfs(remaining - coin * j, coin_count + j, i + 1)
    
    dfs(amount, 0, 0)
    return min_coins[0] if min_coins[0] != float('inf') else -1


# Test cases
def test_coin_change():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1, 3, 4], 6, 2),
        ([2], 3, -1),
        ([1], 0, 0),
        ([1, 2, 5], 11, 3),
        ([2, 5, 10, 1], 27, 4),
        ([1, 3, 4], 6, 2),
        ([1, 5, 10, 25], 30, 2),
        ([186, 419, 83, 408], 6249, 20)
    ]
    
    print("Testing Coin Change Solutions:")
    print("=" * 70)
    
    for i, (coins, amount, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: coins = {coins}, amount = {amount}")
        print(f"Expected: {expected}")
        
        # Test all approaches (skip brute force for large inputs)
        if amount <= 20:
            brute = coin_change_bruteforce(coins.copy(), amount)
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        memo = coin_change_memoization(coins.copy(), amount)
        tab = coin_change_tabulation(coins.copy(), amount)
        bfs = coin_change_bfs(coins.copy(), amount)
        opt_dp = coin_change_optimized_dp(coins.copy(), amount)
        dfs_prune = coin_change_dfs_pruning(coins.copy(), amount)
        
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        print(f"Tabulation:       {tab:>3} {'✓' if tab == expected else '✗'}")
        print(f"BFS:              {bfs:>3} {'✓' if bfs == expected else '✗'}")
        print(f"Optimized DP:     {opt_dp:>3} {'✓' if opt_dp == expected else '✗'}")
        print(f"DFS with Pruning: {dfs_prune:>3} {'✓' if dfs_prune == expected else '✗'}")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(amount^coins), Space: O(amount)")
    print("Memoization:      Time: O(amount*coins),  Space: O(amount)")
    print("Tabulation:       Time: O(amount*coins),  Space: O(amount)")
    print("BFS:              Time: O(amount*coins),  Space: O(amount)")
    print("Optimized DP:     Time: O(amount*coins),  Space: O(amount)")
    print("DFS with Pruning: Time: O(amount^coins),  Space: O(amount) - but pruning helps")


if __name__ == "__main__":
    test_coin_change()


"""
PATTERN RECOGNITION:
==================
This is an Unbounded Knapsack variant:
- Unlimited supply of each coin (item)
- Minimize number of coins (items) used
- dp[amount] = min coins needed to make that amount

KEY INSIGHTS:
============
1. For each amount, try using each coin and solve for remaining amount
2. Take minimum across all coin choices
3. This is similar to Fibonacci but with multiple choices at each step
4. Can be solved with both DP and BFS approaches

STATE DEFINITION:
================
dp[i] = minimum number of coins needed to make amount i

RECURRENCE RELATION:
===================
dp[i] = min(dp[i], dp[i - coin] + 1) for all coins where coin <= i
Base case: dp[0] = 0

VARIANTS TO PRACTICE:
====================
- Coin Change 2 (518) - count number of ways
- Combination Sum IV (377) - count combinations
- Perfect Squares (279) - special case with square numbers
- Minimum Cost For Tickets (983) - similar optimization problem

OPTIMIZATION TECHNIQUES:
=======================
1. Sort coins for early termination
2. Use BFS for potentially better average case
3. DFS with pruning for large search spaces
4. Greedy approach works only for specific coin systems

INTERVIEW TIPS:
==============
1. Identify this as an optimization problem (minimize coins)
2. Recognize unlimited usage of coins (unbounded knapsack)
3. Start with recursive solution, then add memoization
4. Discuss BFS as alternative approach
5. Mention greedy approach limitations
6. Handle edge cases (amount = 0, impossible amounts)
"""
