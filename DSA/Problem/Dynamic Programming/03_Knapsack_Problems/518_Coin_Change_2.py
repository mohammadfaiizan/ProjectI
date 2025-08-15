"""
LeetCode 518: Coin Change 2
Difficulty: Medium
Category: Knapsack Problems (Unbounded Knapsack - Count Ways)

PROBLEM DESCRIPTION:
===================
You are given an integer array coins representing coins of different denominations and an integer 
amount representing a total amount of money.

Return the number of combinations that make up that amount. If that amount of money cannot be made 
up by any combination of the coins, return 0.

You may assume that you have an infinite number of each kind of coin.

The answer is guaranteed to fit into a signed 32-bit integer.

Example 1:
Input: amount = 5, coins = [1,2,5]
Output: 4
Explanation: there are four ways to make up the amount:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1

Example 2:
Input: amount = 3, coins = [2]
Output: 0
Explanation: the amount of 3 cannot be made up just with coins of 2.

Example 3:
Input: amount = 10, coins = [10]
Output: 1

Constraints:
- 1 <= coins.length <= 300
- 1 <= coins[i] <= 5000
- All the values of coins are unique.
- 0 <= amount <= 5000
"""

def change_bruteforce(amount, coins):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible combinations using recursion.
    
    Time Complexity: O(amount^len(coins)) - exponential combinations
    Space Complexity: O(amount) - recursion stack depth
    """
    def count_ways(remaining, coin_index):
        if remaining == 0:
            return 1
        if remaining < 0 or coin_index >= len(coins):
            return 0
        
        # Two choices: use current coin or skip to next coin
        use_coin = count_ways(remaining - coins[coin_index], coin_index)  # Can reuse same coin
        skip_coin = count_ways(remaining, coin_index + 1)
        
        return use_coin + skip_coin
    
    return count_ways(amount, 0)


def change_memoization(amount, coins):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to cache subproblem results.
    
    Time Complexity: O(amount * len(coins)) - states: remaining, coin_index
    Space Complexity: O(amount * len(coins)) - memoization table
    """
    memo = {}
    
    def count_ways(remaining, coin_index):
        if remaining == 0:
            return 1
        if remaining < 0 or coin_index >= len(coins):
            return 0
        
        if (remaining, coin_index) in memo:
            return memo[(remaining, coin_index)]
        
        # Use current coin (can reuse) or skip to next coin
        use_coin = count_ways(remaining - coins[coin_index], coin_index)
        skip_coin = count_ways(remaining, coin_index + 1)
        
        result = use_coin + skip_coin
        memo[(remaining, coin_index)] = result
        return result
    
    return count_ways(amount, 0)


def change_dp_2d(amount, coins):
    """
    2D DYNAMIC PROGRAMMING:
    =======================
    Use 2D DP table for coins and amounts.
    
    Time Complexity: O(len(coins) * amount) - fill DP table
    Space Complexity: O(len(coins) * amount) - 2D DP table
    """
    n = len(coins)
    
    # dp[i][j] = ways to make amount j using first i coins
    dp = [[0] * (amount + 1) for _ in range(n + 1)]
    
    # Base case: one way to make amount 0 (use no coins)
    for i in range(n + 1):
        dp[i][0] = 1
    
    for i in range(1, n + 1):
        for j in range(1, amount + 1):
            # Don't use current coin
            dp[i][j] = dp[i - 1][j]
            
            # Use current coin if possible
            if j >= coins[i - 1]:
                dp[i][j] += dp[i][j - coins[i - 1]]  # Can reuse same coin
    
    return dp[n][amount]


def change_dp_1d(amount, coins):
    """
    1D DYNAMIC PROGRAMMING (SPACE OPTIMIZED):
    =========================================
    Use 1D DP array, process coins one by one.
    
    Time Complexity: O(len(coins) * amount) - same iterations
    Space Complexity: O(amount) - 1D DP array
    """
    # dp[i] = number of ways to make amount i
    dp = [0] * (amount + 1)
    dp[0] = 1  # One way to make amount 0
    
    # Process each coin
    for coin in coins:
        # Update dp array for current coin
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]


def change_optimized_order(amount, coins):
    """
    OPTIMIZED WITH COIN ORDER:
    =========================
    Sort coins and add early termination optimizations.
    
    Time Complexity: O(len(coins) * amount) - worst case, often better
    Space Complexity: O(amount) - 1D DP array
    """
    # Sort coins for potential optimizations
    coins.sort()
    
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        # Early termination: if coin > amount, skip remaining coins
        if coin > amount:
            break
        
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]


def change_with_combinations(amount, coins):
    """
    FIND ACTUAL COMBINATIONS:
    ========================
    Return both count and actual combinations.
    
    Time Complexity: O(amount^len(coins)) - generate all combinations
    Space Complexity: O(amount^len(coins)) - store combinations
    """
    def find_combinations(remaining, coin_index, current_combination):
        if remaining == 0:
            combinations.append(current_combination[:])
            return
        if remaining < 0 or coin_index >= len(coins):
            return
        
        # Use current coin
        current_combination.append(coins[coin_index])
        find_combinations(remaining - coins[coin_index], coin_index, current_combination)
        current_combination.pop()
        
        # Skip to next coin
        find_combinations(remaining, coin_index + 1, current_combination)
    
    combinations = []
    find_combinations(amount, 0, [])
    return len(combinations), combinations


def change_iterative_generation(amount, coins):
    """
    ITERATIVE COMBINATION GENERATION:
    ================================
    Generate combinations iteratively using BFS-like approach.
    
    Time Complexity: O(amount * len(coins)) - DP computation
    Space Complexity: O(amount) - storage optimization
    """
    if amount == 0:
        return 1
    
    # Use set to track reachable amounts at each step
    current_ways = {0: 1}  # amount -> number of ways
    
    for coin in coins:
        new_ways = current_ways.copy()
        
        for amt, ways in current_ways.items():
            new_amt = amt + coin
            
            while new_amt <= amount:
                new_ways[new_amt] = new_ways.get(new_amt, 0) + ways
                new_amt += coin
        
        current_ways = new_ways
    
    return current_ways.get(amount, 0)


def change_mathematical_analysis(amount, coins):
    """
    MATHEMATICAL ANALYSIS:
    =====================
    Analyze the problem with mathematical insights.
    
    Time Complexity: O(len(coins) * amount) - DP computation
    Space Complexity: O(amount) - DP array
    """
    # This is a classic "compositions with repetition" problem
    # We want to find number of ways to write amount as sum of coins
    # where order doesn't matter (combinations, not permutations)
    
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    # Critical: process coins in outer loop to avoid permutations
    # This ensures we count combinations, not permutations
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
            
            # Optional: track which coins contribute most
            if i == amount and dp[i] > 0:
                print(f"Coin {coin} contributes to {dp[i]} ways")
    
    return dp[amount]


def change_comparison_with_permutations(amount, coins):
    """
    COMPARISON: COMBINATIONS VS PERMUTATIONS:
    ========================================
    Show difference between combination and permutation counting.
    
    Time Complexity: O(len(coins) * amount) - both computations
    Space Complexity: O(amount) - DP arrays
    """
    # COMBINATIONS (this problem): order doesn't matter
    # Process coins in outer loop
    combinations_dp = [0] * (amount + 1)
    combinations_dp[0] = 1
    
    for coin in coins:
        for i in range(coin, amount + 1):
            combinations_dp[i] += combinations_dp[i - coin]
    
    # PERMUTATIONS: order matters (like Combination Sum IV)
    # Process amounts in outer loop
    permutations_dp = [0] * (amount + 1)
    permutations_dp[0] = 1
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                permutations_dp[i] += permutations_dp[i - coin]
    
    print(f"Amount: {amount}, Coins: {coins}")
    print(f"Combinations (order doesn't matter): {combinations_dp[amount]}")
    print(f"Permutations (order matters): {permutations_dp[amount]}")
    
    return combinations_dp[amount]


# Test cases
def test_change():
    """Test all implementations with various inputs"""
    test_cases = [
        (5, [1,2,5], 4),
        (3, [2], 0),
        (10, [10], 1),
        (4, [1,2,3], 4),
        (0, [1,2], 1),
        (1, [1], 1),
        (2, [1], 1),
        (3, [1,2], 2),
        (4, [1,2], 3),
        (5, [1,2], 3),
        (6, [1,3,4], 4),
        (500, [3,5,7,8,9,10,11], 35502874)
    ]
    
    print("Testing Coin Change 2 Solutions:")
    print("=" * 70)
    
    for i, (amount, coins, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: amount = {amount}, coins = {coins}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for large inputs)
        if amount <= 10 and len(coins) <= 3:
            brute = change_bruteforce(amount, coins.copy())
            print(f"Brute Force:      {brute:>8} {'✓' if brute == expected else '✗'}")
        
        memo = change_memoization(amount, coins.copy())
        dp_2d = change_dp_2d(amount, coins.copy())
        dp_1d = change_dp_1d(amount, coins.copy())
        optimized = change_optimized_order(amount, coins.copy())
        iterative = change_iterative_generation(amount, coins.copy())
        
        print(f"Memoization:      {memo:>8} {'✓' if memo == expected else '✗'}")
        print(f"2D DP:            {dp_2d:>8} {'✓' if dp_2d == expected else '✗'}")
        print(f"1D DP:            {dp_1d:>8} {'✓' if dp_1d == expected else '✗'}")
        print(f"Optimized:        {optimized:>8} {'✓' if optimized == expected else '✗'}")
        print(f"Iterative:        {iterative:>8} {'✓' if iterative == expected else '✗'}")
        
        # Show actual combinations for small cases
        if expected > 0 and expected <= 10 and amount <= 10:
            count, combinations = change_with_combinations(amount, coins.copy())
            print(f"Combinations: {combinations}")
    
    # Show comparison between combinations and permutations
    print(f"\n" + "=" * 70)
    print("COMBINATIONS vs PERMUTATIONS:")
    
    test_comparison = [
        (4, [1,2]),
        (3, [1,2]),
        (4, [1,2,3])
    ]
    
    for amount, coins in test_comparison:
        print(f"\n{'-' * 40}")
        change_comparison_with_permutations(amount, coins)
    
    print("\n" + "=" * 70)
    print("Key Insight:")
    print("Coin Change 2 (518): Combinations - for coin in coins: for amount")
    print("Combination Sum IV (377): Permutations - for amount: for coin in coins")
    print("Loop order determines whether we count combinations or permutations!")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(amount^len(coins)), Space: O(amount)")
    print("Memoization:      Time: O(amount*len(coins)), Space: O(amount*len(coins))")
    print("2D DP:            Time: O(amount*len(coins)), Space: O(amount*len(coins))")
    print("1D DP:            Time: O(amount*len(coins)), Space: O(amount)")
    print("Optimized:        Time: O(amount*len(coins)), Space: O(amount)")
    print("Iterative:        Time: O(amount*len(coins)), Space: O(amount)")


if __name__ == "__main__":
    test_change()


"""
PATTERN RECOGNITION:
==================
This is an Unbounded Knapsack counting problem:
- Unlimited use of each coin (unbounded)
- Count number of ways to reach target (not minimize/maximize)
- Order doesn't matter: combinations, not permutations
- Key insight: loop order determines combinations vs permutations

KEY INSIGHT - COMBINATIONS vs PERMUTATIONS:
==========================================
**CRITICAL LOOP ORDER:**

COMBINATIONS (this problem):
```python
for coin in coins:
    for amount in range(coin, target + 1):
        dp[amount] += dp[amount - coin]
```

PERMUTATIONS (Combination Sum IV):
```python
for amount in range(1, target + 1):
    for coin in coins:
        dp[amount] += dp[amount - coin]
```

The outer loop determines what we're iterating over first!

STATE DEFINITION:
================
dp[i] = number of ways to make amount i using available coins

RECURRENCE RELATION:
===================
For each coin c and amount i ≥ c:
dp[i] += dp[i - c]

Base case: dp[0] = 1 (one way to make amount 0)

ALGORITHM PROGRESSION:
=====================
1. **Brute Force**: O(amount^coins) - try all combinations
2. **Memoization**: O(amount×coins) - cache subproblems
3. **2D DP**: O(amount×coins) time, O(amount×coins) space
4. **1D DP**: O(amount×coins) time, O(amount) space - optimal
5. **Optimized**: Add sorting and early termination

WHY LOOP ORDER MATTERS:
======================
- **Coins outer**: Ensures each coin considered once per amount
- **Amount outer**: Allows multiple coins for same amount (permutations)

Example with amount=3, coins=[1,2]:
- Combinations: {1,1,1}, {1,2} → 2 ways
- Permutations: {1,1,1}, {1,2}, {2,1} → 3 ways

MATHEMATICAL INSIGHT:
====================
This is a "compositions with restricted parts" problem:
- Find number of ways to write n as sum of given numbers
- Parts can be repeated (unbounded)
- Order doesn't matter (combinations)

VARIANTS TO PRACTICE:
====================
- Coin Change (322) - minimize coins used
- Combination Sum IV (377) - permutations (order matters)
- Perfect Squares (279) - special case with square numbers
- Partition Equal Subset Sum (416) - 0/1 knapsack variant

INTERVIEW TIPS:
==============
1. Clarify: combinations or permutations?
2. Identify as unbounded knapsack counting
3. Show brute force recursive approach first
4. Optimize with memoization
5. **Critical**: Explain loop order for combinations
6. Space optimize from 2D to 1D
7. Compare with Combination Sum IV (permutations)
8. Handle edge cases (amount=0, empty coins)
"""
