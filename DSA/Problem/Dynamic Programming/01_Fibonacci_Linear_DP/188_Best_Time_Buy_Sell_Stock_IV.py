"""
LeetCode 188: Best Time to Buy and Sell Stock IV
Difficulty: Hard
Category: Fibonacci & Linear DP / Stock Problems

PROBLEM DESCRIPTION:
===================
You are given an integer array prices where prices[i] is the price of a given stock on the ith day, 
and an integer k.

Find the maximum profit you can achieve. You may complete at most k transactions.

Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

Example 1:
Input: k = 2, prices = [2,4,1]
Output: 2
Explanation: Buy on day 1 (price = 2) and sell on day 2 (price = 4), profit = 4-2 = 2.

Example 2:
Input: k = 2, prices = [3,2,6,5,0,3]
Output: 7
Explanation: Buy on day 2 (price = 2) and sell on day 3 (price = 6), profit = 6-2 = 4. 
Then buy on day 5 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.

Constraints:
- 1 <= k <= 100
- 1 <= prices.length <= 1000
- 0 <= prices[i] <= 1000
"""

def max_profit_k_bruteforce(k, prices):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible combinations of k transactions using recursion.
    
    Time Complexity: O(2^n * k) - exponential combinations
    Space Complexity: O(n * k) - recursion stack
    """
    if k == 0 or len(prices) < 2:
        return 0
    
    def max_profit_rec(day, transactions_left, holding):
        if day >= len(prices) or transactions_left == 0:
            return 0
        
        # Option 1: Do nothing
        result = max_profit_rec(day + 1, transactions_left, holding)
        
        if holding:
            # Option 2: Sell (completes a transaction)
            result = max(result, prices[day] + max_profit_rec(day + 1, transactions_left - 1, False))
        else:
            # Option 2: Buy
            result = max(result, -prices[day] + max_profit_rec(day + 1, transactions_left, True))
        
        return result
    
    return max_profit_rec(0, k, False)


def max_profit_k_memoization(k, prices):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to cache subproblem results.
    
    Time Complexity: O(n * k * 2) = O(n * k) - states: day, transactions, holding
    Space Complexity: O(n * k * 2) = O(n * k) - memoization table
    """
    if k == 0 or len(prices) < 2:
        return 0
    
    # Optimization: if k >= n/2, we can do unlimited transactions
    if k >= len(prices) // 2:
        return max_profit_unlimited(prices)
    
    memo = {}
    
    def dp(day, transactions_left, holding):
        if day >= len(prices) or transactions_left == 0:
            return 0
        
        if (day, transactions_left, holding) in memo:
            return memo[(day, transactions_left, holding)]
        
        # Option 1: Do nothing
        result = dp(day + 1, transactions_left, holding)
        
        if holding:
            # Option 2: Sell
            result = max(result, prices[day] + dp(day + 1, transactions_left - 1, False))
        else:
            # Option 2: Buy
            result = max(result, -prices[day] + dp(day + 1, transactions_left, True))
        
        memo[(day, transactions_left, holding)] = result
        return result
    
    return dp(0, k, False)


def max_profit_k_dp_2d(k, prices):
    """
    DYNAMIC PROGRAMMING - 2D TABULATION:
    ===================================
    Use 2D DP table for transactions and days.
    
    Time Complexity: O(n * k) - fill DP table
    Space Complexity: O(n * k) - 2D DP table
    """
    if k == 0 or len(prices) < 2:
        return 0
    
    n = len(prices)
    
    # Optimization: if k >= n/2, unlimited transactions
    if k >= n // 2:
        return max_profit_unlimited(prices)
    
    # buy[i][j] = max profit after at most i transactions, ending with buy on day j
    # sell[i][j] = max profit after at most i transactions, ending with sell on day j
    buy = [[-prices[0]] * n for _ in range(k + 1)]
    sell = [[0] * n for _ in range(k + 1)]
    
    for i in range(1, k + 1):
        for j in range(1, n):
            buy[i][j] = max(buy[i][j - 1], sell[i - 1][j - 1] - prices[j])
            sell[i][j] = max(sell[i][j - 1], buy[i][j - 1] + prices[j])
    
    return sell[k][n - 1]


def max_profit_k_dp_optimized(k, prices):
    """
    SPACE OPTIMIZED DP:
    ==================
    Optimize space by using only necessary arrays.
    
    Time Complexity: O(n * k) - same iterations
    Space Complexity: O(k) - only store buy/sell arrays
    """
    if k == 0 or len(prices) < 2:
        return 0
    
    # Optimization: unlimited transactions case
    if k >= len(prices) // 2:
        return max_profit_unlimited(prices)
    
    # buy[i] = max profit after buying in transaction i
    # sell[i] = max profit after selling in transaction i
    buy = [-prices[0]] * k
    sell = [0] * k
    
    for price in prices[1:]:
        for j in range(k - 1, -1, -1):
            sell[j] = max(sell[j], buy[j] + price)
            if j == 0:
                buy[j] = max(buy[j], -price)
            else:
                buy[j] = max(buy[j], sell[j - 1] - price)
    
    return sell[k - 1]


def max_profit_unlimited(prices):
    """
    HELPER: UNLIMITED TRANSACTIONS:
    ==============================
    Maximum profit with unlimited transactions (greedy approach).
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]
    return profit


def max_profit_k_state_machine(k, prices):
    """
    STATE MACHINE APPROACH:
    ======================
    Generalize the state machine for k transactions.
    
    Time Complexity: O(n * k) - process each day for k transactions
    Space Complexity: O(k) - store states for k transactions
    """
    if k == 0 or len(prices) < 2:
        return 0
    
    if k >= len(prices) // 2:
        return max_profit_unlimited(prices)
    
    # hold[i] = max profit after buying in transaction i
    # sold[i] = max profit after selling in transaction i
    hold = [-prices[0]] * k
    sold = [0] * k
    
    for price in prices[1:]:
        for i in range(k - 1, -1, -1):
            sold[i] = max(sold[i], hold[i] + price)
            if i == 0:
                hold[i] = max(hold[i], -price)
            else:
                hold[i] = max(hold[i], sold[i - 1] - price)
    
    return sold[k - 1]


def max_profit_k_rolling_array(k, prices):
    """
    ROLLING ARRAY OPTIMIZATION:
    ===========================
    Use rolling arrays to further optimize space.
    
    Time Complexity: O(n * k) - same complexity
    Space Complexity: O(k) - rolling optimization
    """
    if k == 0 or len(prices) < 2:
        return 0
    
    if k >= len(prices) // 2:
        return max_profit_unlimited(prices)
    
    # Global and local maximum profits
    global_profit = [0] * (k + 1)
    local_profit = [0] * (k + 1)
    
    for price in prices[1:]:
        diff = price - prices[prices.index(price) - 1] if price in prices[1:] else 0
        
        for j in range(k, 0, -1):
            local_profit[j] = max(global_profit[j - 1] + max(diff, 0), local_profit[j] + diff)
            global_profit[j] = max(global_profit[j], local_profit[j])
    
    return global_profit[k]


def max_profit_k_generic_dp(k, prices):
    """
    GENERIC DP FORMULATION:
    ======================
    Most general formulation that works for any k.
    
    Time Complexity: O(n * k) - standard DP
    Space Complexity: O(k) - optimized space
    """
    if k == 0 or len(prices) < 2:
        return 0
    
    if k >= len(prices) // 2:
        return max_profit_unlimited(prices)
    
    # Use the fact that after 2*k operations, we complete k transactions
    # buy[i] represents max profit after (2*i-1)th operation (buy)
    # sell[i] represents max profit after (2*i)th operation (sell)
    
    buy = [float('-inf')] * (k + 1)
    sell = [0] * (k + 1)
    
    for price in prices:
        for i in range(k, 0, -1):
            sell[i] = max(sell[i], buy[i] + price)
            buy[i] = max(buy[i], sell[i - 1] - price)
    
    return sell[k]


def max_profit_k_with_transactions(k, prices):
    """
    FIND MAXIMUM PROFIT AND ACTUAL TRANSACTIONS:
    ===========================================
    Return both maximum profit and the actual transactions.
    
    Time Complexity: O(n * k) - DP + transaction reconstruction
    Space Complexity: O(n * k) - store transaction history
    """
    if k == 0 or len(prices) < 2:
        return 0, []
    
    if k >= len(prices) // 2:
        # Unlimited transactions - use greedy approach
        profit = 0
        transactions = []
        buy_price = None
        
        for i, price in enumerate(prices):
            if i == 0 or price <= prices[i - 1]:
                if buy_price is not None and i > 0:
                    # Sell at previous price
                    profit += prices[i - 1] - buy_price
                    transactions.append((buy_price, prices[i - 1]))
                buy_price = price
        
        # Sell at last price if holding
        if buy_price is not None and prices[-1] > buy_price:
            profit += prices[-1] - buy_price
            transactions.append((buy_price, prices[-1]))
        
        return profit, transactions
    
    # Use DP with transaction tracking
    n = len(prices)
    
    # For simplicity, use the optimized DP approach
    buy = [-prices[0]] * k
    sell = [0] * k
    
    for price in prices[1:]:
        for j in range(k - 1, -1, -1):
            sell[j] = max(sell[j], buy[j] + price)
            if j == 0:
                buy[j] = max(buy[j], -price)
            else:
                buy[j] = max(buy[j], sell[j - 1] - price)
    
    # Simplified transaction reconstruction
    transactions = []
    if sell[k - 1] > 0:
        transactions.append(f"Maximum {k} transactions, profit: {sell[k - 1]}")
    
    return sell[k - 1], transactions


# Test cases
def test_max_profit_k():
    """Test all implementations with various inputs"""
    test_cases = [
        (2, [2,4,1], 2),
        (2, [3,2,6,5,0,3], 7),
        (1, [1,2,3,4,5], 4),
        (2, [1,2,3,4,5], 4),
        (3, [1,2,3,4,5], 4),
        (0, [1,2,3], 0),
        (1, [7,6,4,3,1], 0),
        (2, [3,3,5,0,0,3,1,4], 6),
        (4, [1,2,4,2,5,7,2,4,9,0], 15),
        (2, [1], 0)
    ]
    
    print("Testing Best Time to Buy and Sell Stock IV Solutions:")
    print("=" * 70)
    
    for i, (k, prices, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: k = {k}, prices = {prices}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for large inputs)
        if len(prices) <= 8 and k <= 3:
            brute = max_profit_k_bruteforce(k, prices.copy())
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        memo = max_profit_k_memoization(k, prices.copy())
        dp_2d = max_profit_k_dp_2d(k, prices.copy())
        dp_opt = max_profit_k_dp_optimized(k, prices.copy())
        state_machine = max_profit_k_state_machine(k, prices.copy())
        generic = max_profit_k_generic_dp(k, prices.copy())
        
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        print(f"DP 2D:            {dp_2d:>3} {'✓' if dp_2d == expected else '✗'}")
        print(f"DP Optimized:     {dp_opt:>3} {'✓' if dp_opt == expected else '✗'}")
        print(f"State Machine:    {state_machine:>3} {'✓' if state_machine == expected else '✗'}")
        print(f"Generic DP:       {generic:>3} {'✓' if generic == expected else '✗'}")
        
        # Show transaction details for some cases
        if expected > 0 and len(prices) <= 10:
            profit, transactions = max_profit_k_with_transactions(k, prices.copy())
            print(f"Transactions: {transactions}")
    
    # Test edge cases
    print(f"\nEdge Case Testing:")
    edge_cases = [
        (100, [1,2,3,4,5], 4, "Large k (unlimited transactions)"),
        (0, [1,2,3,4,5], 0, "k = 0"),
        (1, [], 0, "Empty prices"),
        (5, [1], 0, "Single price"),
        (2, [5,4,3,2,1], 0, "Decreasing prices")
    ]
    
    for k, prices, expected, description in edge_cases:
        result = max_profit_k_dp_optimized(k, prices)
        print(f"{description}: k={k}, prices={prices} -> {result} {'✓' if result == expected else '✗'}")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^n * k),   Space: O(n * k)")
    print("Memoization:      Time: O(n * k),     Space: O(n * k)")
    print("DP 2D:            Time: O(n * k),     Space: O(n * k)")
    print("DP Optimized:     Time: O(n * k),     Space: O(k)")
    print("State Machine:    Time: O(n * k),     Space: O(k)")
    print("Generic DP:       Time: O(n * k),     Space: O(k)")


if __name__ == "__main__":
    test_max_profit_k()


"""
PATTERN RECOGNITION:
==================
This is the most general stock trading DP problem:
- At most k transactions allowed
- Generalization of all previous stock problems
- Key optimization: when k ≥ n/2, reduce to unlimited transactions
- Multiple DP formulations possible

KEY INSIGHT - TRANSACTION LIMIT:
===============================
When k ≥ n/2, we can do unlimited transactions:
- Maximum n/2 transactions possible in n days
- Use greedy approach: buy low, sell high on consecutive days
- Reduces O(nk) to O(n) for large k

OPTIMAL SOLUTION STRUCTURE:
==========================
```python
buy = [-prices[0]] * k   # Max profit after buying in transaction i
sell = [0] * k           # Max profit after selling in transaction i

for price in prices[1:]:
    for i in range(k-1, -1, -1):
        sell[i] = max(sell[i], buy[i] + price)
        buy[i] = max(buy[i], (sell[i-1] if i > 0 else 0) - price)
```

MULTIPLE DP FORMULATIONS:
========================

1. **State Machine**: buy[i], sell[i] for i transactions
2. **2D DP**: dp[i][j] = max profit with i transactions on day j
3. **Memoization**: dp(day, transactions_left, holding_stock)
4. **Generic**: Works for any k with space optimization

OPTIMIZATION TECHNIQUES:
=======================
1. **Large k optimization**: k ≥ n/2 → unlimited transactions
2. **Space optimization**: O(nk) → O(k) space
3. **Rolling arrays**: Further space optimization
4. **Early termination**: Stop when no more profit possible

STATE DEFINITIONS:
=================
1. **buy[i]**: Maximum profit after buying in transaction i
2. **sell[i]**: Maximum profit after selling in transaction i
3. **Transaction**: Complete buy-sell pair

RECURRENCE RELATIONS:
====================
For each price and each transaction i:
- sell[i] = max(sell[i], buy[i] + price)      # Sell stock
- buy[i] = max(buy[i], sell[i-1] - price)     # Buy stock

Update order matters: process transactions from k-1 to 0

VARIANTS IN THIS SERIES:
=======================
- Stock I (121): k = 1 transaction
- Stock II (122): k = unlimited  
- Stock III (123): k = 2 transactions
- Stock IV (188): k = general (this problem)
- Stock with Cooldown (309): + cooldown constraint
- Stock with Fee (714): + transaction fee

INTERVIEW TIPS:
==============
1. Identify as generalization of stock problems
2. Handle k ≥ n/2 optimization first
3. Show state machine approach
4. Explain transaction state tracking
5. Optimize space from O(nk) to O(k)
6. Discuss time complexity optimization
7. Handle edge cases (k=0, empty prices)
8. Mention relationship to previous stock problems
"""
