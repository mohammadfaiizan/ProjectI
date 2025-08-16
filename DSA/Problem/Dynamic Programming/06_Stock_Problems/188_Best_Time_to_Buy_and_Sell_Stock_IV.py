"""
LeetCode 188: Best Time to Buy and Sell Stock IV
Difficulty: Hard
Category: Stock Problems - General K Transactions

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
Total profit is 4 + 3 = 7.

Constraints:
- 1 <= k <= 100
- 1 <= prices.length <= 1000
- 0 <= prices[i] <= 1000
"""

def max_profit_unlimited_transactions(prices):
    """
    UNLIMITED TRANSACTIONS (when k >= n/2):
    ======================================
    Use greedy approach when k is large enough.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if not prices:
        return 0
    
    total_profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            total_profit += prices[i] - prices[i-1]
    
    return total_profit


def max_profit_dp_2d(k, prices):
    """
    2D DP APPROACH:
    ==============
    dp[i][j] = max profit using at most i transactions by day j.
    
    Time Complexity: O(k*n^2) - naive implementation
    Space Complexity: O(k*n) - 2D DP table
    """
    n = len(prices)
    if n <= 1 or k == 0:
        return 0
    
    # If k is large enough, equivalent to unlimited transactions
    if k >= n // 2:
        return max_profit_unlimited_transactions(prices)
    
    # dp[i][j] = max profit using at most i transactions by day j
    dp = [[0] * n for _ in range(k + 1)]
    
    for i in range(1, k + 1):
        for j in range(1, n):
            # Option 1: Don't trade on day j
            dp[i][j] = dp[i][j-1]
            
            # Option 2: Sell on day j (find best buy day)
            for m in range(j):
                profit = prices[j] - prices[m] + dp[i-1][m]
                dp[i][j] = max(dp[i][j], profit)
    
    return dp[k][n-1]


def max_profit_dp_optimized(k, prices):
    """
    OPTIMIZED DP APPROACH:
    =====================
    Optimize the inner loop using max_diff technique.
    
    Time Complexity: O(k*n) - optimized inner loop
    Space Complexity: O(k*n) - 2D DP table
    """
    n = len(prices)
    if n <= 1 or k == 0:
        return 0
    
    if k >= n // 2:
        return max_profit_unlimited_transactions(prices)
    
    # dp[i][j] = max profit using at most i transactions by day j
    dp = [[0] * n for _ in range(k + 1)]
    
    for i in range(1, k + 1):
        max_diff = -prices[0]  # max(dp[i-1][m] - prices[m]) for m < j
        
        for j in range(1, n):
            # Don't trade on day j
            dp[i][j] = dp[i][j-1]
            
            # Sell on day j
            dp[i][j] = max(dp[i][j], prices[j] + max_diff)
            
            # Update max_diff for next iteration
            max_diff = max(max_diff, dp[i-1][j] - prices[j])
    
    return dp[k][n-1]


def max_profit_space_optimized(k, prices):
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Use O(k) space instead of O(k*n).
    
    Time Complexity: O(k*n) - optimized DP
    Space Complexity: O(k) - single row
    """
    n = len(prices)
    if n <= 1 or k == 0:
        return 0
    
    if k >= n // 2:
        return max_profit_unlimited_transactions(prices)
    
    # Use two arrays: buy[i] and sell[i] for transaction i
    buy = [-prices[0]] * k   # Max profit after buying in transaction i
    sell = [0] * k           # Max profit after selling in transaction i
    
    for price in prices[1:]:
        for i in range(k-1, -1, -1):  # Process in reverse to avoid dependency
            # Sell in transaction i
            sell[i] = max(sell[i], buy[i] + price)
            
            # Buy in transaction i (use profit from previous transaction)
            if i == 0:
                buy[i] = max(buy[i], -price)
            else:
                buy[i] = max(buy[i], sell[i-1] - price)
    
    return sell[k-1] if k > 0 else 0


def max_profit_state_machine(k, prices):
    """
    STATE MACHINE APPROACH:
    ======================
    Model each transaction as a buy-sell state pair.
    
    Time Complexity: O(k*n) - k states per day
    Space Complexity: O(k) - state arrays
    """
    n = len(prices)
    if n <= 1 or k == 0:
        return 0
    
    if k >= n // 2:
        return max_profit_unlimited_transactions(prices)
    
    # States for each transaction: [buy_state, sell_state]
    states = [[-prices[0], 0] for _ in range(k)]
    
    for price in prices[1:]:
        for i in range(k-1, -1, -1):  # Process in reverse
            # Update sell state for transaction i
            states[i][1] = max(states[i][1], states[i][0] + price)
            
            # Update buy state for transaction i
            if i == 0:
                states[i][0] = max(states[i][0], -price)
            else:
                states[i][0] = max(states[i][0], states[i-1][1] - price)
    
    return states[k-1][1] if k > 0 else 0


def max_profit_with_transactions(k, prices):
    """
    TRACK ACTUAL TRANSACTIONS:
    ==========================
    Return profit and list of optimal transactions.
    
    Time Complexity: O(k*n^2) - reconstruction phase
    Space Complexity: O(k*n) - DP table + transaction tracking
    """
    n = len(prices)
    if n <= 1 or k == 0:
        return 0, []
    
    if k >= n // 2:
        # Use greedy approach for unlimited transactions
        transactions = []
        profit = 0
        i = 0
        
        while i < n - 1:
            # Find valley
            while i < n - 1 and prices[i + 1] <= prices[i]:
                i += 1
            if i == n - 1:
                break
            buy_day = i
            
            # Find peak
            while i < n - 1 and prices[i + 1] >= prices[i]:
                i += 1
            sell_day = i
            
            trans_profit = prices[sell_day] - prices[buy_day]
            profit += trans_profit
            transactions.append((buy_day, sell_day, prices[buy_day], prices[sell_day], trans_profit))
        
        return profit, transactions
    
    # Use DP with transaction tracking
    dp = [[0] * n for _ in range(k + 1)]
    transactions_dp = [[[] for _ in range(n)] for _ in range(k + 1)]
    
    for i in range(1, k + 1):
        max_diff = -prices[0]
        best_buy_day = 0
        
        for j in range(1, n):
            # Don't trade on day j
            if dp[i][j-1] > dp[i][j]:
                dp[i][j] = dp[i][j-1]
                transactions_dp[i][j] = transactions_dp[i][j-1][:]
            
            # Sell on day j
            profit_if_sell = prices[j] + max_diff
            if profit_if_sell > dp[i][j]:
                dp[i][j] = profit_if_sell
                transactions_dp[i][j] = transactions_dp[i-1][best_buy_day][:]
                if prices[j] > prices[best_buy_day]:  # Profitable transaction
                    trans_profit = prices[j] - prices[best_buy_day]
                    transactions_dp[i][j].append((best_buy_day, j, prices[best_buy_day], prices[j], trans_profit))
            
            # Update max_diff and best_buy_day
            if dp[i-1][j] - prices[j] > max_diff:
                max_diff = dp[i-1][j] - prices[j]
                best_buy_day = j
    
    return dp[k][n-1], transactions_dp[k][n-1]


def max_profit_analysis(k, prices):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation for k transactions.
    """
    print(f"At Most {k} Transactions Stock Analysis:")
    print(f"Prices: {prices}")
    
    if not prices:
        print("No prices available.")
        return 0
    
    n = len(prices)
    print(f"Days: {list(range(n))}")
    
    if k >= n // 2:
        print(f"\nk={k} >= n/2={n//2}, using unlimited transactions approach")
        return max_profit_unlimited_transactions(prices)
    
    # State machine analysis
    print(f"\nState machine evolution (k={k}):")
    states = [[-prices[0], 0] for _ in range(k)]
    
    print(f"Day  Price", end="")
    for i in range(k):
        print(f"  Buy{i+1}  Sell{i+1}", end="")
    print()
    
    print(f"---  -----", end="")
    for i in range(k):
        print(f"  -----  ------", end="")
    print()
    
    print(f"  0  {prices[0]:5}", end="")
    for i in range(k):
        print(f"  {states[i][0]:5}  {states[i][1]:6}", end="")
    print()
    
    for day in range(1, len(prices)):
        price = prices[day]
        
        for i in range(k-1, -1, -1):
            # Update states
            states[i][1] = max(states[i][1], states[i][0] + price)
            if i == 0:
                states[i][0] = max(states[i][0], -price)
            else:
                states[i][0] = max(states[i][0], states[i-1][1] - price)
        
        print(f"{day:3}  {price:5}", end="")
        for i in range(k):
            print(f"  {states[i][0]:5}  {states[i][1]:6}", end="")
        print()
    
    final_profit = states[k-1][1] if k > 0 else 0
    print(f"\nFinal maximum profit: {final_profit}")
    
    # Show optimal transactions
    profit, transactions = max_profit_with_transactions(k, prices)
    print(f"\nOptimal transactions (total profit: {profit}):")
    
    if transactions:
        for i, (buy_day, sell_day, buy_price, sell_price, trans_profit) in enumerate(transactions):
            print(f"  {i+1}. Buy day {buy_day} (${buy_price}) → Sell day {sell_day} (${sell_price}) = ${trans_profit}")
        print(f"Used {len(transactions)} out of {k} allowed transactions")
    else:
        print(f"  No profitable transactions")
    
    return final_profit


def max_profit_variants():
    """
    K TRANSACTIONS VARIANTS:
    =======================
    Different scenarios and edge cases.
    """
    
    def max_profit_exactly_k(k, prices):
        """Require exactly k transactions"""
        n = len(prices)
        if n < 2*k or k == 0:  # Need at least 2k days for k transactions
            return 0
        
        # Modified DP that requires exactly k transactions
        dp = [[0] * n for _ in range(k + 1)]
        
        for i in range(1, k + 1):
            max_diff = -prices[0]
            for j in range(2*i-1, n):  # Need at least 2*i days for i transactions
                if j > 0:
                    dp[i][j] = max(dp[i][j-1], prices[j] + max_diff)
                    max_diff = max(max_diff, dp[i-1][j] - prices[j])
        
        return dp[k][n-1]
    
    def max_profit_with_fees(k, prices, fee):
        """K transactions with transaction fee"""
        n = len(prices)
        if n <= 1 or k == 0:
            return 0
        
        if k >= n // 2:
            # Unlimited with fee
            profit = 0
            for i in range(1, n):
                if prices[i] > prices[i-1]:
                    profit += prices[i] - prices[i-1]
            # Subtract fees for actual transactions
            transactions = 0
            i = 0
            while i < n - 1:
                while i < n - 1 and prices[i + 1] <= prices[i]:
                    i += 1
                if i == n - 1:
                    break
                while i < n - 1 and prices[i + 1] >= prices[i]:
                    i += 1
                transactions += 1
            return max(0, profit - transactions * fee)
        
        # DP with fees
        buy = [-prices[0]] * k
        sell = [0] * k
        
        for price in prices[1:]:
            for i in range(k-1, -1, -1):
                sell[i] = max(sell[i], buy[i] + price - fee)
                if i == 0:
                    buy[i] = max(buy[i], -price)
                else:
                    buy[i] = max(buy[i], sell[i-1] - price)
        
        return sell[k-1] if k > 0 else 0
    
    def max_profit_increasing_k(prices):
        """Show how profit changes with increasing k"""
        n = len(prices)
        results = []
        
        for k in range(1, min(10, n//2 + 3)):
            profit = max_profit_space_optimized(k, prices)
            results.append((k, profit))
        
        return results
    
    # Test variants
    test_cases = [
        (2, [2, 4, 1]),
        (2, [3, 2, 6, 5, 0, 3]),
        (3, [1, 2, 4, 2, 5, 7, 2, 4, 9, 0]),
        (1, [7, 1, 5, 3, 6, 4]),
        (5, [1, 2, 3, 4, 5])
    ]
    
    print("K Transactions Variants:")
    print("=" * 50)
    
    for k, prices in test_cases:
        print(f"\nk={k}, Prices: {prices}")
        
        at_most_k = max_profit_space_optimized(k, prices)
        print(f"At most {k} transactions: {at_most_k}")
        
        exactly_k = max_profit_exactly_k(k, prices)
        print(f"Exactly {k} transactions: {exactly_k}")
        
        with_fee = max_profit_with_fees(k, prices, 1)
        print(f"With $1 fee: {with_fee}")
        
        # Show increasing k
        if len(prices) <= 10:
            increasing = max_profit_increasing_k(prices)
            print(f"Increasing k: {increasing}")


# Test cases
def test_max_profit_iv():
    """Test all implementations with various inputs"""
    test_cases = [
        (2, [2, 4, 1], 2),
        (2, [3, 2, 6, 5, 0, 3], 7),
        (1, [1, 2, 3, 4, 5], 4),
        (2, [1, 2, 3, 4, 5], 4),
        (3, [1, 2, 3, 4, 5], 4),
        (0, [1, 2, 3], 0),
        (1, [7, 6, 4, 3, 1], 0),
        (2, [1], 0),
        (2, [], 0)
    ]
    
    print("Testing Best Time to Buy and Sell Stock IV Solutions:")
    print("=" * 70)
    
    for i, (k, prices, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"k={k}, Prices: {prices}")
        print(f"Expected: {expected}")
        
        # Test different approaches
        dp_2d = max_profit_dp_2d(k, prices)
        dp_optimized = max_profit_dp_optimized(k, prices)
        space_optimized = max_profit_space_optimized(k, prices)
        state_machine = max_profit_state_machine(k, prices)
        
        print(f"2D DP:            {dp_2d:>3} {'✓' if dp_2d == expected else '✗'}")
        print(f"Optimized DP:     {dp_optimized:>3} {'✓' if dp_optimized == expected else '✗'}")
        print(f"Space Optimized:  {space_optimized:>3} {'✓' if space_optimized == expected else '✗'}")
        print(f"State Machine:    {state_machine:>3} {'✓' if state_machine == expected else '✗'}")
        
        # Show transactions for small cases
        if len(prices) <= 8 and k <= 3:
            profit, transactions = max_profit_with_transactions(k, prices)
            print(f"Transactions: {len(transactions)} total, profit: {profit}")
            for j, (buy_day, sell_day, buy_price, sell_price, trans_profit) in enumerate(transactions):
                print(f"  {j+1}. Day {buy_day}→{sell_day}: ${buy_price}→${sell_price} = +${trans_profit}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    max_profit_analysis(2, [3, 2, 6, 5, 0, 3])
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    max_profit_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. K THRESHOLD: When k >= n/2, equivalent to unlimited")
    print("2. STATE MACHINE: k buy-sell state pairs")
    print("3. SPACE OPTIMIZATION: O(k) instead of O(k*n)")
    print("4. ORDER DEPENDENCY: Update states in reverse order")
    print("5. TRANSACTION LIMIT: Must choose k best opportunities")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Portfolio Management: Limited transaction strategies")
    print("• Algorithmic Trading: Constrained optimization")
    print("• Resource Allocation: k optimal investment windows")
    print("• Risk Management: Limiting exposure through caps")
    print("• Financial Planning: Multi-stage investment strategies")


if __name__ == "__main__":
    test_max_profit_iv()


"""
BEST TIME TO BUY AND SELL STOCK IV - GENERAL K-TRANSACTION OPTIMIZATION:
========================================================================

This is the most general form of the stock trading problem:
- At most k transactions allowed (buy-sell pairs)
- Must choose k best opportunities from all possibilities
- Demonstrates the full power of dynamic programming for constrained optimization
- Unifies all previous stock problems under one framework

KEY INSIGHTS:
============
1. **K THRESHOLD**: When k ≥ n/2, equivalent to unlimited transactions
2. **STATE EXPLOSION**: Need to track k different transaction states
3. **SPACE OPTIMIZATION**: Reduce from O(k×n) to O(k) space
4. **ORDER DEPENDENCY**: State updates must maintain transaction sequence
5. **GENERAL FRAMEWORK**: Encompasses all other stock trading problems

ALGORITHM APPROACHES:
====================

1. **Naive 2D DP**: O(k×n²) time, O(k×n) space
   - For each transaction and day, try all possible buy days
   - Clear but inefficient

2. **Optimized 2D DP**: O(k×n) time, O(k×n) space
   - Use max_diff technique to eliminate inner loop
   - Standard DP solution

3. **Space Optimized**: O(k×n) time, O(k) space
   - Use buy/sell arrays instead of full 2D table
   - Most practical solution

4. **State Machine**: O(k×n) time, O(k) space
   - Model as k buy-sell state pairs
   - Most intuitive for understanding

CORE ALGORITHM (SPACE OPTIMIZED):
================================
```python
if k >= n // 2:
    return unlimited_transactions(prices)  # Greedy approach

buy = [-prices[0]] * k   # Max profit after buying in transaction i
sell = [0] * k           # Max profit after selling in transaction i

for price in prices[1:]:
    for i in range(k-1, -1, -1):  # Reverse order crucial
        sell[i] = max(sell[i], buy[i] + price)
        if i == 0:
            buy[i] = max(buy[i], -price)
        else:
            buy[i] = max(buy[i], sell[i-1] - price)

return sell[k-1]
```

STATE MACHINE FORMULATION:
=========================
```
k Transaction States:
T1: buy1 → sell1
T2: buy2 → sell2
...
Tk: buyk → sellk

Transitions:
- buy[i] = max(buy[i], sell[i-1] - price)  // Buy in transaction i
- sell[i] = max(sell[i], buy[i] + price)   // Sell in transaction i

Dependencies:
- Transaction i depends on completion of transaction i-1
- Must update in reverse order to maintain dependencies
```

K THRESHOLD OPTIMIZATION:
========================
Critical optimization: when k ≥ n/2, use unlimited transactions:

**Reasoning**:
- Maximum possible transactions in n days: ⌊n/2⌋
- If k ≥ n/2, the constraint is not binding
- Use O(n) greedy algorithm instead of O(k×n) DP

```python
if k >= len(prices) // 2:
    # Unlimited transactions (greedy)
    return sum(max(0, prices[i] - prices[i-1]) for i in range(1, len(prices)))
```

DP RECURRENCE RELATIONS:
=======================
General k-transaction DP:
```
dp[i][j] = max profit using at most i transactions by day j

Base cases:
- dp[0][j] = 0  (no transactions allowed)
- dp[i][0] = 0  (only one day)

Recurrence:
dp[i][j] = max(
    dp[i][j-1],                              // Don't trade on day j
    max(prices[j] - prices[m] + dp[i-1][m])  // Sell on day j, bought on day m
)

Optimization with max_diff:
max_diff = max(dp[i-1][m] - prices[m]) for m < j

dp[i][j] = max(dp[i][j-1], prices[j] + max_diff)
```

SPACE OPTIMIZATION TECHNIQUE:
============================
Transform 2D DP to 1D:
```
Original: dp[i][j] and dp[i-1][j]
Optimized: buy[i] and sell[i] arrays

Mapping:
- buy[i] ≈ max profit state after buying in transaction i
- sell[i] ≈ max profit state after selling in transaction i

Update order crucial:
- Process transactions i from k-1 down to 0
- Ensures sell[i-1] is from previous iteration when updating buy[i]
```

COMPLEXITY ANALYSIS:
===================
| Approach         | Time    | Space | Best Use Case        |
|------------------|---------|-------|----------------------|
| Naive 2D DP      | O(k×n²) | O(k×n)| Educational          |
| Optimized 2D DP  | O(k×n)  | O(k×n)| Standard solution    |
| Space Optimized  | O(k×n)  | O(k)  | Memory constrained   |
| State Machine    | O(k×n)  | O(k)  | Intuitive reasoning  |

SPECIAL CASES:
=============
- **k = 0**: No transactions allowed → profit = 0
- **k = 1**: Single transaction → Stock Problem I
- **k = 2**: Two transactions → Stock Problem III  
- **k ≥ n/2**: Unlimited transactions → Stock Problem II
- **n ≤ 1**: Insufficient days → profit = 0

EDGE CASES:
==========
- **Empty prices**: Return 0
- **Single day**: Return 0 (can't trade)
- **Decreasing prices**: Return 0 (no profit possible)
- **Increasing prices**: May not need all k transactions
- **k > n/2**: Use unlimited transaction algorithm

RELATIONSHIP TO OTHER PROBLEMS:
==============================
This problem is the master template:
- **Stock I (121)**: k = 1
- **Stock II (122)**: k = ∞ (unlimited)
- **Stock III (123)**: k = 2
- **Stock with Cooldown (309)**: Add state for cooldown
- **Stock with Fee (714)**: Subtract fee from each transaction

EXTENSIONS AND VARIANTS:
=======================
- **Exactly k transactions**: Modify DP to require all k transactions
- **Transaction fees**: Subtract fee from each sell operation
- **Minimum gap**: Require minimum days between transactions
- **Volume limits**: Limited shares per transaction
- **Multi-stock**: Extend to multiple stocks

PRACTICAL CONSIDERATIONS:
========================
- **Transaction Costs**: Real trading has brokerage fees
- **Market Impact**: Large orders affect stock prices
- **Liquidity**: May not be able to buy/sell desired quantities
- **Tax Implications**: Short-term vs long-term capital gains
- **Risk Management**: Diversification vs concentration

MATHEMATICAL PROPERTIES:
========================
- **Monotonicity**: profit(k) ≤ profit(k+1) for same prices
- **Convergence**: profit(k) stabilizes at k = ⌊n/2⌋
- **Optimal Substructure**: Optimal k-transaction solution contains optimal (k-1)-transaction subsolution
- **Overlapping Subproblems**: Same states computed multiple times

ALGORITHM DESIGN INSIGHTS:
==========================
This problem showcases several key algorithmic techniques:
1. **Threshold Optimization**: Switch algorithms based on parameter size
2. **Space-Time Tradeoffs**: 2D→1D space reduction
3. **State Compression**: Represent complex states efficiently
4. **Order Dependencies**: Careful state update sequencing
5. **General→Specific**: Unify multiple related problems

This represents the pinnacle of stock trading optimization problems,
demonstrating how dynamic programming can solve complex constrained
optimization problems efficiently.
"""
