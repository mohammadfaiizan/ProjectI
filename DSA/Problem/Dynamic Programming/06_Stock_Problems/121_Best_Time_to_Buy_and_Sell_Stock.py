"""
LeetCode 121: Best Time to Buy and Sell Stock
Difficulty: Easy
Category: Stock Problems - Foundation

PROBLEM DESCRIPTION:
===================
You are given an array prices where prices[i] is the price of a given stock on the ith day.
You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

Example 1:
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.

Example 2:
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are completed, so the max profit = 0.

Constraints:
- 1 <= prices.length <= 10^5
- 0 <= prices[i] <= 10^4
"""

def max_profit_brute_force(prices):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible buy-sell combinations.
    
    Time Complexity: O(n^2) - nested loops
    Space Complexity: O(1) - constant space
    """
    max_profit = 0
    
    for i in range(len(prices)):
        for j in range(i + 1, len(prices)):
            profit = prices[j] - prices[i]
            max_profit = max(max_profit, profit)
    
    return max_profit


def max_profit_min_price_tracking(prices):
    """
    MIN PRICE TRACKING APPROACH:
    ============================
    Track minimum price seen so far and maximum profit.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if not prices:
        return 0
    
    min_price = prices[0]
    max_profit = 0
    
    for price in prices:
        # Update minimum price if current price is lower
        min_price = min(min_price, price)
        
        # Calculate profit if we sell at current price
        current_profit = price - min_price
        max_profit = max(max_profit, current_profit)
    
    return max_profit


def max_profit_kadane_variant(prices):
    """
    KADANE'S ALGORITHM VARIANT:
    ===========================
    Transform to maximum subarray problem.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if len(prices) <= 1:
        return 0
    
    # Transform to difference array
    max_ending_here = 0
    max_so_far = 0
    
    for i in range(1, len(prices)):
        # Daily profit/loss
        daily_change = prices[i] - prices[i-1]
        
        # Kadane's algorithm
        max_ending_here = max(daily_change, max_ending_here + daily_change)
        max_so_far = max(max_so_far, max_ending_here)
    
    return max_so_far


def max_profit_dp_state_machine(prices):
    """
    DP STATE MACHINE APPROACH:
    ==========================
    Model as state machine with buy/sell states.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if not prices:
        return 0
    
    # States: held[i] = max profit when holding stock on day i
    #         sold[i] = max profit when not holding stock on day i
    
    held = -prices[0]  # Must buy stock, so negative price
    sold = 0           # No stock, no profit initially
    
    for i in range(1, len(prices)):
        # Update states
        new_held = max(held, -prices[i])  # Keep holding or buy today
        new_sold = max(sold, held + prices[i])  # Keep not holding or sell today
        
        held = new_held
        sold = new_sold
    
    return sold  # We want to end without holding stock


def max_profit_dp_explicit(prices):
    """
    EXPLICIT DP APPROACH:
    ====================
    Use DP arrays to track states explicitly.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(n) - DP arrays
    """
    n = len(prices)
    if n <= 1:
        return 0
    
    # dp[i][0] = max profit on day i when not holding stock
    # dp[i][1] = max profit on day i when holding stock
    dp = [[0, 0] for _ in range(n)]
    
    # Base case
    dp[0][0] = 0          # Not holding stock on day 0
    dp[0][1] = -prices[0] # Holding stock on day 0 (bought it)
    
    for i in range(1, n):
        # Not holding stock: either didn't have it or sold it today
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
        
        # Holding stock: either had it or bought it today
        # Note: can only buy once, so previous state must be 0
        dp[i][1] = max(dp[i-1][1], -prices[i])
    
    return dp[n-1][0]  # Best profit when not holding stock at end


def max_profit_with_details(prices):
    """
    FIND OPTIMAL BUY/SELL DAYS:
    ===========================
    Return maximum profit and the optimal buy/sell days.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if not prices:
        return 0, -1, -1
    
    min_price = prices[0]
    min_day = 0
    max_profit = 0
    buy_day = -1
    sell_day = -1
    
    for i, price in enumerate(prices):
        if price < min_price:
            min_price = price
            min_day = i
        
        current_profit = price - min_price
        if current_profit > max_profit:
            max_profit = current_profit
            buy_day = min_day
            sell_day = i
    
    return max_profit, buy_day, sell_day


def max_profit_analysis(prices):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation and profit analysis.
    """
    print(f"Stock Price Analysis:")
    print(f"Prices: {prices}")
    
    if not prices:
        print("No prices available.")
        return 0
    
    n = len(prices)
    print(f"Days: {list(range(n))}")
    
    # Track computation step by step
    min_price = prices[0]
    min_day = 0
    max_profit = 0
    transactions = []
    
    print(f"\nStep-by-step analysis:")
    print(f"Day  Price  MinPrice  MinDay  Profit  MaxProfit  Action")
    print(f"---  -----  --------  ------  ------  ---------  ------")
    
    for i, price in enumerate(prices):
        old_min_price = min_price
        old_min_day = min_day
        
        if price < min_price:
            min_price = price
            min_day = i
            action = f"New min price"
        else:
            action = "Hold"
        
        current_profit = price - min_price
        if current_profit > max_profit:
            max_profit = current_profit
            action += f", New max profit"
        
        print(f"{i:3}  {price:5}  {min_price:8}  {min_day:6}  {current_profit:6}  {max_profit:9}  {action}")
        
        # Record potential transaction
        if current_profit > 0:
            transactions.append((min_day, i, current_profit))
    
    print(f"\nFinal Result:")
    max_profit, buy_day, sell_day = max_profit_with_details(prices)
    print(f"Maximum Profit: {max_profit}")
    
    if max_profit > 0:
        print(f"Optimal Strategy: Buy on day {buy_day} (price: {prices[buy_day]})")
        print(f"                 Sell on day {sell_day} (price: {prices[sell_day]})")
        print(f"                 Profit: {prices[sell_day]} - {prices[buy_day]} = {max_profit}")
    else:
        print(f"No profitable transaction possible")
    
    # Show all profitable transactions
    if transactions:
        print(f"\nAll profitable transactions:")
        for buy_d, sell_d, profit in sorted(transactions, key=lambda x: x[2], reverse=True):
            print(f"  Buy day {buy_d} (${prices[buy_d]}) → Sell day {sell_d} (${prices[sell_d]}) = ${profit} profit")
    
    return max_profit


def max_profit_variants():
    """
    STOCK PROBLEM VARIANTS:
    ======================
    Different scenarios and extensions.
    """
    
    def max_profit_with_fee(prices, fee):
        """Stock with transaction fee"""
        if not prices:
            return 0
        
        held = -prices[0]
        sold = 0
        
        for price in prices[1:]:
            held = max(held, sold - price)
            sold = max(sold, held + price - fee)
        
        return sold
    
    def max_profit_cooldown(prices):
        """Stock with 1-day cooldown after selling"""
        if len(prices) <= 1:
            return 0
        
        # sold[i] = max profit on day i when just sold
        # held[i] = max profit on day i when holding
        # rest[i] = max profit on day i when resting (cooldown)
        
        sold = 0
        held = -prices[0]
        rest = 0
        
        for price in prices[1:]:
            prev_sold = sold
            sold = held + price
            held = max(held, rest - price)
            rest = max(rest, prev_sold)
        
        return max(sold, rest)
    
    def find_all_profitable_periods(prices):
        """Find all periods where stock is increasing"""
        if not prices:
            return []
        
        periods = []
        start = 0
        
        for i in range(1, len(prices)):
            if prices[i] < prices[i-1]:
                if start < i-1:
                    periods.append((start, i-1, prices[i-1] - prices[start]))
                start = i
        
        # Check last period
        if start < len(prices) - 1:
            periods.append((start, len(prices)-1, prices[-1] - prices[start]))
        
        return periods
    
    # Test variants
    test_cases = [
        [7, 1, 5, 3, 6, 4],
        [7, 6, 4, 3, 1],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [2, 1, 2, 1, 0, 1, 2]
    ]
    
    print("Stock Problem Variants:")
    print("=" * 50)
    
    for prices in test_cases:
        print(f"\nPrices: {prices}")
        basic_profit = max_profit_min_price_tracking(prices)
        print(f"Basic max profit: {basic_profit}")
        
        # With transaction fee
        fee = 2
        fee_profit = max_profit_with_fee(prices, fee)
        print(f"With ${fee} fee: {fee_profit}")
        
        # With cooldown
        cooldown_profit = max_profit_cooldown(prices)
        print(f"With cooldown: {cooldown_profit}")
        
        # All profitable periods
        periods = find_all_profitable_periods(prices)
        print(f"Profitable periods: {periods}")


# Test cases
def test_max_profit():
    """Test all implementations with various inputs"""
    test_cases = [
        ([7, 1, 5, 3, 6, 4], 5),
        ([7, 6, 4, 3, 1], 0),
        ([1, 2, 3, 4, 5], 4),
        ([5, 4, 3, 2, 1], 0),
        ([2, 1, 2, 1, 0, 1, 2], 2),
        ([1], 0),
        ([], 0),
        ([3, 3], 0),
        ([1, 2], 1),
        ([2, 4, 1], 2)
    ]
    
    print("Testing Best Time to Buy and Sell Stock Solutions:")
    print("=" * 70)
    
    for i, (prices, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Prices: {prices}")
        print(f"Expected: {expected}")
        
        if len(prices) <= 10:  # Only for small arrays
            brute_force = max_profit_brute_force(prices)
            print(f"Brute Force:      {brute_force:>3} {'✓' if brute_force == expected else '✗'}")
        
        min_tracking = max_profit_min_price_tracking(prices)
        kadane = max_profit_kadane_variant(prices)
        state_machine = max_profit_dp_state_machine(prices)
        dp_explicit = max_profit_dp_explicit(prices)
        
        print(f"Min Tracking:     {min_tracking:>3} {'✓' if min_tracking == expected else '✗'}")
        print(f"Kadane Variant:   {kadane:>3} {'✓' if kadane == expected else '✗'}")
        print(f"State Machine:    {state_machine:>3} {'✓' if state_machine == expected else '✗'}")
        print(f"DP Explicit:      {dp_explicit:>3} {'✓' if dp_explicit == expected else '✗'}")
        
        # Show optimal transaction
        if prices:
            profit, buy_day, sell_day = max_profit_with_details(prices)
            if profit > 0:
                print(f"Optimal: Buy day {buy_day}, Sell day {sell_day}")
            else:
                print(f"No profitable transaction")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    max_profit_analysis([7, 1, 5, 3, 6, 4])
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    max_profit_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. SINGLE PASS: Optimal solution requires only one pass")
    print("2. MIN TRACKING: Track minimum price seen so far")
    print("3. STATE MACHINE: Model as buy/sell states")
    print("4. KADANE CONNECTION: Similar to maximum subarray")
    print("5. GREEDY CHOICE: Always sell at highest price after min")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Trading: Basic buy-low-sell-high strategy")
    print("• Investment: Single transaction optimization")
    print("• Algorithm Design: Foundation for more complex stock problems")
    print("• Economics: Market timing analysis")
    print("• Optimization: Constrained profit maximization")


if __name__ == "__main__":
    test_max_profit()


"""
BEST TIME TO BUY AND SELL STOCK - FOUNDATION OF STOCK PROBLEMS:
===============================================================

This is the fundamental stock trading problem that introduces key concepts:
- Single transaction constraint (buy once, sell once)
- Must buy before selling
- Maximize profit or return 0 if no profit possible
- Foundation for all other stock trading problems

KEY INSIGHTS:
============
1. **SINGLE PASS SOLUTION**: Optimal O(n) solution with one pass
2. **MIN PRICE TRACKING**: Keep track of minimum price seen so far
3. **GREEDY STRATEGY**: Always consider selling at current price
4. **STATE MACHINE**: Can model as buy/sell state transitions
5. **KADANE CONNECTION**: Similar structure to maximum subarray

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(n²) time, O(1) space
   - Try all buy-sell pairs
   - Simple but inefficient

2. **Min Price Tracking**: O(n) time, O(1) space
   - Track minimum price and maximum profit
   - Most intuitive optimal solution

3. **Kadane's Variant**: O(n) time, O(1) space
   - Transform to maximum subarray problem
   - Find maximum sum of price differences

4. **State Machine DP**: O(n) time, O(1) space
   - Model holding/not holding stock as states
   - Foundation for more complex stock problems

CORE ALGORITHM (MIN PRICE TRACKING):
===================================
```python
min_price = prices[0]
max_profit = 0

for price in prices:
    min_price = min(min_price, price)        # Update minimum
    profit = price - min_price               # Profit if sell today
    max_profit = max(max_profit, profit)     # Update maximum

return max_profit
```

STATE MACHINE FORMULATION:
=========================
Two states: HOLDING stock, NOT HOLDING stock

```python
held = -prices[0]    # Cost to buy on day 0
sold = 0             # No stock initially

for price in prices[1:]:
    held = max(held, -price)           # Keep holding or buy
    sold = max(sold, held + price)     # Keep not holding or sell

return sold  # Want to end without holding stock
```

KADANE'S ALGORITHM CONNECTION:
=============================
Transform to maximum subarray of daily price changes:
```python
max_ending_here = 0
max_so_far = 0

for i in range(1, len(prices)):
    daily_change = prices[i] - prices[i-1]
    max_ending_here = max(daily_change, max_ending_here + daily_change)
    max_so_far = max(max_so_far, max_ending_here)

return max_so_far
```

MATHEMATICAL PROPERTIES:
========================
- **Optimal Substructure**: Optimal solution contains optimal subsolutions
- **Greedy Choice**: Selling at any price after minimum is valid consideration
- **Monotonicity**: Maximum profit is non-decreasing as we process more days
- **Single Transaction**: Constraint simplifies to tracking single minimum

COMPLEXITY ANALYSIS:
===================
| Approach        | Time | Space | Notes                    |
|-----------------|------|-------|--------------------------|
| Brute Force     | O(n²)| O(1)  | Try all pairs           |
| Min Tracking    | O(n) | O(1)  | Optimal simple solution |
| Kadane Variant  | O(n) | O(1)  | Transform to max subarray|
| State Machine   | O(n) | O(1)  | DP foundation           |
| Explicit DP     | O(n) | O(n)  | Educational version     |

EDGE CASES:
==========
- **Empty array**: Return 0
- **Single element**: Return 0 (can't buy and sell)
- **Decreasing prices**: Return 0 (no profit possible)
- **Increasing prices**: Profit = last - first
- **All same prices**: Return 0

VARIANTS AND EXTENSIONS:
=======================
- **Transaction Fee**: Subtract fee from each transaction
- **Cooldown**: Must wait one day after selling before buying
- **Multiple Transactions**: Allow multiple buy-sell cycles
- **Limited Transactions**: At most k transactions allowed

APPLICATIONS:
============
- **Financial Trading**: Single stock transaction optimization
- **Investment Planning**: When to buy and sell assets
- **Resource Allocation**: Optimal timing for resource acquisition/disposal
- **Market Analysis**: Identifying optimal trading opportunities

RELATED PROBLEMS:
================
- **Best Time to Buy and Sell Stock II (122)**: Multiple transactions
- **Best Time to Buy and Sell Stock III (123)**: At most 2 transactions
- **Best Time to Buy and Sell Stock IV (188)**: At most k transactions
- **Best Time to Buy and Sell with Cooldown (309)**: Cooldown constraint
- **Best Time to Buy and Sell with Fee (714)**: Transaction fee

STATE MACHINE FOUNDATION:
========================
This problem introduces the state machine concept used in all stock problems:

```
States:
- HOLD: Currently holding stock
- SOLD: Not holding stock

Transitions:
- HOLD → HOLD: Continue holding
- HOLD → SOLD: Sell stock
- SOLD → HOLD: Buy stock (only if haven't bought before)
- SOLD → SOLD: Continue not holding
```

OPTIMIZATION INSIGHTS:
=====================
1. **Single Pass**: One pass through array is sufficient
2. **Constant Space**: No need to store intermediate results
3. **Early Termination**: Can stop if remaining days can't improve profit
4. **Greedy Valid**: Local optimal choices lead to global optimum

This foundational problem establishes the patterns and techniques used
throughout the entire family of stock trading problems.
"""
