"""
LeetCode 122: Best Time to Buy and Sell Stock II
Difficulty: Medium
Category: Stock Problems - Multiple Transactions

PROBLEM DESCRIPTION:
===================
You are given an integer array prices where prices[i] is the price of a given stock on the ith day.
On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. 
However, you can buy it then immediately sell it on the same day.
Find and return the maximum profit you can achieve.

Example 1:
Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
Total profit is 4 + 3 = 7.

Example 2:
Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are engaging multiple transactions at the same time. You must sell before buying again.

Example 3:
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: There is no way to make a positive profit, so we never buy the stock to achieve the maximum profit of 0.

Constraints:
- 1 <= prices.length <= 3 * 10^4
- 0 <= prices[i] <= 10^4
"""

def max_profit_greedy(prices):
    """
    GREEDY APPROACH:
    ===============
    Buy before every increase, sell before every decrease.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if not prices:
        return 0
    
    total_profit = 0
    
    for i in range(1, len(prices)):
        # If price increases, capture the profit
        if prices[i] > prices[i-1]:
            total_profit += prices[i] - prices[i-1]
    
    return total_profit


def max_profit_state_machine(prices):
    """
    STATE MACHINE APPROACH:
    ======================
    Model with buy/sell states allowing multiple transactions.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if not prices:
        return 0
    
    # States: held = max profit when holding stock
    #         sold = max profit when not holding stock
    held = -prices[0]  # Cost to buy on day 0
    sold = 0           # No stock initially
    
    for i in range(1, len(prices)):
        # Update states
        new_held = max(held, sold - prices[i])  # Keep holding or buy today
        new_sold = max(sold, held + prices[i])  # Keep not holding or sell today
        
        held = new_held
        sold = new_sold
    
    return sold  # End without holding stock


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
    dp[0][1] = -prices[0] # Holding stock on day 0
    
    for i in range(1, n):
        # Not holding: either didn't have it or sold it today
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
        
        # Holding: either had it or bought it today
        # Can buy multiple times since we can do multiple transactions
        dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
    
    return dp[n-1][0]


def max_profit_peak_valley(prices):
    """
    PEAK-VALLEY APPROACH:
    ====================
    Identify peaks and valleys for optimal transactions.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if len(prices) <= 1:
        return 0
    
    total_profit = 0
    i = 0
    
    while i < len(prices) - 1:
        # Find valley (local minimum)
        while i < len(prices) - 1 and prices[i + 1] <= prices[i]:
            i += 1
        
        if i == len(prices) - 1:
            break
        
        valley = prices[i]
        
        # Find peak (local maximum)
        while i < len(prices) - 1 and prices[i + 1] >= prices[i]:
            i += 1
        
        peak = prices[i]
        
        # Add profit from this valley-peak pair
        total_profit += peak - valley
    
    return total_profit


def max_profit_with_transactions(prices):
    """
    TRACK ACTUAL TRANSACTIONS:
    ==========================
    Return profit and list of optimal transactions.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(k) - k transactions
    """
    if len(prices) <= 1:
        return 0, []
    
    transactions = []
    total_profit = 0
    i = 0
    
    while i < len(prices) - 1:
        # Find valley
        while i < len(prices) - 1 and prices[i + 1] <= prices[i]:
            i += 1
        
        if i == len(prices) - 1:
            break
        
        buy_day = i
        buy_price = prices[i]
        
        # Find peak
        while i < len(prices) - 1 and prices[i + 1] >= prices[i]:
            i += 1
        
        sell_day = i
        sell_price = prices[i]
        
        profit = sell_price - buy_price
        total_profit += profit
        transactions.append((buy_day, sell_day, buy_price, sell_price, profit))
    
    return total_profit, transactions


def max_profit_analysis(prices):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step profit computation and transaction analysis.
    """
    print(f"Multiple Transactions Stock Analysis:")
    print(f"Prices: {prices}")
    
    if not prices:
        print("No prices available.")
        return 0
    
    n = len(prices)
    print(f"Days: {list(range(n))}")
    
    # Greedy approach analysis
    print(f"\nGreedy approach (capture every increase):")
    print(f"Day  Price  Change  Profit  Total")
    print(f"---  -----  ------  ------  -----")
    
    total_profit = 0
    for i, price in enumerate(prices):
        if i == 0:
            change = 0
            daily_profit = 0
        else:
            change = price - prices[i-1]
            daily_profit = max(0, change)
            total_profit += daily_profit
        
        print(f"{i:3}  {price:5}  {change:6}  {daily_profit:6}  {total_profit:5}")
    
    print(f"\nGreedy total profit: {total_profit}")
    
    # Peak-valley analysis
    profit, transactions = max_profit_with_transactions(prices)
    print(f"\nPeak-Valley approach:")
    print(f"Total profit: {profit}")
    
    if transactions:
        print(f"\nOptimal transactions:")
        for i, (buy_day, sell_day, buy_price, sell_price, trans_profit) in enumerate(transactions):
            print(f"  {i+1}. Buy day {buy_day} (${buy_price}) → Sell day {sell_day} (${sell_price}) = ${trans_profit}")
    else:
        print(f"No profitable transactions possible")
    
    # State machine visualization
    print(f"\nState machine evolution:")
    held = -prices[0]
    sold = 0
    
    print(f"Day  Price  Held   Sold   Action")
    print(f"---  -----  -----  -----  ------")
    print(f"  0  {prices[0]:5}  {held:5}  {sold:5}  Buy")
    
    for i in range(1, len(prices)):
        price = prices[i]
        new_held = max(held, sold - price)
        new_sold = max(sold, held + price)
        
        if new_held > held:
            action = f"Buy"
        elif new_sold > sold:
            action = f"Sell"
        else:
            action = f"Hold"
        
        held = new_held
        sold = new_sold
        
        print(f"{i:3}  {price:5}  {held:5}  {sold:5}  {action}")
    
    return total_profit


def max_profit_variants():
    """
    MULTIPLE TRANSACTION VARIANTS:
    =============================
    Different scenarios and constraints.
    """
    
    def max_profit_with_fee(prices, fee):
        """Multiple transactions with transaction fee"""
        if not prices:
            return 0
        
        held = -prices[0]
        sold = 0
        
        for price in prices[1:]:
            held = max(held, sold - price)
            sold = max(sold, held + price - fee)
        
        return sold
    
    def max_profit_cooldown(prices):
        """Multiple transactions with cooldown"""
        if len(prices) <= 1:
            return 0
        
        # Three states: sold (just sold), held (holding), rest (cooldown)
        sold = 0
        held = -prices[0]
        rest = 0
        
        for price in prices[1:]:
            prev_sold = sold
            sold = held + price
            held = max(held, rest - price)
            rest = max(rest, prev_sold)
        
        return max(sold, rest)
    
    def max_profit_min_transactions(prices):
        """Find minimum number of transactions for maximum profit"""
        if len(prices) <= 1:
            return 0, 0
        
        profit, transactions = max_profit_with_transactions(prices)
        return profit, len(transactions)
    
    def max_profit_with_hold_limit(prices, max_hold_days):
        """Limit how long we can hold stock"""
        if not prices or max_hold_days <= 0:
            return 0
        
        # dp[i][j] = max profit on day i holding stock for j days
        n = len(prices)
        dp = [[-float('inf')] * (max_hold_days + 1) for _ in range(n)]
        sold = [0] * n
        
        # Base case
        dp[0][1] = -prices[0]  # Buy on day 0, hold for 1 day
        sold[0] = 0
        
        for i in range(1, n):
            sold[i] = sold[i-1]
            
            # Try selling stock held for different days
            for hold_days in range(1, max_hold_days + 1):
                if dp[i-1][hold_days] != -float('inf'):
                    sold[i] = max(sold[i], dp[i-1][hold_days] + prices[i])
            
            # Buy stock today
            dp[i][1] = max(dp[i-1][1] if i > 0 else -float('inf'), sold[i-1] - prices[i])
            
            # Continue holding
            for hold_days in range(2, max_hold_days + 1):
                if i > 0:
                    dp[i][hold_days] = dp[i-1][hold_days-1]
        
        return sold[n-1]
    
    # Test variants
    test_cases = [
        [7, 1, 5, 3, 6, 4],
        [1, 2, 3, 4, 5],
        [7, 6, 4, 3, 1],
        [2, 1, 4, 5, 2, 9, 7]
    ]
    
    print("Multiple Transaction Variants:")
    print("=" * 50)
    
    for prices in test_cases:
        print(f"\nPrices: {prices}")
        
        basic_profit = max_profit_greedy(prices)
        print(f"Basic (unlimited): {basic_profit}")
        
        # With transaction fee
        fee = 2
        fee_profit = max_profit_with_fee(prices, fee)
        print(f"With ${fee} fee: {fee_profit}")
        
        # With cooldown
        cooldown_profit = max_profit_cooldown(prices)
        print(f"With cooldown: {cooldown_profit}")
        
        # Minimum transactions
        profit, min_trans = max_profit_min_transactions(prices)
        print(f"Min transactions: {min_trans} for ${profit}")
        
        # With hold limit
        hold_limit = 3
        hold_profit = max_profit_with_hold_limit(prices, hold_limit)
        print(f"Max hold {hold_limit} days: {hold_profit}")


# Test cases
def test_max_profit_ii():
    """Test all implementations with various inputs"""
    test_cases = [
        ([7, 1, 5, 3, 6, 4], 7),
        ([1, 2, 3, 4, 5], 4),
        ([7, 6, 4, 3, 1], 0),
        ([1, 2, 1, 2], 2),
        ([2, 1, 2, 0, 1], 2),
        ([1], 0),
        ([], 0),
        ([3, 3], 0),
        ([1, 2], 1),
        ([2, 1], 0)
    ]
    
    print("Testing Best Time to Buy and Sell Stock II Solutions:")
    print("=" * 70)
    
    for i, (prices, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Prices: {prices}")
        print(f"Expected: {expected}")
        
        greedy = max_profit_greedy(prices)
        state_machine = max_profit_state_machine(prices)
        dp_explicit = max_profit_dp_explicit(prices)
        peak_valley = max_profit_peak_valley(prices)
        
        print(f"Greedy:           {greedy:>3} {'✓' if greedy == expected else '✗'}")
        print(f"State Machine:    {state_machine:>3} {'✓' if state_machine == expected else '✗'}")
        print(f"DP Explicit:      {dp_explicit:>3} {'✓' if dp_explicit == expected else '✗'}")
        print(f"Peak-Valley:      {peak_valley:>3} {'✓' if peak_valley == expected else '✗'}")
        
        # Show transactions for small cases
        if len(prices) <= 8:
            profit, transactions = max_profit_with_transactions(prices)
            if transactions:
                print(f"Transactions: {len(transactions)} total")
                for j, (buy_day, sell_day, buy_price, sell_price, trans_profit) in enumerate(transactions):
                    print(f"  {j+1}. Day {buy_day}→{sell_day}: ${buy_price}→${sell_price} = +${trans_profit}")
            else:
                print(f"No profitable transactions")
    
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
    print("1. GREEDY OPTIMAL: Capture every price increase")
    print("2. MULTIPLE TRANSACTIONS: No limit on number of trades")
    print("3. PEAK-VALLEY: Buy at valleys, sell at peaks")
    print("4. STATE MACHINE: Hold/sold states with transitions")
    print("5. DAILY DECISIONS: Can buy and sell on same day")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Day Trading: Unlimited transaction strategies")
    print("• Algorithmic Trading: Trend following algorithms")
    print("• Investment: Multiple entry/exit strategies")
    print("• Market Analysis: Optimal timing identification")
    print("• Resource Trading: Buy low, sell high repeatedly")


if __name__ == "__main__":
    test_max_profit_ii()


"""
BEST TIME TO BUY AND SELL STOCK II - UNLIMITED TRANSACTIONS:
============================================================

This problem extends the single transaction case to unlimited transactions:
- Can buy and sell as many times as desired
- Must sell before buying again (can't hold multiple shares)
- Can buy and sell on the same day
- Demonstrates the power of greedy algorithms in stock trading

KEY INSIGHTS:
============
1. **GREEDY OPTIMAL**: Capturing every price increase is optimal
2. **UNLIMITED TRANSACTIONS**: No constraint on number of trades
3. **PEAK-VALLEY PATTERN**: Buy at local minima, sell at local maxima
4. **SAME-DAY TRADING**: Can buy and sell on the same day
5. **ADDITIVE PROFIT**: Total profit = sum of all profitable segments

ALGORITHM APPROACHES:
====================

1. **Greedy Approach**: O(n) time, O(1) space
   - Capture every price increase
   - Most intuitive and efficient

2. **State Machine**: O(n) time, O(1) space
   - Model holding/not holding states
   - Foundation for constrained versions

3. **Peak-Valley**: O(n) time, O(1) space
   - Find local minima and maxima
   - Natural trading strategy

4. **Explicit DP**: O(n) time, O(n) space
   - Educational version showing DP structure

CORE GREEDY ALGORITHM:
=====================
```python
total_profit = 0

for i in range(1, len(prices)):
    if prices[i] > prices[i-1]:
        total_profit += prices[i] - prices[i-1]

return total_profit
```

**Why Greedy Works:**
- Every profitable segment contributes to optimal solution
- No benefit in skipping profitable opportunities
- Mathematical proof: any optimal solution can be decomposed into profitable segments

PEAK-VALLEY ALGORITHM:
=====================
```python
total_profit = 0
i = 0

while i < len(prices) - 1:
    # Find valley (local minimum)
    while i < len(prices) - 1 and prices[i + 1] <= prices[i]:
        i += 1
    valley = prices[i]
    
    # Find peak (local maximum)
    while i < len(prices) - 1 and prices[i + 1] >= prices[i]:
        i += 1
    peak = prices[i]
    
    total_profit += peak - valley

return total_profit
```

STATE MACHINE FORMULATION:
=========================
```
States:
- HELD: Currently holding stock
- SOLD: Not holding stock

Transitions:
- HELD → HELD: Continue holding
- HELD → SOLD: Sell stock (profit = price)
- SOLD → HELD: Buy stock (cost = -price)
- SOLD → SOLD: Continue not holding

Update equations:
held = max(held, sold - price)    # Keep holding or buy
sold = max(sold, held + price)    # Keep not holding or sell
```

MATHEMATICAL ANALYSIS:
=====================
**Theorem**: The greedy algorithm is optimal.

**Proof Sketch**:
1. Any optimal solution can be viewed as a sequence of buy-sell pairs
2. Each pair contributes (sell_price - buy_price) to total profit
3. Between any buy day i and sell day j (i < j), the greedy algorithm captures:
   sum(max(0, prices[k] - prices[k-1])) for k in [i+1, j]
4. This sum equals prices[j] - prices[i] when prices[j] > prices[i]
5. Therefore, greedy captures all profitable opportunities optimally

COMPLEXITY COMPARISON:
=====================
| Approach        | Time | Space | Intuition            |
|-----------------|------|-------|----------------------|
| Greedy          | O(n) | O(1)  | Capture every gain   |
| State Machine   | O(n) | O(1)  | DP state transitions |
| Peak-Valley     | O(n) | O(1)  | Natural trading      |
| Explicit DP     | O(n) | O(n)  | Educational clarity  |

EDGE CASES:
==========
- **Empty array**: Return 0
- **Single element**: Return 0 (can't trade)
- **Decreasing prices**: Return 0 (no profit possible)
- **Increasing prices**: Profit = last - first
- **Constant prices**: Return 0 (no profitable trades)

EXTENSIONS AND VARIANTS:
=======================
- **Transaction Fee**: Subtract fee from each sale
- **Cooldown**: Must wait one day after selling
- **Hold Limit**: Maximum days to hold stock
- **Volume Constraints**: Limited shares available

APPLICATIONS:
============
- **Day Trading**: Unlimited transaction strategies
- **Algorithmic Trading**: Trend-following algorithms
- **Market Making**: Continuous buy-sell operations
- **Arbitrage**: Exploiting price differences
- **Resource Trading**: Commodity trading strategies

RELATIONSHIP TO OTHER PROBLEMS:
==============================
- **Stock I (121)**: Single transaction version
- **Stock III (123)**: At most 2 transactions
- **Stock IV (188)**: At most k transactions
- **Stock with Cooldown (309)**: Adds cooldown constraint
- **Stock with Fee (714)**: Adds transaction cost

GREEDY VS DP PERSPECTIVE:
========================
This problem showcases when greedy algorithms are optimal:

**Greedy Properties Met**:
1. **Greedy Choice Property**: Local optimal choices lead to global optimum
2. **Optimal Substructure**: Problem breaks down optimally
3. **No Dependencies**: Each decision is independent of future unknown prices

**When Greedy Fails** (other stock problems):
- Limited transactions (must choose which opportunities to take)
- Cooldown periods (decisions affect future availability)
- Transaction costs (may make some trades unprofitable)

PRACTICAL TRADING INSIGHTS:
===========================
- **Trend Following**: Always stay with the trend
- **No Market Timing**: Don't try to predict perfect peaks/valleys
- **Transaction Costs**: In reality, fees make frequent trading expensive
- **Slippage**: Real markets have bid-ask spreads
- **Risk Management**: Unlimited transactions increase exposure

This problem elegantly demonstrates how unlimited resources (transactions)
can simplify an optimization problem to a simple greedy strategy.
"""
