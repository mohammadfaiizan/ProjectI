"""
LeetCode 714: Best Time to Buy and Sell Stock with Transaction Fee
Difficulty: Medium
Category: Stock Problems - Transaction Costs

PROBLEM DESCRIPTION:
===================
You are given an array prices where prices[i] is the price of a given stock on the ith day, 
and an integer fee representing a transaction fee.
Find the maximum profit you can achieve. You may complete as many transactions as you like, 
but you need to pay the transaction fee for each transaction.
Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

Example 1:
Input: prices = [1,3,2,6,5,4], fee = 2
Output: 4
Explanation: The maximum profit can be achieved by:
- Buy at prices[0] = 1
- Sell at prices[3] = 6
- Total profit = 6 - 1 - 2 = 3
Then repeat the above to gain profit of 1 more.

Example 2:
Input: prices = [1,3,7,5,10,3], fee = 3
Output: 6

Constraints:
- 1 <= prices.length <= 5 * 10^4
- 1 <= prices[i] < 5 * 10^4
- 0 <= fee < 5 * 10^4
"""

def max_profit_state_machine(prices, fee):
    """
    STATE MACHINE APPROACH:
    ======================
    Two states: holding stock, not holding stock.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if len(prices) <= 1:
        return 0
    
    # States:
    # held = max profit when holding stock
    # sold = max profit when not holding stock
    
    held = -prices[0]  # Buy stock on day 0
    sold = 0           # Don't hold stock on day 0
    
    for price in prices[1:]:
        # Update states
        new_held = max(held, sold - price)          # Keep holding or buy
        new_sold = max(sold, held + price - fee)    # Keep not holding or sell (pay fee)
        
        held = new_held
        sold = new_sold
    
    return sold  # We want to end without holding stock


def max_profit_dp_explicit(prices, fee):
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
        # Not holding: either didn't have it or sold it today (pay fee)
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i] - fee)
        
        # Holding: either had it or bought it today
        dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
    
    return dp[n-1][0]


def max_profit_greedy_simulation(prices, fee):
    """
    GREEDY SIMULATION APPROACH:
    ===========================
    Simulate greedy trading with fee consideration.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if len(prices) <= 1:
        return 0
    
    total_profit = 0
    min_price = prices[0]
    
    for price in prices[1:]:
        if price < min_price:
            min_price = price
        elif price > min_price + fee:
            # Profitable to sell (price - min_price > fee)
            total_profit += price - min_price - fee
            min_price = price - fee  # Effective cost basis for next trade
    
    return total_profit


def max_profit_threshold_approach(prices, fee):
    """
    THRESHOLD APPROACH:
    ==================
    Only trade when profit exceeds fee threshold.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if len(prices) <= 1:
        return 0
    
    buy_price = prices[0]
    profit = 0
    
    for price in prices[1:]:
        if price > buy_price + fee:
            # Profitable to sell
            profit += price - buy_price - fee
            buy_price = price - fee  # New effective buy price
        elif price < buy_price:
            # Better buy opportunity
            buy_price = price
    
    return profit


def max_profit_with_transactions(prices, fee):
    """
    TRACK ACTUAL TRANSACTIONS:
    ==========================
    Return profit and list of optimal transactions with fees.
    
    Time Complexity: O(n) - single pass with tracking
    Space Complexity: O(k) - k transactions
    """
    if len(prices) <= 1:
        return 0, []
    
    transactions = []
    total_profit = 0
    buy_day = 0
    buy_price = prices[0]
    
    for i in range(1, len(prices)):
        price = prices[i]
        
        if price > buy_price + fee:
            # Profitable to sell
            sell_profit = price - buy_price - fee
            total_profit += sell_profit
            transactions.append((buy_day, i, buy_price, price, sell_profit))
            
            # Set new buy price for potential next transaction
            buy_price = price - fee
            buy_day = i
        elif price < buy_price:
            # Better buy opportunity
            buy_price = price
            buy_day = i
    
    return total_profit, transactions


def max_profit_analysis(prices, fee):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation with transaction fees.
    """
    print(f"Stock Trading with Transaction Fee Analysis:")
    print(f"Prices: {prices}")
    print(f"Transaction Fee: ${fee}")
    
    if not prices:
        print("No prices available.")
        return 0
    
    n = len(prices)
    print(f"Days: {list(range(n))}")
    
    # State machine analysis
    print(f"\nState machine evolution:")
    held = -prices[0]
    sold = 0
    
    print(f"Day  Price  Held   Sold   Net Profit  Action")
    print(f"---  -----  -----  -----  ----------  ------")
    print(f"  0  {prices[0]:5}  {held:5}  {sold:5}  {sold:10}  Buy")
    
    for i in range(1, len(prices)):
        price = prices[i]
        
        # Calculate potential new states
        potential_held = max(held, sold - price)
        potential_sold = max(sold, held + price - fee)
        
        # Determine action
        action = "Hold"
        if potential_held > held and potential_held == sold - price:
            action = "Buy"
        elif potential_sold > sold and potential_sold == held + price - fee:
            action = f"Sell (pay ${fee} fee)"
        
        held = potential_held
        sold = potential_sold
        
        print(f"{i:3}  {price:5}  {held:5}  {sold:5}  {sold:10}  {action}")
    
    print(f"\nFinal maximum profit: {sold}")
    
    # Show optimal transactions
    profit, transactions = max_profit_with_transactions(prices, fee)
    print(f"\nOptimal transactions with ${fee} fee (total profit: {profit}):")
    
    if transactions:
        total_fees = 0
        for i, (buy_day, sell_day, buy_price, sell_price, net_profit) in enumerate(transactions):
            gross_profit = sell_price - buy_price
            print(f"  {i+1}. Buy day {buy_day} (${buy_price}) → Sell day {sell_day} (${sell_price})")
            print(f"      Gross profit: ${gross_profit}, Fee: ${fee}, Net profit: ${net_profit}")
            total_fees += fee
        print(f"Total fees paid: ${total_fees}")
    else:
        print(f"  No profitable transactions (all trades would lose money after fees)")
    
    # Compare with no-fee scenario
    no_fee_profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            no_fee_profit += prices[i] - prices[i-1]
    
    print(f"\nComparison:")
    print(f"  With ${fee} fee: {profit}")
    print(f"  Without fee: {no_fee_profit}")
    print(f"  Fee impact: -{no_fee_profit - profit}")
    
    # Show fee efficiency
    if transactions:
        avg_profit_per_transaction = profit / len(transactions)
        print(f"  Average net profit per transaction: ${avg_profit_per_transaction:.2f}")
        print(f"  Fee as % of gross profit: {(total_fees / (profit + total_fees) * 100):.1f}%")
    
    return profit


def max_profit_variants():
    """
    TRANSACTION FEE VARIANTS:
    ========================
    Different fee scenarios and optimizations.
    """
    
    def max_profit_progressive_fee(prices, base_fee, fee_increment):
        """Progressive fee that increases with each transaction"""
        if len(prices) <= 1:
            return 0
        
        held = -prices[0]
        sold = 0
        transaction_count = 0
        
        for price in prices[1:]:
            current_fee = base_fee + transaction_count * fee_increment
            
            new_held = max(held, sold - price)
            new_sold = max(sold, held + price - current_fee)
            
            if new_sold > sold and new_sold == held + price - current_fee:
                transaction_count += 1
            
            held = new_held
            sold = new_sold
        
        return sold
    
    def max_profit_percentage_fee(prices, fee_percentage):
        """Fee as percentage of transaction value"""
        if len(prices) <= 1:
            return 0
        
        held = -prices[0]
        sold = 0
        
        for price in prices[1:]:
            fee = price * fee_percentage / 100
            
            new_held = max(held, sold - price)
            new_sold = max(sold, held + price - fee)
            
            held = new_held
            sold = new_sold
        
        return sold
    
    def max_profit_buy_sell_fees(prices, buy_fee, sell_fee):
        """Different fees for buying and selling"""
        if len(prices) <= 1:
            return 0
        
        held = -prices[0] - buy_fee  # Pay buy fee when buying
        sold = 0
        
        for price in prices[1:]:
            new_held = max(held, sold - price - buy_fee)
            new_sold = max(sold, held + price - sell_fee)
            
            held = new_held
            sold = new_sold
        
        return sold
    
    def optimal_fee_threshold(prices):
        """Find the fee threshold where trading becomes unprofitable"""
        if len(prices) <= 1:
            return 0
        
        # Binary search for maximum fee where profit > 0
        left, right = 0, max(prices) - min(prices)
        
        while left < right:
            mid = (left + right + 1) // 2
            profit = max_profit_state_machine(prices, mid)
            
            if profit > 0:
                left = mid
            else:
                right = mid - 1
        
        return left
    
    # Test variants
    test_cases = [
        ([1, 3, 2, 6, 5, 4], 2),
        ([1, 3, 7, 5, 10, 3], 3),
        ([1, 2, 3, 4, 5], 1),
        ([5, 4, 3, 2, 1], 1),
        ([2, 1, 4, 9], 2)
    ]
    
    print("Transaction Fee Variants:")
    print("=" * 50)
    
    for prices, base_fee in test_cases:
        print(f"\nPrices: {prices}, Base fee: ${base_fee}")
        
        standard_fee = max_profit_state_machine(prices, base_fee)
        print(f"Standard fee (${base_fee}): {standard_fee}")
        
        progressive_fee = max_profit_progressive_fee(prices, base_fee, 1)
        print(f"Progressive fee (+$1 per transaction): {progressive_fee}")
        
        percentage_fee = max_profit_percentage_fee(prices, 5)  # 5%
        print(f"5% percentage fee: {percentage_fee:.2f}")
        
        buy_sell_fees = max_profit_buy_sell_fees(prices, base_fee//2, base_fee//2)
        print(f"Split buy/sell fees: {buy_sell_fees}")
        
        threshold = optimal_fee_threshold(prices)
        print(f"Maximum profitable fee: ${threshold}")


# Test cases
def test_max_profit_with_fee():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1, 3, 2, 6, 5, 4], 2, 4),
        ([1, 3, 7, 5, 10, 3], 3, 6),
        ([1, 2, 3, 4, 5], 1, 3),
        ([7, 6, 4, 3, 1], 2, 0),
        ([1, 4, 6, 2, 8, 3, 10, 14], 3, 13),
        ([1], 1, 0),
        ([], 1, 0),
        ([5, 5], 1, 0),
        ([1, 10], 5, 4)
    ]
    
    print("Testing Best Time to Buy and Sell Stock with Transaction Fee:")
    print("=" * 70)
    
    for i, (prices, fee, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Prices: {prices}, Fee: ${fee}")
        print(f"Expected: {expected}")
        
        state_machine = max_profit_state_machine(prices, fee)
        dp_explicit = max_profit_dp_explicit(prices, fee)
        greedy_sim = max_profit_greedy_simulation(prices, fee)
        threshold = max_profit_threshold_approach(prices, fee)
        
        print(f"State Machine:    {state_machine:>3} {'✓' if state_machine == expected else '✗'}")
        print(f"Explicit DP:      {dp_explicit:>3} {'✓' if dp_explicit == expected else '✗'}")
        print(f"Greedy Sim:       {greedy_sim:>3} {'✓' if greedy_sim == expected else '✗'}")
        print(f"Threshold:        {threshold:>3} {'✓' if threshold == expected else '✗'}")
        
        # Show transactions for small cases
        if len(prices) <= 8:
            profit, transactions = max_profit_with_transactions(prices, fee)
            print(f"Transactions: {len(transactions)} total, profit: {profit}")
            for j, (buy_day, sell_day, buy_price, sell_price, net_profit) in enumerate(transactions):
                gross = sell_price - buy_price
                print(f"  {j+1}. Day {buy_day}→{sell_day}: ${buy_price}→${sell_price} (gross: ${gross}, net: ${net_profit})")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    max_profit_analysis([1, 3, 2, 6, 5, 4], 2)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    max_profit_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. FEE THRESHOLD: Only trade when profit > fee")
    print("2. EFFECTIVE COST: After selling, new buy cost = sell_price - fee")
    print("3. PROFIT REDUCTION: Fees significantly impact small-margin trades")
    print("4. TRANSACTION FREQUENCY: Higher fees favor longer holding periods")
    print("5. BREAKEVEN ANALYSIS: Some trades become unprofitable with fees")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Real Trading: Brokerage fees and commissions")
    print("• Investment: Transaction cost optimization")
    print("• Economic Modeling: Friction costs in markets")
    print("• System Design: Cost-aware resource allocation")
    print("• Financial Planning: Net profit maximization")


if __name__ == "__main__":
    test_max_profit_with_fee()


"""
BEST TIME TO BUY AND SELL STOCK WITH TRANSACTION FEE - COST OPTIMIZATION:
=========================================================================

This problem adds transaction costs to unlimited stock trading:
- Can trade as many times as desired
- Must pay a fixed fee for each transaction (sell operation)
- Demonstrates how costs change optimal trading strategies
- Shows the balance between transaction frequency and profitability

KEY INSIGHTS:
============
1. **FEE THRESHOLD**: Only trade when profit exceeds transaction fee
2. **EFFECTIVE COST BASIS**: After selling, new cost basis = sell_price - fee
3. **PROFIT EROSION**: Fees can make small-margin trades unprofitable
4. **FREQUENCY REDUCTION**: Higher fees favor longer holding periods
5. **BREAKEVEN ANALYSIS**: Must consider net profit, not gross profit

ALGORITHM APPROACHES:
====================

1. **State Machine**: O(n) time, O(1) space
   - Two states: holding, not holding
   - Subtract fee from sell transactions
   - Most straightforward approach

2. **Greedy Simulation**: O(n) time, O(1) space
   - Track minimum price and sell when profitable
   - Natural trading simulation

3. **Threshold Approach**: O(n) time, O(1) space
   - Only execute trades above profit threshold
   - Clear decision logic

4. **Explicit DP**: O(n) time, O(n) space
   - Educational version showing DP structure

CORE STATE MACHINE:
==================
```python
held = -prices[0]    # Cost to hold stock
sold = 0             # Profit when not holding

for price in prices[1:]:
    held = max(held, sold - price)          # Keep holding or buy
    sold = max(sold, held + price - fee)    # Keep not holding or sell (pay fee)

return sold
```

**Key Modification**: Subtract fee when selling:
`sold = max(sold, held + price - fee)`

GREEDY SIMULATION:
=================
```python
total_profit = 0
min_price = prices[0]

for price in prices[1:]:
    if price < min_price:
        min_price = price
    elif price > min_price + fee:
        # Profitable after fee
        total_profit += price - min_price - fee
        min_price = price - fee  # New effective cost basis

return total_profit
```

**Effective Cost Basis**: After selling at price P with fee F:
- Gross profit from buy_price B: P - B
- Net profit: P - B - F  
- If we immediately rebuy at price P: effective cost = P - F
- This allows potential profit from future price increases

THRESHOLD DECISION LOGIC:
========================
```python
buy_price = prices[0]
profit = 0

for price in prices[1:]:
    if price > buy_price + fee:
        # Profitable to sell
        profit += price - buy_price - fee
        buy_price = price - fee  # New effective buy price
    elif price < buy_price:
        buy_price = price  # Better buy opportunity

return profit
```

FEE IMPACT ANALYSIS:
===================
Transaction fees affect trading in several ways:

1. **Minimum Profit Threshold**: 
   - Without fee: Any price increase is profitable
   - With fee: Need price increase > fee

2. **Transaction Frequency**:
   - Without fee: Can capture every small fluctuation
   - With fee: Prefer larger price movements

3. **Holding Period**:
   - Without fee: Can day-trade efficiently
   - With fee: Longer holds become more attractive

4. **Breakeven Calculation**:
   - Gross profit = sell_price - buy_price
   - Net profit = gross_profit - fee
   - Only trade if net profit > 0

MATHEMATICAL RELATIONSHIPS:
==========================

**Profit Comparison**:
- Without fee: profit = Σ max(0, prices[i] - prices[i-1])
- With fee: profit = Σ (segment_profit - fee) for profitable segments

**Effective Cost After Trading**:
- Buy at price B, sell at price S with fee F
- Net profit: S - B - F
- Effective cost for next trade: S - F (not S)
- This maintains optimal position for future opportunities

COMPLEXITY ANALYSIS:
===================
| Approach        | Time | Space | Trade-offs            |
|-----------------|------|-------|-----------------------|
| State Machine   | O(n) | O(1)  | Clean, efficient      |
| Greedy Sim      | O(n) | O(1)  | Natural simulation    |
| Threshold       | O(n) | O(1)  | Clear decision logic  |
| Explicit DP     | O(n) | O(n)  | Educational value     |

EDGE CASES:
==========
- **High fees**: May make all trades unprofitable
- **Low volatility**: Small price movements insufficient to cover fees
- **Single large move**: One trade may be optimal despite unlimited transactions
- **Decreasing prices**: No profitable trades regardless of fee
- **Fee = 0**: Reduces to unlimited transactions problem

VARIANTS AND EXTENSIONS:
=======================
- **Progressive fees**: Increasing cost per transaction
- **Percentage fees**: Fee as % of transaction value
- **Buy/sell fees**: Different costs for buying vs selling
- **Volume-based fees**: Lower fees for larger transactions
- **Time-based fees**: Different fees based on holding period

REAL-WORLD APPLICATIONS:
=======================
- **Stock Trading**: Brokerage commissions and fees
- **Cryptocurrency**: Exchange trading fees
- **Forex Trading**: Spread and commission costs
- **Investment Management**: Transaction cost analysis
- **Algorithmic Trading**: Cost-aware strategy optimization

OPTIMIZATION STRATEGIES:
=======================
1. **Fee Amortization**: Spread costs over larger price movements
2. **Batch Trading**: Combine multiple small trades into larger ones
3. **Threshold Setting**: Establish minimum profit requirements
4. **Frequency Reduction**: Trade less often but more profitably
5. **Cost Monitoring**: Track fee impact on overall returns

RELATIONSHIP TO OTHER PROBLEMS:
==============================
- **Stock II (122)**: Remove transaction fee
- **Stock with Cooldown (309)**: Add temporal constraint
- **Stock III (123)**: Add transaction count limit
- **Stock IV (188)**: Generalize transaction limits

PRACTICAL CONSIDERATIONS:
========================
- **Market Impact**: Large trades affect prices
- **Slippage**: Execution price differs from quoted price
- **Bid-Ask Spread**: Additional implicit cost
- **Tax Implications**: Short-term vs long-term capital gains
- **Opportunity Cost**: Consider alternative investments

This problem elegantly demonstrates how transaction costs fundamentally
alter trading strategies, requiring a balance between transaction frequency
and profitability that mirrors real-world trading scenarios.
"""
