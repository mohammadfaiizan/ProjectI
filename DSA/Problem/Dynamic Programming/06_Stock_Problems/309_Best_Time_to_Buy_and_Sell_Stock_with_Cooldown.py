"""
LeetCode 309: Best Time to Buy and Sell Stock with Cooldown
Difficulty: Medium
Category: Stock Problems - Constraints

PROBLEM DESCRIPTION:
===================
You are given an array prices where prices[i] is the price of a given stock on the ith day.
Find the maximum profit you can achieve. You may complete as many transactions as you like 
(i.e., buy one and sell one share of the stock multiple times) with the following restrictions:
- After you sell your stock, you cannot buy stock on next day (i.e., cooldown 1 day).

Example 1:
Input: prices = [1,2,3,0,2]
Output: 3
Explanation: transactions = [buy, sell, cooldown, buy, sell]

Example 2:
Input: prices = [1]
Output: 0

Constraints:
- 1 <= prices.length <= 5000
- 0 <= prices[i] <= 1000
"""

def max_profit_state_machine_3_states(prices):
    """
    3-STATE MACHINE APPROACH:
    ========================
    held, sold, rest states with cooldown constraint.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if len(prices) <= 1:
        return 0
    
    # Three states:
    # held: currently holding stock
    # sold: just sold stock (must cooldown next day)
    # rest: resting (can buy next day)
    
    held = -prices[0]  # Bought stock on day 0
    sold = 0           # Sold stock (impossible on day 0, but initialize)
    rest = 0           # Resting on day 0
    
    for price in prices[1:]:
        # Update states (order matters to avoid using updated values)
        new_held = max(held, rest - price)     # Keep holding or buy (only if resting)
        new_sold = held + price                # Sell the stock we're holding
        new_rest = max(rest, sold)             # Keep resting or transition from sold
        
        held = new_held
        sold = new_sold
        rest = new_rest
    
    # We want to end without holding stock, so max of sold and rest
    return max(sold, rest)


def max_profit_dp_explicit(prices):
    """
    EXPLICIT DP APPROACH:
    ====================
    Use DP arrays to track all states explicitly.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(n) - DP arrays
    """
    n = len(prices)
    if n <= 1:
        return 0
    
    # dp[i][0] = max profit on day i when holding stock
    # dp[i][1] = max profit on day i when not holding stock (can buy next day)
    # dp[i][2] = max profit on day i when just sold (must cooldown)
    
    dp = [[0] * 3 for _ in range(n)]
    
    # Base case
    dp[0][0] = -prices[0]  # Buy on day 0
    dp[0][1] = 0           # Rest on day 0
    dp[0][2] = 0           # Can't sell on day 0
    
    for i in range(1, n):
        # Holding stock: either kept holding or bought today (only if was resting)
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] - prices[i])
        
        # Resting: either kept resting or transitioned from cooldown
        dp[i][1] = max(dp[i-1][1], dp[i-1][2])
        
        # Just sold: sold the stock we were holding
        dp[i][2] = dp[i-1][0] + prices[i]
    
    # Best profit when not holding stock at the end
    return max(dp[n-1][1], dp[n-1][2])


def max_profit_state_machine_2_states(prices):
    """
    2-STATE MACHINE APPROACH:
    =========================
    Simplified to hold and not_hold states.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if len(prices) <= 1:
        return 0
    
    # hold[i] = max profit when holding stock on day i
    # not_hold[i] = max profit when not holding stock on day i
    # prev_not_hold = not_hold[i-2] (for cooldown constraint)
    
    hold = -prices[0]
    not_hold = 0
    prev_not_hold = 0
    
    for price in prices[1:]:
        new_hold = max(hold, prev_not_hold - price)  # Buy only after cooldown
        new_not_hold = max(not_hold, hold + price)   # Sell if profitable
        
        # Update for next iteration
        prev_not_hold = not_hold
        hold = new_hold
        not_hold = new_not_hold
    
    return not_hold


def max_profit_memoization(prices):
    """
    MEMOIZATION APPROACH:
    ====================
    Top-down DP with memoization.
    
    Time Complexity: O(n) - each state computed once
    Space Complexity: O(n) - memoization table
    """
    if len(prices) <= 1:
        return 0
    
    memo = {}
    
    def dp(day, holding):
        """
        day: current day
        holding: 0 if not holding, 1 if holding, 2 if just sold (cooldown)
        """
        if day >= len(prices):
            return 0
        
        if (day, holding) in memo:
            return memo[(day, holding)]
        
        if holding == 0:  # Not holding, can buy
            # Either buy today or skip
            result = max(dp(day + 1, 1) - prices[day],  # Buy
                        dp(day + 1, 0))                # Skip
        elif holding == 1:  # Holding stock
            # Either sell today or keep holding
            result = max(dp(day + 1, 2) + prices[day],  # Sell (go to cooldown)
                        dp(day + 1, 1))                # Keep holding
        else:  # holding == 2, cooldown state
            # Must rest, can't buy
            result = dp(day + 1, 0)
        
        memo[(day, holding)] = result
        return result
    
    return dp(0, 0)  # Start not holding stock


def max_profit_with_transactions(prices):
    """
    TRACK ACTUAL TRANSACTIONS:
    ==========================
    Return profit and list of optimal transactions with cooldown.
    
    Time Complexity: O(n^2) - reconstruction
    Space Complexity: O(n) - DP table and tracking
    """
    n = len(prices)
    if n <= 1:
        return 0, []
    
    # Use explicit DP to track decisions
    dp = [[0] * 3 for _ in range(n)]
    actions = [[''] * 3 for _ in range(n)]  # Track what action led to this state
    
    dp[0][0] = -prices[0]
    dp[0][1] = 0
    dp[0][2] = 0
    actions[0][0] = 'buy'
    actions[0][1] = 'rest'
    actions[0][2] = 'none'
    
    for i in range(1, n):
        # Holding stock
        if dp[i-1][0] >= dp[i-1][1] - prices[i]:
            dp[i][0] = dp[i-1][0]
            actions[i][0] = 'hold'
        else:
            dp[i][0] = dp[i-1][1] - prices[i]
            actions[i][0] = 'buy'
        
        # Resting
        if dp[i-1][1] >= dp[i-1][2]:
            dp[i][1] = dp[i-1][1]
            actions[i][1] = 'rest'
        else:
            dp[i][1] = dp[i-1][2]
            actions[i][1] = 'cooldown_end'
        
        # Just sold
        dp[i][2] = dp[i-1][0] + prices[i]
        actions[i][2] = 'sell'
    
    # Find optimal final state
    if dp[n-1][1] >= dp[n-1][2]:
        final_profit = dp[n-1][1]
        final_state = 1
    else:
        final_profit = dp[n-1][2]
        final_state = 2
    
    # Reconstruct transactions
    transactions = []
    current_state = final_state
    buy_day = -1
    
    for i in range(n-1, -1, -1):
        action = actions[i][current_state]
        
        if action == 'sell':
            sell_day = i
            sell_price = prices[i]
            if buy_day != -1:
                profit = sell_price - prices[buy_day]
                transactions.append((buy_day, sell_day, prices[buy_day], sell_price, profit))
            current_state = 0  # Was holding before selling
        elif action == 'buy':
            buy_day = i
            current_state = 1  # Was resting before buying
        elif action == 'hold':
            current_state = 0  # Was holding
        elif action == 'cooldown_end':
            current_state = 2  # Was in cooldown
        elif action == 'rest':
            current_state = 1  # Was resting
    
    transactions.reverse()
    return final_profit, transactions


def max_profit_analysis(prices):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step state transitions with cooldown.
    """
    print(f"Stock Trading with Cooldown Analysis:")
    print(f"Prices: {prices}")
    
    if not prices:
        print("No prices available.")
        return 0
    
    n = len(prices)
    print(f"Days: {list(range(n))}")
    
    # 3-state machine analysis
    print(f"\n3-State machine evolution:")
    held = -prices[0]
    sold = 0
    rest = 0
    
    print(f"Day  Price  Held   Sold   Rest   Action")
    print(f"---  -----  -----  -----  -----  ------")
    print(f"  0  {prices[0]:5}  {held:5}  {sold:5}  {rest:5}  Buy")
    
    for i in range(1, len(prices)):
        price = prices[i]
        
        # Determine actions
        actions = []
        
        new_held = max(held, rest - price)
        if new_held == rest - price and new_held != held:
            actions.append("Buy")
        elif new_held == held:
            actions.append("Hold")
        
        new_sold = held + price
        if new_sold > max(held, rest):
            actions.append("Sell")
        
        new_rest = max(rest, sold)
        if new_rest == sold and new_rest != rest:
            actions.append("End Cooldown")
        elif new_rest == rest:
            actions.append("Rest")
        
        held = new_held
        sold = new_sold
        rest = new_rest
        
        action_str = ", ".join(actions) if actions else "Wait"
        print(f"{i:3}  {price:5}  {held:5}  {sold:5}  {rest:5}  {action_str}")
    
    final_profit = max(sold, rest)
    print(f"\nFinal maximum profit: {final_profit}")
    
    # Show optimal transactions
    profit, transactions = max_profit_with_transactions(prices)
    print(f"\nOptimal transactions with cooldown (total profit: {profit}):")
    
    if transactions:
        for i, (buy_day, sell_day, buy_price, sell_price, trans_profit) in enumerate(transactions):
            print(f"  {i+1}. Buy day {buy_day} (${buy_price}) → Sell day {sell_day} (${sell_price}) = ${trans_profit}")
            if i < len(transactions) - 1:  # Not the last transaction
                next_buy = transactions[i+1][0]
                cooldown_days = next_buy - sell_day - 1
                print(f"      Cooldown: {cooldown_days} day(s)")
    else:
        print(f"  No profitable transactions")
    
    # Compare with unlimited transactions (no cooldown)
    unlimited_profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            unlimited_profit += prices[i] - prices[i-1]
    
    print(f"\nComparison:")
    print(f"  With cooldown: {profit}")
    print(f"  Without cooldown: {unlimited_profit}")
    print(f"  Cooldown penalty: {unlimited_profit - profit}")
    
    return final_profit


def max_profit_variants():
    """
    COOLDOWN VARIANTS:
    =================
    Different cooldown scenarios and constraints.
    """
    
    def max_profit_k_day_cooldown(prices, k):
        """Cooldown for k days after selling"""
        n = len(prices)
        if n <= 1:
            return 0
        
        # hold[i] = max profit when holding stock on day i
        # sold[i][j] = max profit when sold j days ago (j = 1 to k)
        # rest[i] = max profit when resting (can buy)
        
        hold = -prices[0]
        sold = [0] * (k + 1)  # sold[0] unused, sold[1] = just sold, ..., sold[k] = k days ago
        rest = 0
        
        for price in prices[1:]:
            new_hold = max(hold, rest - price)
            new_sold = [0] * (k + 1)
            new_sold[1] = hold + price  # Just sold
            
            # Update cooldown states
            for j in range(2, k + 1):
                new_sold[j] = sold[j - 1]
            
            new_rest = max(rest, sold[k])  # Can buy after k-day cooldown
            
            hold = new_hold
            sold = new_sold
            rest = new_rest
        
        return max(hold, max(sold), rest)
    
    def max_profit_fee_and_cooldown(prices, fee):
        """Transaction fee and cooldown"""
        if len(prices) <= 1:
            return 0
        
        held = -prices[0]
        sold = 0
        rest = 0
        
        for price in prices[1:]:
            new_held = max(held, rest - price)
            new_sold = held + price - fee  # Subtract fee when selling
            new_rest = max(rest, sold)
            
            held = new_held
            sold = new_sold
            rest = new_rest
        
        return max(sold, rest)
    
    def max_profit_variable_cooldown(prices, cooldown_days):
        """Different cooldown periods for different days"""
        n = len(prices)
        if n <= 1 or not cooldown_days:
            return 0
        
        # Use general approach with dynamic cooldown tracking
        # This is more complex and would require state tracking for each possible cooldown
        # For simplicity, we'll use the 1-day cooldown approach
        return max_profit_state_machine_3_states(prices)
    
    # Test variants
    test_cases = [
        [1, 2, 3, 0, 2],
        [1, 2, 4, 2, 5, 7, 2, 4, 9, 0],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1]
    ]
    
    print("Cooldown Variants:")
    print("=" * 50)
    
    for prices in test_cases:
        print(f"\nPrices: {prices}")
        
        standard_cooldown = max_profit_state_machine_3_states(prices)
        print(f"1-day cooldown: {standard_cooldown}")
        
        two_day_cooldown = max_profit_k_day_cooldown(prices, 2)
        print(f"2-day cooldown: {two_day_cooldown}")
        
        fee_and_cooldown = max_profit_fee_and_cooldown(prices, 1)
        print(f"$1 fee + cooldown: {fee_and_cooldown}")
        
        # Compare with unlimited (no cooldown)
        unlimited = sum(max(0, prices[i] - prices[i-1]) for i in range(1, len(prices)))
        print(f"No cooldown: {unlimited}")


# Test cases
def test_max_profit_cooldown():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1, 2, 3, 0, 2], 3),
        ([1], 0),
        ([1, 2], 1),
        ([2, 1], 0),
        ([1, 2, 3, 4, 5], 4),
        ([5, 4, 3, 2, 1], 0),
        ([2, 1, 4, 5, 2, 9, 7], 11),
        ([1, 4, 2], 3),
        ([], 0)
    ]
    
    print("Testing Best Time to Buy and Sell Stock with Cooldown:")
    print("=" * 70)
    
    for i, (prices, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Prices: {prices}")
        print(f"Expected: {expected}")
        
        state_3 = max_profit_state_machine_3_states(prices)
        dp_explicit = max_profit_dp_explicit(prices)
        state_2 = max_profit_state_machine_2_states(prices)
        memoization = max_profit_memoization(prices)
        
        print(f"3-State Machine:  {state_3:>3} {'✓' if state_3 == expected else '✗'}")
        print(f"Explicit DP:      {dp_explicit:>3} {'✓' if dp_explicit == expected else '✗'}")
        print(f"2-State Machine:  {state_2:>3} {'✓' if state_2 == expected else '✗'}")
        print(f"Memoization:      {memoization:>3} {'✓' if memoization == expected else '✗'}")
        
        # Show transactions for small cases
        if len(prices) <= 8:
            profit, transactions = max_profit_with_transactions(prices)
            print(f"Transactions: {len(transactions)} total, profit: {profit}")
            for j, (buy_day, sell_day, buy_price, sell_price, trans_profit) in enumerate(transactions):
                print(f"  {j+1}. Day {buy_day}→{sell_day}: ${buy_price}→${sell_price} = +${trans_profit}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    max_profit_analysis([1, 2, 3, 0, 2])
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    max_profit_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. COOLDOWN CONSTRAINT: Must wait 1 day after selling")
    print("2. THREE STATES: Holding, just sold (cooldown), resting")
    print("3. STATE DEPENDENCIES: Selling leads to forced cooldown")
    print("4. OPTIMAL TIMING: May skip profitable trades due to cooldown")
    print("5. CONSTRAINT PENALTY: Cooldown reduces overall profit")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Real Trading: Settlement periods and restrictions")
    print("• Resource Management: Recovery time after usage")
    print("• System Design: Rate limiting and throttling")
    print("• Game Theory: Cooldown mechanics in strategy")
    print("• Economics: Market restrictions and waiting periods")


if __name__ == "__main__":
    test_max_profit_cooldown()


"""
BEST TIME TO BUY AND SELL STOCK WITH COOLDOWN - CONSTRAINT MODELING:
====================================================================

This problem adds a temporal constraint to unlimited stock trading:
- Can trade as many times as desired
- After selling, must wait 1 day before buying again (cooldown)
- Demonstrates how constraints change optimal strategies
- Shows state machine design for temporal dependencies

KEY INSIGHTS:
============
1. **COOLDOWN CONSTRAINT**: Must wait 1 day after selling before buying
2. **THREE STATES**: Holding, just sold (cooldown), resting (can buy)
3. **TEMPORAL DEPENDENCY**: Current action affects future availability
4. **CONSTRAINT PENALTY**: Cooldown may force suboptimal timing
5. **STATE TRANSITIONS**: Clear rules govern state changes

ALGORITHM APPROACHES:
====================

1. **3-State Machine**: O(n) time, O(1) space
   - Model holding, sold (cooldown), rest states
   - Most intuitive approach

2. **2-State Machine**: O(n) time, O(1) space
   - Simplified to hold/not-hold with cooldown tracking
   - More compact representation

3. **Explicit DP**: O(n) time, O(n) space
   - Use DP arrays for educational clarity
   - Shows all intermediate states

4. **Memoization**: O(n) time, O(n) space
   - Top-down approach with state caching
   - Natural recursive formulation

CORE 3-STATE MACHINE:
====================
```python
held = -prices[0]    # Currently holding stock
sold = 0             # Just sold (in cooldown)
rest = 0             # Resting (can buy)

for price in prices[1:]:
    new_held = max(held, rest - price)    # Hold or buy from rest
    new_sold = held + price               # Sell current stock
    new_rest = max(rest, sold)            # Rest or end cooldown
    
    held, sold, rest = new_held, new_sold, new_rest

return max(sold, rest)  # Don't end holding stock
```

STATE TRANSITION DIAGRAM:
========================
```
    REST ←------ SOLD
     ↓            ↑
    BUY          SELL
     ↓            ↑
    HELD -------> HELD
         HOLD
```

**State Meanings**:
- **HELD**: Currently holding stock, can sell or continue holding
- **SOLD**: Just sold stock, must cooldown (automatic transition to REST)
- **REST**: Not holding stock, can buy or continue resting

**Transition Rules**:
- HELD → HELD: Continue holding (no action)
- HELD → SOLD: Sell stock (forced cooldown next day)
- SOLD → REST: Cooldown period ends (automatic, no choice)
- REST → REST: Continue resting (no action)
- REST → HELD: Buy stock

2-STATE SIMPLIFICATION:
======================
```python
hold = -prices[0]
not_hold = 0
prev_not_hold = 0  # not_hold from 2 days ago

for price in prices[1:]:
    new_hold = max(hold, prev_not_hold - price)  # Buy after cooldown
    new_not_hold = max(not_hold, hold + price)   # Sell if profitable
    
    prev_not_hold = not_hold
    hold = new_hold
    not_hold = new_not_hold

return not_hold
```

EXPLICIT DP FORMULATION:
=======================
```python
dp[i][0] = max profit on day i when holding stock
dp[i][1] = max profit on day i when resting (can buy)
dp[i][2] = max profit on day i when in cooldown (just sold)

Recurrence:
dp[i][0] = max(dp[i-1][0], dp[i-1][1] - prices[i])  # Hold or buy
dp[i][1] = max(dp[i-1][1], dp[i-1][2])              # Rest or end cooldown
dp[i][2] = dp[i-1][0] + prices[i]                   # Sell stock
```

CONSTRAINT IMPACT ANALYSIS:
===========================
Cooldown constraint affects strategy in several ways:

1. **Timing Shifts**: May delay optimal buy opportunities
2. **Profit Reduction**: Missing immediate re-entry after profitable sales
3. **Strategy Change**: Prefer longer holding periods
4. **Pattern Breaking**: Can't capture every small fluctuation

Example: [1, 3, 2, 4]
- **Without cooldown**: Buy 1→Sell 3, Buy 2→Sell 4 = 2 + 2 = 4
- **With cooldown**: Buy 1→Sell 3, Cooldown, Buy 2→Sell 4 = 2 + 2 = 4 (same)

Example: [1, 2, 1, 3]
- **Without cooldown**: Buy 1→Sell 2, Buy 1→Sell 3 = 1 + 2 = 3
- **With cooldown**: Buy 1→Sell 2, Cooldown, Buy 1→Sell 3 = 1 + 2 = 3 (same)

Example: [1, 3, 1, 2]
- **Without cooldown**: Buy 1→Sell 3, Buy 1→Sell 2 = 2 + 1 = 3
- **With cooldown**: Buy 1→Sell 3, Cooldown, Buy 1→Sell 2 = 2 + 1 = 3 (same)

But: [1, 2, 3, 1, 2]
- **Without cooldown**: Buy 1→Sell 2, Buy 2→Sell 3, Buy 1→Sell 2 = 1 + 1 + 1 = 3
- **With cooldown**: Buy 1→Sell 3 = 2 (forced to hold longer due to cooldown)

COMPLEXITY ANALYSIS:
===================
| Approach        | Time | Space | Advantages           |
|-----------------|------|-------|----------------------|
| 3-State Machine | O(n) | O(1)  | Intuitive modeling   |
| 2-State Machine | O(n) | O(1)  | Compact              |
| Explicit DP     | O(n) | O(n)  | Educational clarity  |
| Memoization     | O(n) | O(n)  | Natural recursion    |

EDGE CASES:
==========
- **Single day**: Return 0 (can't trade)
- **Two days**: If profitable, buy day 0, sell day 1
- **Decreasing prices**: Return 0 (no profitable trades)
- **Increasing prices**: Hold from start to end (single transaction optimal)
- **Alternating pattern**: Cooldown may force longer holds

RELATIONSHIP TO OTHER PROBLEMS:
==============================
- **Stock II (122)**: Remove cooldown constraint
- **Stock with Fee (714)**: Add transaction cost
- **Stock III (123)**: Add transaction count limit
- **Stock IV (188)**: Generalize transaction limits

EXTENSIONS:
==========
- **k-day cooldown**: Extend cooldown period to k days
- **Variable cooldown**: Different cooldown based on profit/loss
- **Partial cooldown**: Can buy limited amount during cooldown
- **Warming up**: Gradual transition out of cooldown

PRACTICAL APPLICATIONS:
======================
- **Real Trading**: Settlement periods (T+2, T+3)
- **Resource Management**: Recovery time after resource usage
- **System Design**: Rate limiting and API throttling
- **Game Mechanics**: Cooldown periods for abilities
- **Economics**: Market restrictions and waiting periods

MATHEMATICAL PROPERTIES:
========================
- **Monotonicity**: Removing cooldown never decreases profit
- **Convergence**: As cooldown period increases, approaches single transaction
- **Optimality**: Greedy approach within constraints is optimal
- **State Independence**: Current state fully determines future options

This problem elegantly demonstrates how temporal constraints require
careful state modeling and can significantly impact optimal strategies
in dynamic programming problems.
"""
