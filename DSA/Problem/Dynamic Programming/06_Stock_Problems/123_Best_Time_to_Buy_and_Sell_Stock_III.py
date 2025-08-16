"""
LeetCode 123: Best Time to Buy and Sell Stock III
Difficulty: Hard
Category: Stock Problems - Limited Transactions

PROBLEM DESCRIPTION:
===================
You are given an array prices where prices[i] is the price of a given stock on the ith day.
Find the maximum profit you can achieve. You may complete at most two transactions.
Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

Example 1:
Input: prices = [3,3,5,0,0,3,1,4]
Output: 6
Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.
Total profit is 3 + 3 = 6.

Example 2:
Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are engaging multiple transactions at the same time.

Example 3:
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are completed, so max profit = 0.

Constraints:
- 1 <= prices.length <= 10^5
- 0 <= prices[i] <= 10^5
"""

def max_profit_brute_force(prices):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible combinations of at most 2 transactions.
    
    Time Complexity: O(n^4) - four nested loops
    Space Complexity: O(1) - constant space
    """
    n = len(prices)
    if n <= 1:
        return 0
    
    max_profit = 0
    
    # Try all possible first transactions
    for buy1 in range(n):
        for sell1 in range(buy1 + 1, n):
            profit1 = prices[sell1] - prices[buy1]
            
            # Try all possible second transactions after first
            for buy2 in range(sell1 + 1, n):
                for sell2 in range(buy2 + 1, n):
                    profit2 = prices[sell2] - prices[buy2]
                    max_profit = max(max_profit, profit1 + profit2)
            
            # Also consider only one transaction
            max_profit = max(max_profit, profit1)
    
    return max_profit


def max_profit_divide_and_conquer(prices):
    """
    DIVIDE AND CONQUER APPROACH:
    ============================
    Split array and find best transaction in each part.
    
    Time Complexity: O(n^2) - try all split points
    Space Complexity: O(n) - precomputed arrays
    """
    n = len(prices)
    if n <= 1:
        return 0
    
    # Precompute max profit for single transaction ending at or before each day
    left_profits = [0] * n
    min_price = prices[0]
    
    for i in range(1, n):
        min_price = min(min_price, prices[i])
        left_profits[i] = max(left_profits[i-1], prices[i] - min_price)
    
    # Precompute max profit for single transaction starting at or after each day
    right_profits = [0] * n
    max_price = prices[n-1]
    
    for i in range(n-2, -1, -1):
        max_price = max(max_price, prices[i])
        right_profits[i] = max(right_profits[i+1], max_price - prices[i])
    
    # Find best split point
    max_profit = 0
    for i in range(n):
        max_profit = max(max_profit, left_profits[i] + right_profits[i])
    
    return max_profit


def max_profit_state_machine(prices):
    """
    STATE MACHINE APPROACH:
    ======================
    Track 4 states: first buy, first sell, second buy, second sell.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if len(prices) <= 1:
        return 0
    
    # Four states for at most 2 transactions
    first_buy = -prices[0]   # Cost after first buy
    first_sell = 0           # Profit after first sell
    second_buy = -prices[0]  # Cost after second buy
    second_sell = 0          # Profit after second sell
    
    for price in prices[1:]:
        # Update states in reverse order to avoid dependency
        second_sell = max(second_sell, second_buy + price)
        second_buy = max(second_buy, first_sell - price)
        first_sell = max(first_sell, first_buy + price)
        first_buy = max(first_buy, -price)
    
    return second_sell


def max_profit_dp_2d(prices):
    """
    2D DP APPROACH:
    ==============
    dp[i][j] = max profit using at most i transactions by day j.
    
    Time Complexity: O(n) - optimized for k=2
    Space Complexity: O(n) - DP array
    """
    n = len(prices)
    if n <= 1:
        return 0
    
    # For at most 2 transactions
    # dp[t][i] = max profit using at most t transactions by day i
    dp = [[0] * n for _ in range(3)]  # 0, 1, 2 transactions
    
    for t in range(1, 3):  # 1 or 2 transactions
        max_diff = -prices[0]  # Maximum of (dp[t-1][j] - prices[j]) for j < i
        
        for i in range(1, n):
            # Either don't trade on day i, or sell on day i
            dp[t][i] = max(dp[t][i-1], prices[i] + max_diff)
            
            # Update max_diff for next iteration
            max_diff = max(max_diff, dp[t-1][i] - prices[i])
    
    return dp[2][n-1]


def max_profit_optimized_dp(prices):
    """
    SPACE OPTIMIZED DP:
    ==================
    Use O(1) space by tracking only necessary states.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if len(prices) <= 1:
        return 0
    
    # buy[i] = max profit after buying in transaction i
    # sell[i] = max profit after selling in transaction i
    buy1 = buy2 = -prices[0]
    sell1 = sell2 = 0
    
    for price in prices[1:]:
        # Second transaction
        sell2 = max(sell2, buy2 + price)
        buy2 = max(buy2, sell1 - price)
        
        # First transaction
        sell1 = max(sell1, buy1 + price)
        buy1 = max(buy1, -price)
    
    return sell2


def max_profit_with_transactions(prices):
    """
    TRACK ACTUAL TRANSACTIONS:
    ==========================
    Return profit and optimal transaction details.
    
    Time Complexity: O(n^2) - divide and conquer with reconstruction
    Space Complexity: O(n) - transaction tracking
    """
    n = len(prices)
    if n <= 1:
        return 0, []
    
    # Find optimal split point
    left_profits = [0] * n
    left_transactions = [None] * n
    min_price = prices[0]
    min_day = 0
    
    for i in range(1, n):
        if prices[i] < min_price:
            min_price = prices[i]
            min_day = i
        
        profit = prices[i] - min_price
        if profit > left_profits[i-1]:
            left_profits[i] = profit
            left_transactions[i] = (min_day, i, min_price, prices[i], profit)
        else:
            left_profits[i] = left_profits[i-1]
            left_transactions[i] = left_transactions[i-1]
    
    right_profits = [0] * n
    right_transactions = [None] * n
    max_price = prices[n-1]
    max_day = n-1
    
    for i in range(n-2, -1, -1):
        if prices[i] > max_price:
            max_price = prices[i]
            max_day = i
        
        profit = max_price - prices[i]
        if profit > right_profits[i+1]:
            right_profits[i] = profit
            right_transactions[i] = (i, max_day, prices[i], max_price, profit)
        else:
            right_profits[i] = right_profits[i+1]
            right_transactions[i] = right_transactions[i+1]
    
    # Find best split
    max_profit = 0
    best_left = None
    best_right = None
    
    for i in range(n):
        total_profit = left_profits[i] + right_profits[i]
        if total_profit > max_profit:
            max_profit = total_profit
            best_left = left_transactions[i]
            best_right = right_transactions[i]
    
    transactions = []
    if best_left and best_left[4] > 0:  # profit > 0
        transactions.append(best_left)
    if best_right and best_right[4] > 0:  # profit > 0
        transactions.append(best_right)
    
    return max_profit, transactions


def max_profit_analysis(prices):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation and state transitions.
    """
    print(f"At Most 2 Transactions Stock Analysis:")
    print(f"Prices: {prices}")
    
    if not prices:
        print("No prices available.")
        return 0
    
    n = len(prices)
    print(f"Days: {list(range(n))}")
    
    # State machine analysis
    print(f"\nState machine evolution:")
    first_buy = -prices[0]
    first_sell = 0
    second_buy = -prices[0]
    second_sell = 0
    
    print(f"Day  Price  1stBuy  1stSell  2ndBuy  2ndSell  Actions")
    print(f"---  -----  ------  -------  ------  -------  -------")
    print(f"  0  {prices[0]:5}  {first_buy:6}  {first_sell:7}  {second_buy:6}  {second_sell:7}  Initial")
    
    for i in range(1, len(prices)):
        price = prices[i]
        
        # Track what changes
        actions = []
        
        new_second_sell = max(second_sell, second_buy + price)
        if new_second_sell > second_sell:
            actions.append("2nd Sell")
        
        new_second_buy = max(second_buy, first_sell - price)
        if new_second_buy > second_buy:
            actions.append("2nd Buy")
        
        new_first_sell = max(first_sell, first_buy + price)
        if new_first_sell > first_sell:
            actions.append("1st Sell")
        
        new_first_buy = max(first_buy, -price)
        if new_first_buy > first_buy:
            actions.append("1st Buy")
        
        # Update states
        second_sell = new_second_sell
        second_buy = new_second_buy
        first_sell = new_first_sell
        first_buy = new_first_buy
        
        action_str = ", ".join(actions) if actions else "Hold"
        print(f"{i:3}  {price:5}  {first_buy:6}  {first_sell:7}  {second_buy:6}  {second_sell:7}  {action_str}")
    
    print(f"\nFinal maximum profit: {second_sell}")
    
    # Show optimal transactions
    profit, transactions = max_profit_with_transactions(prices)
    print(f"\nOptimal transactions (total profit: {profit}):")
    
    if transactions:
        for i, (buy_day, sell_day, buy_price, sell_price, trans_profit) in enumerate(transactions):
            print(f"  {i+1}. Buy day {buy_day} (${buy_price}) → Sell day {sell_day} (${sell_price}) = ${trans_profit}")
    else:
        print(f"  No profitable transactions")
    
    # Divide and conquer analysis
    print(f"\nDivide and conquer analysis:")
    left_profits = [0] * n
    min_price = prices[0]
    
    for i in range(1, n):
        min_price = min(min_price, prices[i])
        left_profits[i] = max(left_profits[i-1], prices[i] - min_price)
    
    right_profits = [0] * n
    max_price = prices[n-1]
    
    for i in range(n-2, -1, -1):
        max_price = max(max_price, prices[i])
        right_profits[i] = max(right_profits[i+1], max_price - prices[i])
    
    print(f"Day:         {list(range(n))}")
    print(f"Left profit: {left_profits}")
    print(f"Right profit: {right_profits}")
    print(f"Combined:    {[left_profits[i] + right_profits[i] for i in range(n)]}")
    
    best_split = max(range(n), key=lambda i: left_profits[i] + right_profits[i])
    print(f"Best split at day {best_split}: {left_profits[best_split]} + {right_profits[best_split]} = {left_profits[best_split] + right_profits[best_split]}")
    
    return second_sell


def max_profit_variants():
    """
    AT MOST 2 TRANSACTIONS VARIANTS:
    ================================
    Different scenarios and constraints.
    """
    
    def max_profit_exactly_2_transactions(prices):
        """Require exactly 2 transactions"""
        n = len(prices)
        if n < 4:  # Need at least 4 days for 2 transactions
            return 0
        
        max_profit = 0
        
        # Must complete exactly 2 transactions
        for i in range(1, n-2):  # Split point
            # First transaction: find best in [0, i]
            left_profit = 0
            min_price = prices[0]
            for j in range(1, i+1):
                min_price = min(min_price, prices[j])
                left_profit = max(left_profit, prices[j] - min_price)
            
            # Second transaction: find best in [i+1, n-1]
            right_profit = 0
            max_price = prices[i+1]
            for j in range(i+2, n):
                max_price = max(max_price, prices[j])
                right_profit = max(right_profit, max_price - prices[j])
            
            if left_profit > 0 and right_profit > 0:
                max_profit = max(max_profit, left_profit + right_profit)
        
        return max_profit
    
    def max_profit_with_gap(prices, min_gap):
        """Minimum gap between transactions"""
        n = len(prices)
        if n <= 1:
            return 0
        
        max_profit = 0
        
        # Try all possible first transactions
        for buy1 in range(n):
            for sell1 in range(buy1 + 1, n):
                profit1 = prices[sell1] - prices[buy1]
                
                # Second transaction must start after gap
                for buy2 in range(sell1 + min_gap + 1, n):
                    for sell2 in range(buy2 + 1, n):
                        profit2 = prices[sell2] - prices[buy2]
                        max_profit = max(max_profit, profit1 + profit2)
                
                # Also consider only one transaction
                max_profit = max(max_profit, profit1)
        
        return max_profit
    
    def max_profit_with_constraints(prices, max_buy_price, min_sell_price):
        """Constraints on buy/sell prices"""
        if len(prices) <= 1:
            return 0
        
        first_buy = -float('inf')
        first_sell = 0
        second_buy = -float('inf')
        second_sell = 0
        
        for price in prices:
            if price <= max_buy_price:
                second_sell = max(second_sell, second_buy + price)
                second_buy = max(second_buy, first_sell - price)
                first_sell = max(first_sell, first_buy + price)
                first_buy = max(first_buy, -price)
            
            if price >= min_sell_price:
                second_sell = max(second_sell, second_buy + price)
                first_sell = max(first_sell, first_buy + price)
        
        return second_sell
    
    # Test variants
    test_cases = [
        [3, 3, 5, 0, 0, 3, 1, 4],
        [1, 2, 3, 4, 5],
        [7, 6, 4, 3, 1],
        [1, 2, 4, 2, 5, 7, 2, 4, 9, 0]
    ]
    
    print("At Most 2 Transactions Variants:")
    print("=" * 50)
    
    for prices in test_cases:
        print(f"\nPrices: {prices}")
        
        at_most_2 = max_profit_state_machine(prices)
        print(f"At most 2 transactions: {at_most_2}")
        
        exactly_2 = max_profit_exactly_2_transactions(prices)
        print(f"Exactly 2 transactions: {exactly_2}")
        
        gap_2 = max_profit_with_gap(prices, 2)
        print(f"With 2-day gap: {gap_2}")
        
        constrained = max_profit_with_constraints(prices, max(prices)//2, min(prices)*2)
        print(f"With price constraints: {constrained}")


# Test cases
def test_max_profit_iii():
    """Test all implementations with various inputs"""
    test_cases = [
        ([3, 3, 5, 0, 0, 3, 1, 4], 6),
        ([1, 2, 3, 4, 5], 4),
        ([7, 6, 4, 3, 1], 0),
        ([1, 2, 1, 2], 2),
        ([2, 1, 2, 0, 1], 2),
        ([1, 4, 2], 3),
        ([2, 1, 4, 9], 8),
        ([1], 0),
        ([], 0)
    ]
    
    print("Testing Best Time to Buy and Sell Stock III Solutions:")
    print("=" * 70)
    
    for i, (prices, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Prices: {prices}")
        print(f"Expected: {expected}")
        
        # Skip brute force for large inputs
        if len(prices) <= 6:
            try:
                brute_force = max_profit_brute_force(prices)
                print(f"Brute Force:      {brute_force:>3} {'✓' if brute_force == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        divide_conquer = max_profit_divide_and_conquer(prices)
        state_machine = max_profit_state_machine(prices)
        dp_2d = max_profit_dp_2d(prices)
        optimized_dp = max_profit_optimized_dp(prices)
        
        print(f"Divide & Conquer: {divide_conquer:>3} {'✓' if divide_conquer == expected else '✗'}")
        print(f"State Machine:    {state_machine:>3} {'✓' if state_machine == expected else '✗'}")
        print(f"2D DP:            {dp_2d:>3} {'✓' if dp_2d == expected else '✗'}")
        print(f"Optimized DP:     {optimized_dp:>3} {'✓' if optimized_dp == expected else '✗'}")
        
        # Show transactions for small cases
        if len(prices) <= 10:
            profit, transactions = max_profit_with_transactions(prices)
            print(f"Transactions: {len(transactions)} total, profit: {profit}")
            for j, (buy_day, sell_day, buy_price, sell_price, trans_profit) in enumerate(transactions):
                print(f"  {j+1}. Day {buy_day}→{sell_day}: ${buy_price}→${sell_price} = +${trans_profit}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    max_profit_analysis([3, 3, 5, 0, 0, 3, 1, 4])
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    max_profit_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. STATE MACHINE: 4 states for 2 transactions")
    print("2. DIVIDE & CONQUER: Split array optimally")
    print("3. DP OPTIMIZATION: Reduce to O(n) time and O(1) space")
    print("4. TRANSACTION LIMIT: Must choose best 2 opportunities")
    print("5. ORDER DEPENDENCY: States must be updated carefully")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Limited Trading: Portfolio with transaction limits")
    print("• Resource Allocation: Two optimal investment windows")
    print("• Algorithm Design: Constrained optimization problems")
    print("• Financial Planning: Multiple entry/exit strategies")
    print("• Risk Management: Limiting exposure through transaction caps")


if __name__ == "__main__":
    test_max_profit_iii()


"""
BEST TIME TO BUY AND SELL STOCK III - CONSTRAINED OPTIMIZATION:
===============================================================

This problem introduces transaction limits to stock trading:
- At most 2 transactions allowed (buy-sell pairs)
- Must sell before buying again
- Choose the 2 most profitable opportunities from unlimited possibilities
- Demonstrates transition from greedy to dynamic programming

KEY INSIGHTS:
============
1. **TRANSACTION LIMITS**: Must choose best 2 out of many opportunities
2. **STATE MACHINE**: 4 states tracking transaction progress
3. **DIVIDE & CONQUER**: Split timeline optimally between transactions
4. **DP OPTIMIZATION**: Reduce from O(k×n) to O(n) for k=2
5. **ORDER MATTERS**: State updates must maintain dependencies

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(n⁴) time, O(1) space
   - Try all combinations of 2 transactions
   - Only viable for tiny inputs

2. **Divide & Conquer**: O(n²) time, O(n) space
   - Try all split points between transactions
   - Precompute optimal single transactions

3. **State Machine**: O(n) time, O(1) space
   - Track 4 states: 1st buy, 1st sell, 2nd buy, 2nd sell
   - Optimal solution

4. **2D DP**: O(n) time, O(n) space
   - General k-transaction framework specialized for k=2
   - Educational value

CORE STATE MACHINE:
==================
```python
# Four states for at most 2 transactions
first_buy = -prices[0]    # After first purchase
first_sell = 0            # After first sale
second_buy = -prices[0]   # After second purchase  
second_sell = 0           # After second sale

for price in prices[1:]:
    # Update in reverse order to avoid dependencies
    second_sell = max(second_sell, second_buy + price)
    second_buy = max(second_buy, first_sell - price)
    first_sell = max(first_sell, first_buy + price)
    first_buy = max(first_buy, -price)

return second_sell
```

STATE TRANSITIONS:
=================
```
State Machine Flow:
START → first_buy → first_sell → second_buy → second_sell → END

Transition Equations:
- first_buy[i] = max(first_buy[i-1], -prices[i])
- first_sell[i] = max(first_sell[i-1], first_buy[i-1] + prices[i])
- second_buy[i] = max(second_buy[i-1], first_sell[i-1] - prices[i])
- second_sell[i] = max(second_sell[i-1], second_buy[i-1] + prices[i])
```

DIVIDE & CONQUER APPROACH:
=========================
Split the timeline and optimize each part:
```python
# Precompute best single transaction ending at or before day i
left_profits[i] = max profit using prices[0:i+1]

# Precompute best single transaction starting at or after day i  
right_profits[i] = max profit using prices[i:n]

# Find optimal split point
max_profit = max(left_profits[i] + right_profits[i]) for all i
```

DP GENERALIZATION:
=================
This extends to the general k-transactions problem:
```python
# dp[t][i] = max profit using at most t transactions by day i
for t in range(1, k+1):
    max_diff = -prices[0]
    for i in range(1, n):
        dp[t][i] = max(dp[t][i-1], prices[i] + max_diff)
        max_diff = max(max_diff, dp[t-1][i] - prices[i])
```

For k=2, this reduces to our optimized solution.

COMPLEXITY ANALYSIS:
===================
| Approach        | Time  | Space | Trade-offs           |
|-----------------|-------|-------|----------------------|
| Brute Force     | O(n⁴) | O(1)  | Exhaustive search    |
| Divide/Conquer  | O(n²) | O(n)  | Intuitive split     |
| State Machine   | O(n)  | O(1)  | Optimal solution    |
| 2D DP          | O(n)  | O(n)  | General framework   |

WHY GREEDY FAILS:
================
Unlike unlimited transactions, we can't take every profitable opportunity:
- Must choose the 2 BEST opportunities
- Taking a small profit might prevent a larger one
- Need global optimization, not local greedy choices

Example: [1, 5, 3, 6, 4]
- Greedy (unlimited): (1→5) + (3→6) = 4 + 3 = 7
- Optimal (≤2): (1→6) = 5 (single transaction is better)

MATHEMATICAL PROPERTIES:
=======================
- **Optimal Substructure**: Optimal solution contains optimal subsolutions
- **Overlapping Subproblems**: Same states computed multiple times
- **Monotonicity**: Profit is non-decreasing as more transactions allowed
- **Constraint Binding**: Solution quality depends on how binding the limit is

EDGE CASES:
==========
- **n ≤ 1**: Return 0 (can't trade)
- **Decreasing prices**: Return 0 (no profit possible)
- **Increasing prices**: Single transaction optimal (use only 1 of 2 allowed)
- **k ≥ n/2**: Equivalent to unlimited transactions (greedy optimal)

RELATIONSHIP TO OTHER PROBLEMS:
==============================
- **Stock I (121)**: k = 1 (single transaction)
- **Stock II (122)**: k = ∞ (unlimited transactions)
- **Stock IV (188)**: General k transactions
- **Stock with Cooldown (309)**: Unlimited with constraints
- **Stock with Fee (714)**: Unlimited with costs

PRACTICAL CONSIDERATIONS:
========================
- **Transaction Costs**: Real trading has fees
- **Market Impact**: Large orders affect prices
- **Liquidity**: Not all shares may be available
- **Risk Management**: Concentration risk with limited transactions
- **Tax Implications**: Different treatment for different holding periods

OPTIMIZATION TECHNIQUES:
=======================
1. **State Compression**: 4 states instead of full DP table
2. **Order of Updates**: Reverse order prevents dependencies
3. **Space Efficiency**: O(1) space vs O(kn) general solution
4. **Early Termination**: Stop if remaining days can't improve

EXTENSIONS:
==========
- **Exactly k transactions**: Modify state machine
- **Different transaction costs**: Add fees to state transitions
- **Time gaps**: Minimum time between transactions
- **Volume constraints**: Limited shares per transaction

This problem elegantly demonstrates the transition from greedy algorithms
(unlimited transactions) to dynamic programming (constrained optimization),
showing how constraints fundamentally change the solution approach.
"""
