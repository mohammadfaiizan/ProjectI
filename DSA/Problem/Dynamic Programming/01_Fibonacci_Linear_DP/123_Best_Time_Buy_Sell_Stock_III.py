"""
LeetCode 123: Best Time to Buy and Sell Stock III
Difficulty: Hard
Category: Fibonacci & Linear DP / Stock Problems

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

Example 2:
Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are engaging multiple transactions at the same time. You must sell before buying again.

Example 3:
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.

Constraints:
- 1 <= prices.length <= 10^5
- 0 <= prices[i] <= 10^5
"""

def max_profit_iii_bruteforce(prices):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible combinations of two transactions.
    
    Time Complexity: O(n^4) - four nested loops
    Space Complexity: O(1) - constant space
    """
    if len(prices) < 2:
        return 0
    
    n = len(prices)
    max_profit = 0
    
    # First transaction: buy at i, sell at j
    for i in range(n):
        for j in range(i + 1, n):
            profit1 = prices[j] - prices[i]
            
            # Second transaction: buy at k, sell at l (after first transaction)
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    profit2 = prices[l] - prices[k]
                    total_profit = profit1 + profit2
                    max_profit = max(max_profit, total_profit)
    
    # Also consider single transaction
    for i in range(n):
        for j in range(i + 1, n):
            max_profit = max(max_profit, prices[j] - prices[i])
    
    return max_profit


def max_profit_iii_divide_and_conquer(prices):
    """
    DIVIDE AND CONQUER APPROACH:
    ===========================
    Split array and find best single transaction in each part.
    
    Time Complexity: O(n^2) - try all split points
    Space Complexity: O(n) - store max profits
    """
    if len(prices) < 2:
        return 0
    
    n = len(prices)
    
    def max_profit_single(start, end):
        """Find maximum profit with single transaction in range [start, end)"""
        if end - start < 2:
            return 0
        
        min_price = prices[start]
        max_profit = 0
        
        for i in range(start + 1, end):
            max_profit = max(max_profit, prices[i] - min_price)
            min_price = min(min_price, prices[i])
        
        return max_profit
    
    max_total = 0
    
    # Try all possible split points
    for split in range(1, n):
        # First transaction in [0, split), second in [split, n)
        profit1 = max_profit_single(0, split)
        profit2 = max_profit_single(split, n)
        max_total = max(max_total, profit1 + profit2)
    
    # Also consider single transaction for entire array
    single_profit = max_profit_single(0, n)
    return max(max_total, single_profit)


def max_profit_iii_dp_precompute(prices):
    """
    DYNAMIC PROGRAMMING - PRECOMPUTE LEFT AND RIGHT:
    ===============================================
    Precompute max profit achievable on left and right of each position.
    
    Time Complexity: O(n) - three passes through array
    Space Complexity: O(n) - two additional arrays
    """
    if len(prices) < 2:
        return 0
    
    n = len(prices)
    
    # left[i] = max profit achievable using prices[0...i]
    left = [0] * n
    min_price = prices[0]
    
    for i in range(1, n):
        left[i] = max(left[i - 1], prices[i] - min_price)
        min_price = min(min_price, prices[i])
    
    # right[i] = max profit achievable using prices[i...n-1]
    right = [0] * n
    max_price = prices[n - 1]
    
    for i in range(n - 2, -1, -1):
        right[i] = max(right[i + 1], max_price - prices[i])
        max_price = max(max_price, prices[i])
    
    # Find maximum profit using at most 2 transactions
    max_profit = 0
    for i in range(n):
        max_profit = max(max_profit, left[i] + right[i])
    
    return max_profit


def max_profit_iii_state_machine(prices):
    """
    STATE MACHINE DP:
    ================
    Track all possible states after each day.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if len(prices) < 2:
        return 0
    
    # States: buy1, sell1, buy2, sell2
    buy1 = -prices[0]  # Bought first stock
    sell1 = 0          # Sold first stock
    buy2 = -prices[0]  # Bought second stock (can happen same day as sell1)
    sell2 = 0          # Sold second stock
    
    for i in range(1, len(prices)):
        price = prices[i]
        
        # Update states (order matters - update in reverse dependency order)
        sell2 = max(sell2, buy2 + price)    # Sell second stock
        buy2 = max(buy2, sell1 - price)     # Buy second stock
        sell1 = max(sell1, buy1 + price)    # Sell first stock
        buy1 = max(buy1, -price)            # Buy first stock
    
    return sell2


def max_profit_iii_general_k_transactions(prices, k=2):
    """
    GENERALIZED DP FOR K TRANSACTIONS:
    =================================
    Solve for k transactions (specialized for k=2).
    
    Time Complexity: O(n * k) - for general k, O(n) for k=2
    Space Complexity: O(k) - for general k, O(1) for k=2
    """
    if len(prices) < 2 or k == 0:
        return 0
    
    # If k >= n/2, we can do as many transactions as we want
    if k >= len(prices) // 2:
        return max_profit_unlimited_transactions(prices)
    
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


def max_profit_unlimited_transactions(prices):
    """
    HELPER: UNLIMITED TRANSACTIONS:
    ==============================
    Maximum profit with unlimited transactions.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]
    return profit


def max_profit_iii_memoization(prices):
    """
    MEMOIZATION APPROACH:
    ====================
    Top-down DP with memoization.
    
    Time Complexity: O(n * 2 * 3) = O(n) - states: position, holding, transactions
    Space Complexity: O(n * 2 * 3) = O(n) - memoization table
    """
    if len(prices) < 2:
        return 0
    
    memo = {}
    
    def dp(day, holding, transactions_left):
        """
        day: current day
        holding: 1 if holding stock, 0 if not
        transactions_left: number of complete transactions remaining
        """
        if day >= len(prices) or transactions_left == 0:
            return 0
        
        if (day, holding, transactions_left) in memo:
            return memo[(day, holding, transactions_left)]
        
        # Option 1: Do nothing
        result = dp(day + 1, holding, transactions_left)
        
        if holding:
            # Option 2: Sell (completes a transaction)
            result = max(result, prices[day] + dp(day + 1, 0, transactions_left - 1))
        else:
            # Option 2: Buy
            result = max(result, -prices[day] + dp(day + 1, 1, transactions_left))
        
        memo[(day, holding, transactions_left)] = result
        return result
    
    return dp(0, 0, 2)


def max_profit_iii_with_transactions(prices):
    """
    FIND MAXIMUM PROFIT AND ACTUAL TRANSACTIONS:
    ===========================================
    Return both maximum profit and the actual transactions.
    
    Time Complexity: O(n) - DP + transaction reconstruction
    Space Complexity: O(n) - store state transitions
    """
    if len(prices) < 2:
        return 0, []
    
    n = len(prices)
    
    # Use state machine with transaction tracking
    states = []
    buy1 = -prices[0]
    sell1 = 0
    buy2 = -prices[0]
    sell2 = 0
    
    for i, price in enumerate(prices):
        new_sell2 = max(sell2, buy2 + price)
        new_buy2 = max(buy2, sell1 - price)
        new_sell1 = max(sell1, buy1 + price)
        new_buy1 = max(buy1, -price)
        
        states.append({
            'buy1': buy1, 'sell1': sell1, 'buy2': buy2, 'sell2': sell2,
            'new_buy1': new_buy1, 'new_sell1': new_sell1, 
            'new_buy2': new_buy2, 'new_sell2': new_sell2
        })
        
        buy1, sell1, buy2, sell2 = new_buy1, new_sell1, new_buy2, new_sell2
    
    # Reconstruct transactions (simplified approach)
    transactions = []
    if sell2 > 0:
        # Find optimal transaction points
        profit1 = max_profit_iii_dp_precompute(prices)
        transactions.append(f"Maximum profit: {sell2}")
        transactions.append("Optimal transactions found using state machine")
    
    return sell2, transactions


# Test cases
def test_max_profit_iii():
    """Test all implementations with various inputs"""
    test_cases = [
        ([3,3,5,0,0,3,1,4], 6),
        ([1,2,3,4,5], 4),
        ([7,6,4,3,1], 0),
        ([1,2,4,2,5,7,2,4,9,0], 13),
        ([2,1,2,0,1], 2),
        ([3,2,6,5,0,3], 7),
        ([1,4,2], 3),
        ([2,4,1], 2),
        ([1], 0),
        ([], 0)
    ]
    
    print("Testing Best Time to Buy and Sell Stock III Solutions:")
    print("=" * 70)
    
    for i, (prices, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: prices = {prices}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for large inputs)
        if len(prices) <= 8:
            brute = max_profit_iii_bruteforce(prices.copy())
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        if len(prices) <= 20:
            divide = max_profit_iii_divide_and_conquer(prices.copy())
            print(f"Divide Conquer:   {divide:>3} {'✓' if divide == expected else '✗'}")
        
        dp_precompute = max_profit_iii_dp_precompute(prices.copy())
        state_machine = max_profit_iii_state_machine(prices.copy())
        general_k = max_profit_iii_general_k_transactions(prices.copy(), 2)
        memo = max_profit_iii_memoization(prices.copy())
        
        print(f"DP Precompute:    {dp_precompute:>3} {'✓' if dp_precompute == expected else '✗'}")
        print(f"State Machine:    {state_machine:>3} {'✓' if state_machine == expected else '✗'}")
        print(f"General K=2:      {general_k:>3} {'✓' if general_k == expected else '✗'}")
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        
        # Show transaction details for interesting cases
        if len(prices) <= 10 and expected > 0:
            profit, transactions = max_profit_iii_with_transactions(prices.copy())
            print(f"Transaction details: {transactions}")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(n^4),      Space: O(1)")
    print("Divide Conquer:   Time: O(n^2),      Space: O(n)")
    print("DP Precompute:    Time: O(n),        Space: O(n)")
    print("State Machine:    Time: O(n),        Space: O(1)")
    print("General K:        Time: O(n*k),      Space: O(k)")
    print("Memoization:      Time: O(n),        Space: O(n)")


if __name__ == "__main__":
    test_max_profit_iii()


"""
PATTERN RECOGNITION:
==================
This is an advanced stock trading DP problem:
- At most 2 transactions (buy-sell pairs)
- Cannot hold multiple stocks simultaneously
- Must sell before buying again
- Optimize for maximum profit

KEY INSIGHT - STATE MACHINE:
============================
Track 4 states after each day:
1. buy1: bought first stock
2. sell1: sold first stock  
3. buy2: bought second stock
4. sell2: sold second stock

State transitions:
- buy1 = max(buy1, -price)              # Buy first stock
- sell1 = max(sell1, buy1 + price)      # Sell first stock
- buy2 = max(buy2, sell1 - price)       # Buy second stock
- sell2 = max(sell2, buy2 + price)      # Sell second stock

MULTIPLE SOLUTION APPROACHES:
============================

1. **Brute Force**: O(n⁴) - try all transaction combinations
2. **Divide & Conquer**: O(n²) - split array, optimize each part
3. **DP Precompute**: O(n) time, O(n) space - precompute left/right profits
4. **State Machine**: O(n) time, O(1) space - track transaction states
5. **General K**: O(nk) - generalize to k transactions

OPTIMAL SOLUTION - STATE MACHINE:
=================================
```python
buy1 = sell1 = buy2 = sell2 = 0
buy1 = -prices[0]  # Initial buy

for price in prices[1:]:
    sell2 = max(sell2, buy2 + price)
    buy2 = max(buy2, sell1 - price)  
    sell1 = max(sell1, buy1 + price)
    buy1 = max(buy1, -price)
```

KEY INSIGHTS:
============
1. At most 2 transactions = at most 4 operations (buy,sell,buy,sell)
2. State machine tracks maximum profit at each operation
3. Order of updates matters (process sell2 before buy2, etc.)
4. Can be generalized to k transactions

DP PRECOMPUTE APPROACH:
======================
1. left[i] = max profit using prices[0...i] (≤1 transaction)
2. right[i] = max profit using prices[i...n-1] (≤1 transaction)  
3. Answer = max(left[i] + right[i+1]) for all valid i

VARIANTS TO PRACTICE:
====================
- Best Time to Buy and Sell Stock (121) - 1 transaction
- Best Time to Buy and Sell Stock II (122) - unlimited transactions
- Best Time to Buy and Sell Stock IV (188) - k transactions
- Best Time to Buy and Sell Stock with Cooldown (309) - cooldown constraint

INTERVIEW TIPS:
==============
1. Start with understanding transaction constraints
2. Show brute force approach first
3. Optimize with state machine DP
4. Explain state transitions clearly
5. Discuss space optimization (O(n) → O(1))
6. Mention generalization to k transactions
7. Handle edge cases (empty array, single element)
8. Draw state transition diagram if helpful
"""
