"""
LeetCode 121: Best Time to Buy and Sell Stock
Difficulty: Easy
Category: Stock Problems / Array DP

PROBLEM DESCRIPTION:
===================
You are given an array prices where prices[i] is the price of a given stock on the ith day.
You want to maximize your profit by choosing a single day to buy one stock and choosing 
a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve 
any profit, return 0.

Example 1:
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.

Example 2:
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.

Constraints:
- 1 <= prices.length <= 10^5
- 0 <= prices[i] <= 10^4
"""

def max_profit_bruteforce(prices):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible buy-sell combinations.
    For each buy day, try all possible sell days after it.
    
    Time Complexity: O(n^2) - nested loops
    Space Complexity: O(1) - constant space
    """
    if not prices or len(prices) < 2:
        return 0
    
    max_profit = 0
    
    # Try each day as buy day
    for buy_day in range(len(prices) - 1):
        # Try each subsequent day as sell day
        for sell_day in range(buy_day + 1, len(prices)):
            profit = prices[sell_day] - prices[buy_day]
            max_profit = max(max_profit, profit)
    
    return max_profit


def max_profit_two_pass(prices):
    """
    TWO PASS APPROACH:
    =================
    First pass: Find minimum price up to each day
    Second pass: For each day, calculate profit if selling on that day
    
    Time Complexity: O(n) - two passes through array
    Space Complexity: O(n) - additional array for minimum prices
    """
    if not prices or len(prices) < 2:
        return 0
    
    n = len(prices)
    
    # First pass: min_price[i] = minimum price from day 0 to day i
    min_price = [0] * n
    min_price[0] = prices[0]
    
    for i in range(1, n):
        min_price[i] = min(min_price[i - 1], prices[i])
    
    # Second pass: calculate maximum profit
    max_profit = 0
    for i in range(1, n):
        profit = prices[i] - min_price[i - 1]  # Buy before day i, sell on day i
        max_profit = max(max_profit, profit)
    
    return max_profit


def max_profit_one_pass(prices):
    """
    ONE PASS APPROACH (OPTIMAL):
    ===========================
    Keep track of minimum price seen so far and maximum profit.
    For each price, calculate profit if selling today and update maximum.
    
    Time Complexity: O(n) - single pass through array
    Space Complexity: O(1) - constant space
    """
    if not prices or len(prices) < 2:
        return 0
    
    min_price = prices[0]
    max_profit = 0
    
    for price in prices[1:]:
        # Update maximum profit if selling today
        max_profit = max(max_profit, price - min_price)
        # Update minimum price seen so far
        min_price = min(min_price, price)
    
    return max_profit


def max_profit_kadane_variant(prices):
    """
    KADANE'S ALGORITHM VARIANT:
    ==========================
    Transform to maximum subarray problem.
    Convert prices to daily changes and find maximum subarray sum.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if not prices or len(prices) < 2:
        return 0
    
    max_profit = 0
    current_profit = 0
    
    # Calculate profit for each consecutive day pair
    for i in range(1, len(prices)):
        daily_change = prices[i] - prices[i - 1]
        # Either start new transaction or continue current one
        current_profit = max(0, current_profit + daily_change)
        max_profit = max(max_profit, current_profit)
    
    return max_profit


def max_profit_dp_state_machine(prices):
    """
    DYNAMIC PROGRAMMING - STATE MACHINE:
    ===================================
    Two states: hold (own stock) and sold (don't own stock)
    Transition between states based on buy/sell decisions.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if not prices or len(prices) < 2:
        return 0
    
    # hold[i] = max profit when holding stock on day i
    # sold[i] = max profit when not holding stock on day i
    hold = -prices[0]  # Buy on first day
    sold = 0           # Don't own stock initially
    
    for i in range(1, len(prices)):
        # To hold stock today: either keep holding or buy today
        # Since we can only buy once, buying today means we start fresh
        new_hold = max(hold, -prices[i])
        
        # To not hold stock today: either keep not holding or sell today
        new_sold = max(sold, hold + prices[i])
        
        hold = new_hold
        sold = new_sold
    
    return sold  # We want to end without holding stock


def max_profit_with_buy_sell_days(prices):
    """
    FIND MAXIMUM PROFIT AND BUY/SELL DAYS:
    =====================================
    Return both maximum profit and the optimal buy/sell days.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if not prices or len(prices) < 2:
        return 0, -1, -1
    
    min_price = prices[0]
    min_day = 0
    max_profit = 0
    buy_day = 0
    sell_day = 0
    
    for i in range(1, len(prices)):
        # If selling today gives better profit, update
        if prices[i] - min_price > max_profit:
            max_profit = prices[i] - min_price
            buy_day = min_day
            sell_day = i
        
        # Update minimum price and its day
        if prices[i] < min_price:
            min_price = prices[i]
            min_day = i
    
    return max_profit, buy_day, sell_day


def max_profit_divide_conquer(prices):
    """
    DIVIDE AND CONQUER APPROACH:
    ===========================
    Divide array and find maximum in left half, right half, or crossing.
    Not optimal for this problem but demonstrates the technique.
    
    Time Complexity: O(n log n) - divide and conquer
    Space Complexity: O(log n) - recursion stack
    """
    if not prices or len(prices) < 2:
        return 0
    
    def max_profit_helper(left, right):
        if left >= right:
            return 0
        
        if right - left == 1:
            return max(0, prices[right] - prices[left])
        
        mid = (left + right) // 2
        
        # Maximum profit in left half
        left_profit = max_profit_helper(left, mid)
        
        # Maximum profit in right half
        right_profit = max_profit_helper(mid + 1, right)
        
        # Maximum profit crossing the middle
        # Find minimum in left half
        min_left = min(prices[left:mid + 1])
        # Find maximum in right half
        max_right = max(prices[mid + 1:right + 1])
        cross_profit = max(0, max_right - min_left)
        
        return max(left_profit, right_profit, cross_profit)
    
    return max_profit_helper(0, len(prices) - 1)


# Test cases
def test_max_profit():
    """Test all implementations with various inputs"""
    test_cases = [
        ([7, 1, 5, 3, 6, 4], 5),
        ([7, 6, 4, 3, 1], 0),
        ([1, 2], 1),
        ([2, 1], 0),
        ([1], 0),
        ([], 0),
        ([3, 3, 5, 0, 0, 3, 1, 4], 4),
        ([1, 2, 3, 4, 5], 4),
        ([5, 4, 3, 2, 1], 0)
    ]
    
    print("Testing Best Time to Buy and Sell Stock Solutions:")
    print("=" * 70)
    
    for i, (prices, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: {prices}")
        print(f"Expected: {expected}")
        
        if len(prices) <= 100:  # Brute force for small inputs
            brute = max_profit_bruteforce(prices.copy())
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        two_pass = max_profit_two_pass(prices.copy())
        one_pass = max_profit_one_pass(prices.copy())
        kadane = max_profit_kadane_variant(prices.copy())
        state_machine = max_profit_dp_state_machine(prices.copy())
        divide_conquer = max_profit_divide_conquer(prices.copy())
        
        print(f"Two Pass:         {two_pass:>3} {'✓' if two_pass == expected else '✗'}")
        print(f"One Pass:         {one_pass:>3} {'✓' if one_pass == expected else '✗'}")
        print(f"Kadane Variant:   {kadane:>3} {'✓' if kadane == expected else '✗'}")
        print(f"State Machine:    {state_machine:>3} {'✓' if state_machine == expected else '✗'}")
        print(f"Divide Conquer:   {divide_conquer:>3} {'✓' if divide_conquer == expected else '✗'}")
        
        # Show buy/sell days for profitable cases
        if expected > 0 and len(prices) <= 10:
            profit, buy_day, sell_day = max_profit_with_buy_sell_days(prices.copy())
            print(f"Buy day {buy_day} (price={prices[buy_day]}), Sell day {sell_day} (price={prices[sell_day]})")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(n^2),     Space: O(1)")
    print("Two Pass:         Time: O(n),       Space: O(n)")
    print("One Pass:         Time: O(n),       Space: O(1)") 
    print("Kadane Variant:   Time: O(n),       Space: O(1)")
    print("State Machine:    Time: O(n),       Space: O(1)")
    print("Divide Conquer:   Time: O(n log n), Space: O(log n)")


if __name__ == "__main__":
    test_max_profit()


"""
PATTERN RECOGNITION:
==================
This is the simplest stock trading problem:
- Can buy once and sell once
- Must buy before selling
- Maximize profit = selling_price - buying_price

KEY INSIGHTS:
============
1. For each day, we want to know the minimum price before that day
2. Profit on day i = prices[i] - min_price_before_i
3. Track minimum price seen so far and maximum profit
4. This is similar to maximum subarray problem (Kadane's algorithm)

MULTIPLE SOLUTION APPROACHES:
============================

1. BRUTE FORCE: Try all buy-sell pairs - O(n²)
2. TWO PASS: Precompute minimums, then calculate profits - O(n) time, O(n) space
3. ONE PASS: Track minimum and maximum profit simultaneously - O(n) time, O(1) space
4. KADANE'S VARIANT: Transform to maximum subarray sum - O(n) time, O(1) space
5. STATE MACHINE DP: Track hold/sold states - O(n) time, O(1) space

OPTIMAL SOLUTION:
================
One pass approach is optimal: O(n) time, O(1) space

for each price:
    max_profit = max(max_profit, price - min_price)
    min_price = min(min_price, price)

VARIANTS TO PRACTICE:
====================
- Best Time to Buy and Sell Stock II (122) - multiple transactions
- Best Time to Buy and Sell Stock III (123) - at most 2 transactions
- Best Time to Buy and Sell Stock IV (188) - at most k transactions
- Best Time to Buy and Sell Stock with Cooldown (309) - cooldown period
- Best Time to Buy and Sell Stock with Transaction Fee (714) - transaction fee

INTERVIEW TIPS:
==============
1. Start with brute force to show understanding
2. Optimize by tracking minimum price seen so far
3. Explain why we can't use greedy (need to know future)
4. Draw timeline to visualize buy-sell constraint
5. Mention connection to maximum subarray problem
6. Discuss how this extends to multiple transactions
7. Handle edge cases (empty array, single element, decreasing prices)
"""
