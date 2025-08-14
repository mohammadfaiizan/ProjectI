"""
Dynamic Programming - Stock Trading Patterns
This module implements various DP problems related to stock trading including single/multiple
transactions, cooldown periods, transaction fees, and k-constrained trading strategies.
"""

from typing import List, Dict, Tuple, Optional
import time

# ==================== SINGLE TRANSACTION STOCK PROBLEMS ====================

class SingleTransactionStock:
    """
    Single Transaction Stock Problems
    
    Buy and sell stock with at most one transaction to maximize profit.
    These are the foundation problems for more complex trading strategies.
    """
    
    def max_profit_one_transaction(self, prices: List[int]) -> int:
        """
        Maximum profit with at most one transaction
        
        LeetCode 121 - Best Time to Buy and Sell Stock
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            prices: Array of stock prices
        
        Returns:
            Maximum profit possible
        """
        if not prices or len(prices) < 2:
            return 0
        
        min_price = prices[0]
        max_profit = 0
        
        for price in prices[1:]:
            # Update maximum profit if selling today gives better profit
            max_profit = max(max_profit, price - min_price)
            # Update minimum price seen so far
            min_price = min(min_price, price)
        
        return max_profit
    
    def max_profit_with_buy_sell_dates(self, prices: List[int]) -> Tuple[int, int, int]:
        """
        Find maximum profit and the optimal buy/sell dates
        
        Args:
            prices: Array of stock prices
        
        Returns:
            Tuple of (max_profit, buy_day, sell_day)
        """
        if not prices or len(prices) < 2:
            return 0, -1, -1
        
        min_price = prices[0]
        max_profit = 0
        buy_day = 0
        sell_day = 0
        temp_buy_day = 0
        
        for i in range(1, len(prices)):
            current_profit = prices[i] - min_price
            
            if current_profit > max_profit:
                max_profit = current_profit
                buy_day = temp_buy_day
                sell_day = i
            
            if prices[i] < min_price:
                min_price = prices[i]
                temp_buy_day = i
        
        return max_profit, buy_day, sell_day
    
    def max_profit_dp_approach(self, prices: List[int]) -> int:
        """
        DP approach for single transaction (educational purpose)
        
        State: dp[i][0] = max profit on day i with no stock
               dp[i][1] = max profit on day i with stock
        """
        if not prices:
            return 0
        
        n = len(prices)
        # hold[i] = max profit on day i if we hold stock
        # sold[i] = max profit on day i if we don't hold stock
        hold = -prices[0]  # Buy on first day
        sold = 0           # Don't buy on first day
        
        for i in range(1, n):
            # Today we can sell (if we held) or do nothing
            new_sold = max(sold, hold + prices[i])
            # Today we can buy (if we hadn't bought before) or do nothing
            new_hold = max(hold, -prices[i])  # Only one transaction allowed
            
            sold = new_sold
            hold = new_hold
        
        return sold

# ==================== MULTIPLE TRANSACTIONS STOCK PROBLEMS ====================

class MultipleTransactionsStock:
    """
    Multiple Transactions Stock Problems
    
    Buy and sell stock with unlimited transactions to maximize profit.
    """
    
    def max_profit_unlimited_transactions(self, prices: List[int]) -> int:
        """
        Maximum profit with unlimited transactions
        
        LeetCode 122 - Best Time to Buy and Sell Stock II
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Key insight: Capture every profitable opportunity
        """
        if not prices or len(prices) < 2:
            return 0
        
        total_profit = 0
        
        for i in range(1, len(prices)):
            # If price increased, we could have bought yesterday and sold today
            if prices[i] > prices[i - 1]:
                total_profit += prices[i] - prices[i - 1]
        
        return total_profit
    
    def max_profit_with_transactions_list(self, prices: List[int]) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Find maximum profit and list of all transactions
        
        Args:
            prices: Array of stock prices
        
        Returns:
            Tuple of (max_profit, list_of_(buy_day, sell_day))
        """
        if not prices or len(prices) < 2:
            return 0, []
        
        transactions = []
        total_profit = 0
        i = 0
        
        while i < len(prices) - 1:
            # Find local minimum (buy point)
            while i < len(prices) - 1 and prices[i + 1] <= prices[i]:
                i += 1
            
            if i == len(prices) - 1:
                break
            
            buy_day = i
            
            # Find local maximum (sell point)
            while i < len(prices) - 1 and prices[i + 1] > prices[i]:
                i += 1
            
            sell_day = i
            profit = prices[sell_day] - prices[buy_day]
            total_profit += profit
            transactions.append((buy_day, sell_day))
        
        return total_profit, transactions
    
    def max_profit_dp_unlimited(self, prices: List[int]) -> int:
        """
        DP approach for unlimited transactions
        """
        if not prices:
            return 0
        
        # On any day: either hold stock or don't hold stock
        hold = -prices[0]  # Max profit if holding stock
        sold = 0           # Max profit if not holding stock
        
        for i in range(1, len(prices)):
            # To not hold stock today: either we already didn't hold, or we sell today
            new_sold = max(sold, hold + prices[i])
            # To hold stock today: either we already held, or we buy today
            new_hold = max(hold, sold - prices[i])  # Can buy after selling
            
            sold = new_sold
            hold = new_hold
        
        return sold

# ==================== LIMITED TRANSACTIONS STOCK PROBLEMS ====================

class LimitedTransactionsStock:
    """
    Limited Transactions Stock Problems
    
    Buy and sell stock with at most k transactions to maximize profit.
    """
    
    def max_profit_two_transactions(self, prices: List[int]) -> int:
        """
        Maximum profit with at most 2 transactions
        
        LeetCode 123 - Best Time to Buy and Sell Stock III
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not prices or len(prices) < 2:
            return 0
        
        # Track 4 states: first buy, first sell, second buy, second sell
        first_buy = -prices[0]   # After first buy
        first_sell = 0           # After first sell
        second_buy = -prices[0]  # After second buy (could be same day as first)
        second_sell = 0          # After second sell
        
        for i in range(1, len(prices)):
            # Update in reverse order to avoid using updated values
            second_sell = max(second_sell, second_buy + prices[i])
            second_buy = max(second_buy, first_sell - prices[i])
            first_sell = max(first_sell, first_buy + prices[i])
            first_buy = max(first_buy, -prices[i])
        
        return second_sell
    
    def max_profit_k_transactions(self, k: int, prices: List[int]) -> int:
        """
        Maximum profit with at most k transactions
        
        LeetCode 188 - Best Time to Buy and Sell Stock IV
        
        Time Complexity: O(nk) or O(n) if k is large
        Space Complexity: O(k) or O(1) if k is large
        """
        if not prices or len(prices) < 2 or k == 0:
            return 0
        
        n = len(prices)
        
        # If k is large enough, it's equivalent to unlimited transactions
        if k >= n // 2:
            return MultipleTransactionsStock().max_profit_unlimited_transactions(prices)
        
        # buy[i] = max profit after at most i transactions, currently holding stock
        # sell[i] = max profit after at most i transactions, not holding stock
        buy = [-prices[0]] * (k + 1)
        sell = [0] * (k + 1)
        
        for i in range(1, n):
            for j in range(k, 0, -1):  # Reverse order to avoid using updated values
                sell[j] = max(sell[j], buy[j] + prices[i])
                buy[j] = max(buy[j], sell[j - 1] - prices[i])
        
        return sell[k]
    
    def max_profit_k_transactions_2d(self, k: int, prices: List[int]) -> int:
        """
        2D DP approach for k transactions (more intuitive but space-inefficient)
        
        dp[i][j][0] = max profit on day i with at most j transactions, not holding
        dp[i][j][1] = max profit on day i with at most j transactions, holding
        """
        if not prices or len(prices) < 2 or k == 0:
            return 0
        
        n = len(prices)
        
        if k >= n // 2:
            return MultipleTransactionsStock().max_profit_unlimited_transactions(prices)
        
        # 3D DP: [day][transactions][holding_status]
        dp = [[[0, 0] for _ in range(k + 1)] for _ in range(n)]
        
        # Initialize first day
        for j in range(k + 1):
            dp[0][j][0] = 0         # Not holding
            dp[0][j][1] = -prices[0] # Holding (bought today)
        
        for i in range(1, n):
            for j in range(k + 1):
                # Not holding stock today
                dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i])
                
                # Holding stock today
                if j > 0:
                    dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i])
                else:
                    dp[i][j][1] = dp[i - 1][j][1]
        
        return dp[n - 1][k][0]

# ==================== STOCK WITH CONSTRAINTS ====================

class StockWithConstraints:
    """
    Stock Problems with Additional Constraints
    
    Trading with cooldown periods, transaction fees, and other restrictions.
    """
    
    def max_profit_with_cooldown(self, prices: List[int]) -> int:
        """
        Maximum profit with cooldown period
        
        LeetCode 309 - Best Time to Buy and Sell Stock with Cooldown
        
        After selling, must wait one day before buying again.
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not prices or len(prices) < 2:
            return 0
        
        # held[i] = max profit on day i if holding stock
        # sold[i] = max profit on day i if just sold stock
        # rest[i] = max profit on day i if resting (not holding, didn't just sell)
        
        held = -prices[0]  # Bought on day 0
        sold = 0           # Can't sell on day 0
        rest = 0           # Rest on day 0
        
        for i in range(1, len(prices)):
            prev_held = held
            prev_sold = sold
            prev_rest = rest
            
            # To hold today: either already held, or buy today (must have rested yesterday)
            held = max(prev_held, prev_rest - prices[i])
            
            # To sell today: must have held yesterday
            sold = prev_held + prices[i]
            
            # To rest today: either already resting, or just sold yesterday
            rest = max(prev_rest, prev_sold)
        
        # Final answer is max of sold or rest (can't be holding at the end)
        return max(sold, rest)
    
    def max_profit_with_cooldown_explicit_states(self, prices: List[int]) -> int:
        """
        Explicit state machine approach for cooldown problem
        
        States: buy, sell, cooldown
        """
        if not prices or len(prices) < 2:
            return 0
        
        # buy[i] = max profit ending in buy state on day i
        # sell[i] = max profit ending in sell state on day i  
        # cooldown[i] = max profit ending in cooldown state on day i
        
        buy = -prices[0]
        sell = 0
        cooldown = 0
        
        for i in range(1, len(prices)):
            new_buy = max(buy, cooldown - prices[i])  # Buy today or continue holding
            new_sell = buy + prices[i]                # Sell today
            new_cooldown = max(cooldown, sell)        # Rest today
            
            buy = new_buy
            sell = new_sell
            cooldown = new_cooldown
        
        return max(sell, cooldown)
    
    def max_profit_with_transaction_fee(self, prices: List[int], fee: int) -> int:
        """
        Maximum profit with transaction fee
        
        LeetCode 714 - Best Time to Buy and Sell Stock with Transaction Fee
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            prices: Array of stock prices
            fee: Transaction fee (paid when selling)
        """
        if not prices or len(prices) < 2:
            return 0
        
        # cash = max profit when not holding stock
        # hold = max profit when holding stock
        cash = 0
        hold = -prices[0]
        
        for i in range(1, len(prices)):
            # To have cash: either already had cash, or sell today (pay fee)
            new_cash = max(cash, hold + prices[i] - fee)
            # To hold stock: either already holding, or buy today
            new_hold = max(hold, cash - prices[i])
            
            cash = new_cash
            hold = new_hold
        
        return cash
    
    def max_profit_with_transaction_fee_optimized(self, prices: List[int], fee: int) -> int:
        """
        Optimized version that avoids unnecessary transactions
        """
        if not prices or len(prices) < 2:
            return 0
        
        cash = 0
        hold = -prices[0]
        
        for price in prices[1:]:
            cash = max(cash, hold + price - fee)
            hold = max(hold, cash - price)
        
        return cash

# ==================== ADVANCED STOCK STRATEGIES ====================

class AdvancedStockStrategies:
    """
    Advanced Stock Trading Strategies
    
    Complex scenarios with multiple constraints and optimization techniques.
    """
    
    def max_profit_with_holding_limit(self, prices: List[int], max_holding_days: int) -> int:
        """
        Maximum profit with maximum holding period constraint
        
        Args:
            prices: Array of stock prices
            max_holding_days: Maximum days to hold stock
        
        Returns:
            Maximum profit with holding constraint
        """
        if not prices or len(prices) < 2 or max_holding_days <= 0:
            return 0
        
        n = len(prices)
        max_profit = 0
        
        # For each possible buy day
        for buy_day in range(n - 1):
            # Try selling within holding limit
            max_sell_day = min(buy_day + max_holding_days, n - 1)
            
            for sell_day in range(buy_day + 1, max_sell_day + 1):
                profit = prices[sell_day] - prices[buy_day]
                max_profit = max(max_profit, profit)
        
        return max_profit
    
    def max_profit_with_min_holding_period(self, prices: List[int], min_holding_days: int) -> int:
        """
        Maximum profit with minimum holding period
        
        Args:
            prices: Array of stock prices
            min_holding_days: Minimum days to hold stock before selling
        """
        if not prices or len(prices) < min_holding_days + 1:
            return 0
        
        n = len(prices)
        max_profit = 0
        
        for buy_day in range(n - min_holding_days):
            # Must hold for at least min_holding_days
            for sell_day in range(buy_day + min_holding_days, n):
                profit = prices[sell_day] - prices[buy_day]
                max_profit = max(max_profit, profit)
        
        return max_profit
    
    def max_profit_with_volume_constraint(self, prices: List[int], volumes: List[int], 
                                        max_volume: int) -> int:
        """
        Maximum profit with volume trading constraint
        
        Args:
            prices: Array of stock prices
            volumes: Array of available volumes
            max_volume: Maximum volume that can be traded
        """
        if not prices or len(prices) != len(volumes):
            return 0
        
        n = len(prices)
        max_profit = 0
        
        # DP approach considering volume constraint
        # dp[i][v] = max profit up to day i with volume v used
        dp = {}
        
        def solve(day: int, volume_used: int, holding: bool, buy_price: int) -> int:
            if day >= n:
                return 0
            
            if (day, volume_used, holding, buy_price) in dp:
                return dp[(day, volume_used, holding, buy_price)]
            
            result = 0
            
            # Option 1: Do nothing today
            result = max(result, solve(day + 1, volume_used, holding, buy_price))
            
            if not holding and volume_used + volumes[day] <= max_volume:
                # Option 2: Buy today
                result = max(result, solve(day + 1, volume_used + volumes[day], 
                                         True, prices[day]))
            
            if holding and volume_used + volumes[day] <= max_volume:
                # Option 3: Sell today
                profit = prices[day] - buy_price
                result = max(result, profit + solve(day + 1, volume_used + volumes[day], 
                                                  False, 0))
            
            dp[(day, volume_used, holding, buy_price)] = result
            return result
        
        return solve(0, 0, False, 0)
    
    def max_profit_multiple_stocks(self, stock_prices: List[List[int]], 
                                 max_total_investment: int) -> int:
        """
        Maximum profit trading multiple stocks with budget constraint
        
        Args:
            stock_prices: List of price arrays for different stocks
            max_total_investment: Maximum total money that can be invested
        """
        if not stock_prices or max_total_investment <= 0:
            return 0
        
        num_stocks = len(stock_prices)
        max_profit = 0
        
        # For simplicity, consider single transaction per stock
        # More complex versions would need multi-dimensional DP
        
        def backtrack(stock_idx: int, remaining_budget: int, current_profit: int):
            nonlocal max_profit
            
            if stock_idx >= num_stocks:
                max_profit = max(max_profit, current_profit)
                return
            
            # Option 1: Skip this stock
            backtrack(stock_idx + 1, remaining_budget, current_profit)
            
            # Option 2: Trade this stock (find best single transaction)
            prices = stock_prices[stock_idx]
            best_profit = 0
            min_price = float('inf')
            
            for price in prices:
                if price <= remaining_budget:
                    min_price = min(min_price, price)
                    if min_price < price:
                        profit = price - min_price
                        best_profit = max(best_profit, profit)
            
            if best_profit > 0:
                backtrack(stock_idx + 1, remaining_budget - min_price, 
                         current_profit + best_profit)
        
        backtrack(0, max_total_investment, 0)
        return max_profit

# ==================== PERFORMANCE ANALYSIS ====================

def performance_comparison():
    """Compare performance of different stock trading approaches"""
    print("=== Stock Trading DP Performance Analysis ===\n")
    
    import random
    
    # Generate test data
    test_sizes = [100, 500, 1000]
    
    for size in test_sizes:
        prices = [random.randint(1, 200) for _ in range(size)]
        
        print(f"Stock prices array size: {size}")
        
        # Single transaction approaches
        single = SingleTransactionStock()
        
        start_time = time.time()
        profit_simple = single.max_profit_one_transaction(prices)
        time_simple = time.time() - start_time
        
        start_time = time.time()
        profit_dp = single.max_profit_dp_approach(prices)
        time_dp = time.time() - start_time
        
        print(f"  Single Transaction:")
        print(f"    Simple O(n): {profit_simple} ({time_simple:.6f}s)")
        print(f"    DP O(n): {profit_dp} ({time_dp:.6f}s)")
        print(f"    Results match: {profit_simple == profit_dp}")
        
        # Multiple transactions
        multi = MultipleTransactionsStock()
        
        start_time = time.time()
        profit_unlimited = multi.max_profit_unlimited_transactions(prices)
        time_unlimited = time.time() - start_time
        
        print(f"  Unlimited Transactions: {profit_unlimited} ({time_unlimited:.6f}s)")
        
        # K transactions (k=2)
        limited = LimitedTransactionsStock()
        
        start_time = time.time()
        profit_two = limited.max_profit_two_transactions(prices)
        time_two = time.time() - start_time
        
        start_time = time.time()
        profit_k2 = limited.max_profit_k_transactions(2, prices)
        time_k2 = time.time() - start_time
        
        print(f"  Two Transactions:")
        print(f"    Specialized: {profit_two} ({time_two:.6f}s)")
        print(f"    General K=2: {profit_k2} ({time_k2:.6f}s)")
        print(f"    Results match: {profit_two == profit_k2}")
        print()

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Stock Trading DP Demo ===\n")
    
    # Test data
    test_prices = [7, 1, 5, 3, 6, 4]
    
    # Single Transaction
    print("1. Single Transaction Problems:")
    single = SingleTransactionStock()
    
    max_profit = single.max_profit_one_transaction(test_prices)
    profit_with_dates, buy_day, sell_day = single.max_profit_with_buy_sell_dates(test_prices)
    profit_dp = single.max_profit_dp_approach(test_prices)
    
    print(f"  Prices: {test_prices}")
    print(f"  Maximum profit: {max_profit}")
    print(f"  Optimal buy day {buy_day} (price {test_prices[buy_day]}), sell day {sell_day} (price {test_prices[sell_day]})")
    print(f"  DP approach result: {profit_dp}")
    print()
    
    # Multiple Transactions
    print("2. Multiple Transactions Problems:")
    multi = MultipleTransactionsStock()
    
    unlimited_profit = multi.max_profit_unlimited_transactions(test_prices)
    profit_with_transactions, transactions = multi.max_profit_with_transactions_list(test_prices)
    unlimited_dp = multi.max_profit_dp_unlimited(test_prices)
    
    print(f"  Unlimited transactions profit: {unlimited_profit}")
    print(f"  All transactions: {transactions}")
    print(f"  DP approach result: {unlimited_dp}")
    print()
    
    # Limited Transactions
    print("3. Limited Transactions Problems:")
    limited = LimitedTransactionsStock()
    
    two_transactions_profit = limited.max_profit_two_transactions(test_prices)
    k_transactions_profit = limited.max_profit_k_transactions(2, test_prices)
    
    print(f"  At most 2 transactions: {two_transactions_profit}")
    print(f"  At most K=2 transactions (general): {k_transactions_profit}")
    
    # Test with different k values
    for k in [1, 3, 5]:
        k_profit = limited.max_profit_k_transactions(k, test_prices)
        print(f"  At most {k} transactions: {k_profit}")
    print()
    
    # Stock with Constraints
    print("4. Stock with Constraints:")
    constraints = StockWithConstraints()
    
    # Cooldown
    cooldown_prices = [1, 2, 3, 0, 2]
    cooldown_profit = constraints.max_profit_with_cooldown(cooldown_prices)
    cooldown_explicit = constraints.max_profit_with_cooldown_explicit_states(cooldown_prices)
    
    print(f"  Cooldown prices: {cooldown_prices}")
    print(f"  Profit with cooldown: {cooldown_profit}")
    print(f"  Explicit states approach: {cooldown_explicit}")
    
    # Transaction fee
    fee = 2
    fee_profit = constraints.max_profit_with_transaction_fee(test_prices, fee)
    fee_optimized = constraints.max_profit_with_transaction_fee_optimized(test_prices, fee)
    
    print(f"  Transaction fee {fee} profit: {fee_profit}")
    print(f"  Optimized fee approach: {fee_optimized}")
    print()
    
    # Advanced Strategies
    print("5. Advanced Strategies:")
    advanced = AdvancedStockStrategies()
    
    # Holding limit
    max_holding = 3
    holding_limit_profit = advanced.max_profit_with_holding_limit(test_prices, max_holding)
    print(f"  Max holding {max_holding} days profit: {holding_limit_profit}")
    
    # Minimum holding period
    min_holding = 2
    min_holding_profit = advanced.max_profit_with_min_holding_period(test_prices, min_holding)
    print(f"  Min holding {min_holding} days profit: {min_holding_profit}")
    
    # Volume constraint
    volumes = [100, 200, 150, 300, 250, 180]
    max_volume = 500
    volume_profit = advanced.max_profit_with_volume_constraint(test_prices, volumes, max_volume)
    print(f"  Volume constraint (max {max_volume}) profit: {volume_profit}")
    
    # Multiple stocks
    stock_prices = [[1, 3, 2, 4], [2, 1, 4, 3], [3, 2, 1, 5]]
    max_investment = 10
    multi_stock_profit = advanced.max_profit_multiple_stocks(stock_prices, max_investment)
    print(f"  Multiple stocks (budget {max_investment}) profit: {multi_stock_profit}")
    print()
    
    # Performance comparison
    performance_comparison()
    
    # Pattern Recognition Guide
    print("=== Stock Trading DP Pattern Recognition ===")
    print("Common Stock DP Patterns:")
    print("  1. State Definition: [day][transactions_used][holding_status]")
    print("  2. Single Transaction: Track min_price_so_far and max_profit")
    print("  3. Unlimited Transactions: Greedy approach or simple DP")
    print("  4. K Transactions: 2D/3D DP with transaction count")
    print("  5. Constraints: Add extra states for cooldown, fees, etc.")
    
    print("\nKey State Transitions:")
    print("  buy[i] = max(buy[i-1], sell[i-1] - price[i])")
    print("  sell[i] = max(sell[i-1], buy[i-1] + price[i])")
    print("  For constraints: modify transitions accordingly")
    
    print("\nOptimization Techniques:")
    print("  1. Space optimization: O(n) → O(1) when possible")
    print("  2. Large K optimization: Use greedy when k >= n/2")
    print("  3. State compression: Combine related states")
    print("  4. Rolling arrays: For multi-dimensional problems")
    
    print("\nProblem Classification:")
    print("  1. Single vs Multiple transactions")
    print("  2. Limited vs Unlimited transactions")
    print("  3. With vs Without constraints (cooldown, fees)")
    print("  4. Single vs Multiple assets")
    print("  5. Discrete vs Continuous time")
    
    print("\nReal-world Applications:")
    print("  1. Algorithmic trading strategies")
    print("  2. Portfolio optimization")
    print("  3. Options pricing models")
    print("  4. Cryptocurrency trading")
    print("  5. Commodity trading")
    print("  6. Energy trading (buy/sell electricity)")
    
    print("\nCommon Pitfalls:")
    print("  1. Off-by-one errors in transaction counting")
    print("  2. Incorrect handling of constraints")
    print("  3. Not considering edge cases (empty arrays)")
    print("  4. Inefficient space usage for large K")
    print("  5. Missing the greedy optimization opportunity")
    
    print("\n=== Demo Complete ===") 