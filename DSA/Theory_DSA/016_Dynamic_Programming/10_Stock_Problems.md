# Stock Buy Sell DP

## Buy Sell I (One Transaction)

```python
def max_profit_one_transaction(prices):
    if not prices:
        return 0
    min_price = prices[0]
    max_profit = 0
    for p in prices:
        max_profit = max(max_profit, p - min_price)
        min_price = min(min_price, p)
    return max_profit
```

## Buy Sell II (Unlimited)

```python
def max_profit_unlimited(prices):
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]
    return profit
```

## Buy Sell III (At Most 2)

```python
def max_profit_two_transactions(prices):
    if not prices:
        return 0
    n = len(prices)
    left = [0] * n
    right = [0] * n
    min_price = prices[0]
    for i in range(1, n):
        left[i] = max(left[i - 1], prices[i] - min_price)
        min_price = min(min_price, prices[i])
    max_price = prices[-1]
    for i in range(n - 2, -1, -1):
        right[i] = max(right[i + 1], max_price - prices[i])
        max_price = max(max_price, prices[i])
    return max(left[i] + right[i] for i in range(n))
```

## Buy Sell IV (At Most K)

```python
def max_profit_k_transactions(k, prices):
    n = len(prices)
    if n < 2 or k == 0:
        return 0
    if k >= n // 2:
        return sum(max(0, prices[i] - prices[i - 1]) for i in range(1, n))
    dp = [[0] * n for _ in range(k + 1)]
    for t in range(1, k + 1):
        max_diff = -prices[0]
        for i in range(1, n):
            dp[t][i] = max(dp[t][i - 1], prices[i] + max_diff)
            max_diff = max(max_diff, dp[t - 1][i] - prices[i])
    return dp[k][n - 1]
```

## With Cooldown

```python
def max_profit_cooldown(prices):
    sold, held, rest = float('-inf'), float('-inf'), 0
    for p in prices:
        sold, held, rest = held + p, max(held, rest - p), max(rest, sold)
    return max(sold, rest)
```

## With Transaction Fee

```python
def max_profit_fee(prices, fee):
    cash, hold = 0, -prices[0]
    for i in range(1, len(prices)):
        cash = max(cash, hold + prices[i] - fee)
        hold = max(hold, cash - prices[i])
    return cash
```

## State Machine Approach (Hold, Sold, Rest)

```python
def max_profit_state_machine(prices, fee=0, cooldown=False):
    if cooldown:
        sold, held, rest = float('-inf'), float('-inf'), 0
        for p in prices:
            sold, held, rest = held + p, max(held, rest - p), max(rest, sold)
        return max(sold, rest)
    else:
        cash, hold = 0, -prices[0]
        for i in range(1, len(prices)):
            cash = max(cash, hold + prices[i] - fee)
            hold = max(hold, cash - prices[i])
        return cash
```

## Space-Optimized Buy Sell IV

```python
def max_profit_k_optimized(k, prices):
    n = len(prices)
    if n < 2 or k == 0:
        return 0
    if k >= n // 2:
        return sum(max(0, prices[i] - prices[i - 1]) for i in range(1, n))
    dp = [0] * n
    for t in range(1, k + 1):
        max_diff = -prices[0]
        new_dp = [0] * n
        for i in range(1, n):
            new_dp[i] = max(new_dp[i - 1], prices[i] + max_diff)
            max_diff = max(max_diff, dp[i] - prices[i])
        dp = new_dp
    return dp[-1]
```
