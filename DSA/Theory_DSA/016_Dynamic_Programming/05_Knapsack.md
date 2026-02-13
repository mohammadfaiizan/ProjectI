# Knapsack and Related Problems

## 0/1 Knapsack (2D DP)

```python
def knapsack_01_2d(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if w >= weights[i - 1]:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
    return dp[n][capacity]
```

## 0/1 Knapsack (1D Optimized)

```python
def knapsack_01_1d(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]
```

## Unbounded Knapsack

```python
def unbounded_knapsack(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]
```

## Bounded Knapsack

```python
def bounded_knapsack(weights, values, counts, capacity):
    items = []
    for w, v, c in zip(weights, values, counts):
        k = 1
        while k < c:
            items.append((w * k, v * k))
            c -= k
            k *= 2
        if c:
            items.append((w * c, v * c))
    dp = [0] * (capacity + 1)
    for w, v in items:
        for cap in range(capacity, w - 1, -1):
            dp[cap] = max(dp[cap], dp[cap - w] + v)
    return dp[capacity]
```

## Subset Sum

```python
def subset_sum(nums, target):
    dp = [False] * (target + 1)
    dp[0] = True
    for x in nums:
        for s in range(target, x - 1, -1):
            dp[s] = dp[s] or dp[s - x]
    return dp[target]
```

## Equal Partition

```python
def can_partition(nums):
    total = sum(nums)
    if total % 2:
        return False
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for x in nums:
        for s in range(target, x - 1, -1):
            dp[s] = dp[s] or dp[s - x]
    return dp[target]
```

## Target Sum

```python
def find_target_sum_ways(nums, target):
    total = sum(nums)
    if abs(target) > total or (total + target) % 2:
        return 0
    s = (total + target) // 2
    dp = [0] * (s + 1)
    dp[0] = 1
    for x in nums:
        for i in range(s, x - 1, -1):
            dp[i] += dp[i - x]
    return dp[s]
```

## Coin Change I (Min Coins)

```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for c in coins:
            if c <= i:
                dp[i] = min(dp[i], dp[i - c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

## Coin Change II (Count Ways)

```python
def change(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1
    for c in coins:
        for i in range(c, amount + 1):
            dp[i] += dp[i - c]
    return dp[amount]
```

## Rod Cutting

```python
def rod_cutting(prices, n):
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            dp[i] = max(dp[i], prices[j - 1] + dp[i - j])
    return dp[n]
```

## Integer Break

```python
def integer_break(n):
    if n <= 3:
        return n - 1
    dp = [0] * (n + 1)
    dp[1], dp[2], dp[3] = 1, 2, 3
    for i in range(4, n + 1):
        for j in range(1, i):
            dp[i] = max(dp[i], j * dp[i - j])
    return dp[n]
```

## Last Stone Weight II

```python
def last_stone_weight_ii(stones):
    total = sum(stones)
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for s in stones:
        for i in range(target, s - 1, -1):
            dp[i] = dp[i] or dp[i - s]
    for i in range(target, -1, -1):
        if dp[i]:
            return total - 2 * i
    return 0
```

## Ones and Zeroes (Two Constraints)

```python
def find_max_form(strs, m, n):
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for s in strs:
        zeros = s.count('0')
        ones = len(s) - zeros
        for i in range(m, zeros - 1, -1):
            for j in range(n, ones - 1, -1):
                dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
    return dp[m][n]
```

## Profitable Schemes

```python
def profitable_schemes(n, min_profit, group, profit):
    MOD = 10**9 + 7
    dp = [[0] * (min_profit + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    for g, p in zip(group, profit):
        for i in range(n, g - 1, -1):
            for j in range(min_profit, -1, -1):
                dp[i][min(j + p, min_profit)] = (dp[i][min(j + p, min_profit)] + dp[i - g][j]) % MOD
    return sum(dp[i][min_profit] for i in range(n + 1)) % MOD
```

## Combination Sum IV

```python
def combination_sum4(nums, target):
    dp = [0] * (target + 1)
    dp[0] = 1
    for i in range(1, target + 1):
        for n in nums:
            if n <= i:
                dp[i] += dp[i - n]
    return dp[target]
```

## Perfect Squares

```python
def num_squares(n):
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    for i in range(1, n + 1):
        j = 1
        while j * j <= i:
            dp[i] = min(dp[i], dp[i - j * j] + 1)
            j += 1
    return dp[n]
```

## Minimum Cost for Tickets

```python
def mincost_tickets(days, costs):
    dp = [0] * (days[-1] + 1)
    days_set = set(days)
    for i in range(1, days[-1] + 1):
        if i not in days_set:
            dp[i] = dp[i - 1]
        else:
            dp[i] = min(
                dp[max(0, i - 1)] + costs[0],
                dp[max(0, i - 7)] + costs[1],
                dp[max(0, i - 30)] + costs[2]
            )
    return dp[days[-1]]
```

## Word Break

```python
def word_break(s, word_dict):
    word_set = set(word_dict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[len(s)]
```

## Minimum Cost to Cut Stick

```python
def min_cost_cut_stick(n, cuts):
    cuts = sorted([0] + cuts + [n])
    m = len(cuts)
    dp = [[0] * m for _ in range(m)]
    for length in range(2, m):
        for i in range(m - length):
            j = i + length
            dp[i][j] = min(
                dp[i][k] + dp[k][j] for k in range(i + 1, j)
            ) + cuts[j] - cuts[i]
    return dp[0][m - 1]
```
