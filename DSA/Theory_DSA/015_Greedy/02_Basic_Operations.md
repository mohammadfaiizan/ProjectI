# Greedy - Basic Operations

## Greedy Algorithm Design Steps

1. **Define objective**: What are we maximizing or minimizing?
2. **Identify choice**: What decision do we make at each step?
3. **Prove correctness**: Use exchange argument, greedy stays ahead, or structural argument
4. **Implement**: Sort if needed, iterate with greedy selection
5. **Verify**: Check edge cases and complexity

```python
def greedy_design_template(items, objective_func):
    items_sorted = sorted(items, key=objective_func)
    result = []
    for item in items_sorted:
        if is_safe_choice(item, result):
            result.append(item)
    return result
```

## Activity Selection

Select maximum number of non-overlapping activities. Sort by end time; pick each activity that does not overlap with the last chosen.

```python
def activity_selection(activities):
    activities.sort(key=lambda x: x[1])
    count = 0
    last_end = -float('inf')
    for start, end in activities:
        if start >= last_end:
            count += 1
            last_end = end
    return count
```

## Coin Change Greedy (When It Works vs Fails)

Greedy works when denominations are canonical (e.g., standard US coins). It fails for arbitrary denominations.

```python
def coin_change_greedy(coins, amount):
    coins.sort(reverse=True)
    count = 0
    for c in coins:
        count += amount // c
        amount %= c
    return count if amount == 0 else -1
```

Fails for coins=[1, 3, 4], amount=6: greedy gives 4+1+1=3 coins; optimal is 3+3=2 coins. Use DP for general case.

## Fractional Knapsack

Take items by descending value/weight ratio. Can take fractions of items.

```python
def fractional_knapsack(weights, values, capacity):
    items = [(v / w, w, v) for v, w in zip(values, weights)]
    items.sort(key=lambda x: x[0], reverse=True)
    total_value = 0
    for ratio, w, v in items:
        if capacity <= 0:
            break
        take = min(w, capacity)
        total_value += take * ratio
        capacity -= take
    return total_value
```

## Minimum Coins for Value (Greedy vs DP)

Greedy works only for canonical systems. DP always gives optimal.

```python
def min_coins_greedy(coins, amount):
    coins.sort(reverse=True)
    count = 0
    for c in coins:
        count += amount // c
        amount %= c
    return count if amount == 0 else -1

def min_coins_dp(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for c in coins:
            if i >= c:
                dp[i] = min(dp[i], dp[i - c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

## Jump Game I (Can Reach End?)

Track maximum reachable index. If we can reach index i, we can reach any index up to i + nums[i].

```python
def can_jump(nums):
    max_reach = 0
    for i, jump in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + jump)
        if max_reach >= len(nums) - 1:
            return True
    return max_reach >= len(nums) - 1
```

## Assign Cookies

Assign smallest cookie that satisfies each child. Sort both arrays; two pointers.

```python
def find_content_children(g, s):
    g.sort()
    s.sort()
    i = j = 0
    count = 0
    while i < len(g) and j < len(s):
        if s[j] >= g[i]:
            count += 1
            i += 1
        j += 1
    return count
```
