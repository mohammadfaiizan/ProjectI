# Basic DP Operations

## Top-Down (Recursive + Memo)

Recursive solution with a cache to avoid recomputing subproblems.

```python
def fib_memo(n):
    memo = {}
    
    def dp(i):
        if i <= 1:
            return i
        if i in memo:
            return memo[i]
        memo[i] = dp(i - 1) + dp(i - 2)
        return memo[i]
    
    return dp(n)
```

## Bottom-Up (Iterative Table)

Fill a table in dependency order from base cases to the target.

```python
def fib_table(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]
```

## Space Optimization: Rolling Array

When `dp[i]` depends only on a few previous values, use a fixed-size array.

```python
def fib_rolling(n):
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for i in range(2, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    return prev1
```

## Space Optimization: Two Rows

For 2D DP where row `i` depends only on row `i-1`, use two rows.

```python
def knapsack_two_rows(weights, values, capacity):
    n = len(weights)
    prev = [0] * (capacity + 1)
    curr = [0] * (capacity + 1)
    for i in range(n):
        for w in range(capacity + 1):
            curr[w] = prev[w]
            if w >= weights[i]:
                curr[w] = max(curr[w], prev[w - weights[i]] + values[i])
        prev, curr = curr, prev
    return prev[capacity]
```

## Space Optimization: Two Variables

For 1D DP with limited dependency window.

```python
def climb_stairs_two_vars(n):
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b
```

## 1D DP Setup

Single dimension; state is typically an index or a single parameter.

```python
def min_cost_climbing_stairs(cost):
    n = len(cost)
    dp = [0] * (n + 1)
    for i in range(2, n + 1):
        dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
    return dp[n]
```

## 2D DP Setup

Two dimensions; state is typically `(i, j)` for two indices or index + capacity.

```python
def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[m - 1][n - 1]
```

## Print DP Table for Debugging

```python
def print_dp_table(dp, label="DP"):
    print(f"--- {label} ---")
    if isinstance(dp[0], list):
        for row in dp:
            print(row)
    else:
        print(dp)
    print()
```

## Reconstruct Solution from Table (Backtrack)

After filling the DP table, trace back to recover the choices made.

```python
def knapsack_with_reconstruction(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if w >= weights[i - 1]:
                take = dp[i - 1][w - weights[i - 1]] + values[i - 1]
                if take > dp[i][w]:
                    dp[i][w] = take
    
    chosen = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen.append(i - 1)
            w -= weights[i - 1]
    
    return dp[n][capacity], chosen
```

## Identify States

1. List all parameters that define a subproblem
2. Ensure the state uniquely identifies the subproblem
3. Remove redundant parameters

Example for LCS: state `(i, j)` = LCS length of `s1[:i]` and `s2[:j]`.

## Identify Transitions

1. Enumerate choices at the current state
2. Express the recurrence: how does the current state depend on previous states?
3. Handle base cases

Example for LCS:
- If `s1[i-1] == s2[j-1]`: `dp[i][j] = 1 + dp[i-1][j-1]`
- Else: `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`

## Base Case Identification

Base cases are states that do not depend on other DP states. They are typically:
- Smallest valid inputs (empty string, zero capacity)
- Boundary conditions (index 0, first row/column)

```python
def lcs_base_cases(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = 0
    for j in range(n + 1):
        dp[0][j] = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
```
