# Fibonacci and Linear Recurrence Variants

## Fibonacci (Recursive)

```python
def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)
```

## Fibonacci (Memo)

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

## Fibonacci (Table)

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

## Fibonacci (O(1) Space)

```python
def fib_o1_space(n):
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev1 + prev2
    return prev1
```

## Fibonacci (Matrix Exponentiation)

```python
def mat_mult(a, b):
    return [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]]
    ]

def mat_pow(m, n):
    if n == 1:
        return m
    half = mat_pow(m, n // 2)
    result = mat_mult(half, half)
    if n % 2:
        result = mat_mult(result, m)
    return result

def fib_matrix(n):
    if n <= 1:
        return n
    m = [[1, 1], [1, 0]]
    result = mat_pow(m, n - 1)
    return result[0][0]
```

## Climbing Stairs (1 or 2 Steps)

```python
def climb_stairs(n):
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b
```

## Climbing Stairs K Steps

```python
def climb_stairs_k(n, k):
    if n <= 1:
        return 1
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(k, i) + 1):
            dp[i] += dp[i - j]
    return dp[n]
```

## Min Cost Climbing Stairs

```python
def min_cost_climbing_stairs(cost):
    n = len(cost)
    dp = [0] * (n + 1)
    for i in range(2, n + 1):
        dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
    return dp[n]
```

## House Robber I

```python
def rob(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    prev2, prev1 = 0, nums[0]
    for i in range(1, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, curr
    return prev1
```

## House Robber II (Circular)

```python
def rob_circular(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    def rob_range(arr):
        prev2, prev1 = 0, 0
        for x in arr:
            curr = max(prev1, prev2 + x)
            prev2, prev1 = prev1, curr
        return prev1
    
    return max(rob_range(nums[:-1]), rob_range(nums[1:]))
```

## Decode Ways

```python
def num_decodings(s):
    if not s or s[0] == '0':
        return 0
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    for i in range(2, n + 1):
        one = int(s[i - 1])
        two = int(s[i - 2:i])
        if 1 <= one <= 9:
            dp[i] += dp[i - 1]
        if 10 <= two <= 26:
            dp[i] += dp[i - 2]
    return dp[n]
```

## Tiling 2xN with 2x1

```python
def tiling_2xn(n):
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b
```

## Tribonacci

```python
def tribonacci(n):
    if n == 0:
        return 0
    if n <= 2:
        return 1
    a, b, c = 0, 1, 1
    for _ in range(3, n + 1):
        a, b, c = b, c, a + b + c
    return c
```

## Jump Game II (Min Jumps)

```python
def jump(nums):
    n = len(nums)
    if n <= 1:
        return 0
    dp = [float('inf')] * n
    dp[0] = 0
    for i in range(n):
        for j in range(1, nums[i] + 1):
            if i + j < n:
                dp[i + j] = min(dp[i + j], dp[i] + 1)
    return dp[n - 1]
```

## Frog Jump (Varying Steps)

```python
def frog_jump(stones):
    n = len(stones)
    dp = {0: {0}}
    for i in range(1, n):
        dp[i] = set()
        for j in range(i):
            k = stones[i] - stones[j]
            if k in dp[j] or k - 1 in dp[j] or k + 1 in dp[j]:
                dp[i].add(k)
    return len(dp[n - 1]) > 0
```

## Maximum Alternating Subsequence Sum

```python
def max_alternating_sum(nums):
    even, odd = 0, 0
    for x in nums:
        even, odd = max(even, odd - x), max(odd, even + x)
    return max(even, odd)
```

## Delete and Earn

```python
def delete_and_earn(nums):
    if not nums:
        return 0
    max_val = max(nums)
    points = [0] * (max_val + 1)
    for x in nums:
        points[x] += x
    prev2, prev1 = 0, points[0]
    for i in range(1, max_val + 1):
        curr = max(prev1, prev2 + points[i])
        prev2, prev1 = prev1, curr
    return prev1
```
