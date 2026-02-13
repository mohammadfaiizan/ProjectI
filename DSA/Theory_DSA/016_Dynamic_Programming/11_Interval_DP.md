# Interval DP

## Matrix Chain Multiplication

```python
def matrix_chain_order(dims):
    n = len(dims) - 1
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                dp[i][j] = min(dp[i][j], cost)
    return dp[0][n - 1]
```

## Burst Balloons

```python
def max_coins(nums):
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n):
        for i in range(n - length):
            j = i + length
            for k in range(i + 1, j):
                dp[i][j] = max(dp[i][j], dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j])
    return dp[0][n - 1]
```

## Optimal BST

```python
def optimal_bst(keys, freq):
    n = len(keys)
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    prefix = [0]
    for f in freq:
        prefix.append(prefix[-1] + f)
    for length in range(1, n + 1):
        for i in range(n - length + 1):
            j = i + length
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + prefix[j] - prefix[i]
                dp[i][j] = min(dp[i][j], cost)
    return dp[0][n]
```

## Boolean Parenthesization

```python
def count_ways(operands, operators):
    n = len(operands)
    true_dp = [[0] * n for _ in range(n)]
    false_dp = [[0] * n for _ in range(n)]
    for i in range(n):
        true_dp[i][i] = 1 if operands[i] == 'T' else 0
        false_dp[i][i] = 1 if operands[i] == 'F' else 0
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                op = operators[k]
                if op == '&':
                    true_dp[i][j] += true_dp[i][k] * true_dp[k + 1][j]
                    false_dp[i][j] += true_dp[i][k] * false_dp[k + 1][j] + false_dp[i][k] * true_dp[k + 1][j] + false_dp[i][k] * false_dp[k + 1][j]
                elif op == '|':
                    true_dp[i][j] += true_dp[i][k] * true_dp[k + 1][j] + true_dp[i][k] * false_dp[k + 1][j] + false_dp[i][k] * true_dp[k + 1][j]
                    false_dp[i][j] += false_dp[i][k] * false_dp[k + 1][j]
                else:
                    true_dp[i][j] += true_dp[i][k] * false_dp[k + 1][j] + false_dp[i][k] * true_dp[k + 1][j]
                    false_dp[i][j] += true_dp[i][k] * true_dp[k + 1][j] + false_dp[i][k] * false_dp[k + 1][j]
    return true_dp[0][n - 1]
```

## Minimum Cost to Merge Stones

```python
def merge_stones(stones, k):
    n = len(stones)
    if (n - 1) % (k - 1):
        return -1
    prefix = [0]
    for s in stones:
        prefix.append(prefix[-1] + s)
    dp = [[0] * n for _ in range(n)]
    for length in range(k, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for m in range(i, j, k - 1):
                dp[i][j] = min(dp[i][j], dp[i][m] + dp[m + 1][j])
            if (length - 1) % (k - 1) == 0:
                dp[i][j] += prefix[j + 1] - prefix[i]
    return dp[0][n - 1]
```

## Strange Printer

```python
def strange_printer(s):
    if not s:
        return 0
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = 1 + dp[i + 1][j]
            for k in range(i + 1, j + 1):
                if s[k] == s[i]:
                    dp[i][j] = min(dp[i][j], dp[i][k - 1] + (dp[k + 1][j] if k < j else 0))
    return dp[0][n - 1]
```

## Predict the Winner / Stone Game (Interval)

```python
def predict_the_winner(nums):
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = nums[i]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1])
    return dp[0][n - 1] >= 0
```

## Minimum Cost Tree from Leaf Values

```python
def mct_from_leaf_values(arr):
    n = len(arr)
    dp = [[float('inf')] * n for _ in range(n)]
    max_val = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 0
        max_val[i][i] = arr[i]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            max_val[i][j] = max(max_val[i][j - 1], arr[j])
            for k in range(i, j):
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j] + max_val[i][k] * max_val[k + 1][j])
    return dp[0][n - 1]
```

## Palindrome Removal

```python
def minimum_moves(arr):
    n = len(arr)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = 1 + dp[i + 1][j]
            if arr[i] == arr[j]:
                dp[i][j] = min(dp[i][j], (1 if i + 1 == j else dp[i + 1][j - 1]))
            for k in range(i + 1, j):
                if arr[i] == arr[k]:
                    dp[i][j] = min(dp[i][j], dp[i + 1][k - 1] + (dp[k + 1][j] if k < j else 0))
    return dp[0][n - 1]
```

## Minimum Score Triangulation of Polygon

```python
def min_score_triangulation(values):
    n = len(values)
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n):
        for i in range(n - length):
            j = i + length
            dp[i][j] = float('inf')
            for k in range(i + 1, j):
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j] + values[i] * values[k] * values[j])
    return dp[0][n - 1]
```

## Remove Boxes

```python
def remove_boxes(boxes):
    n = len(boxes)
    dp = [[[0] * n for _ in range(n)] for _ in range(n)]
    
    def solve(i, j, k):
        if i > j:
            return 0
        if dp[i][j][k] > 0:
            return dp[i][j][k]
        res = (k + 1) ** 2 + solve(i + 1, j, 0)
        for m in range(i + 1, j + 1):
            if boxes[m] == boxes[i]:
                res = max(res, solve(i + 1, m - 1, 0) + solve(m, j, k + 1))
        dp[i][j][k] = res
        return res
    
    return solve(0, n - 1, 0)
```
