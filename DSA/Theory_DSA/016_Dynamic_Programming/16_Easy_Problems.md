# Easy DP Problems

## 1. Climbing Stairs

**Description:** Count ways to reach the nth stair using 1 or 2 steps.

**Approach:** Linear recurrence. `dp[i] = dp[i-1] + dp[i-2]`. Space: O(1) with two variables.

```python
def climbStairs(n):
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
```

Time: O(n) | Space: O(1)

---

## 2. Min Cost Climbing Stairs

**Description:** Given cost array, find minimum cost to reach top (start at index 0 or 1).

**Approach:** `dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])`. Return `dp[n]`.

```python
def minCostClimbingStairs(cost):
    a, b = 0, 0
    for i in range(2, len(cost) + 1):
        a, b = b, min(a + cost[i-2], b + cost[i-1])
    return b
```

Time: O(n) | Space: O(1)

---

## 3. House Robber

**Description:** Max money from non-adjacent houses.

**Approach:** `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`. Two variables for O(1) space.

```python
def rob(nums):
    prev, cur = 0, 0
    for x in nums:
        prev, cur = cur, max(cur, prev + x)
    return cur
```

Time: O(n) | Space: O(1)

---

## 4. Pascal's Triangle

**Description:** Generate Pascal's triangle rows.

**Approach:** Each row: `row[j] = prev[j-1] + prev[j]` with 1s at boundaries.

```python
def generate(numRows):
    res = [[1]]
    for i in range(1, numRows):
        row = [1]
        for j in range(1, i):
            row.append(res[-1][j-1] + res[-1][j])
        row.append(1)
        res.append(row)
    return res
```

Time: O(n^2) | Space: O(1)

---

## 5. Maximum Subarray (Kadane)

**Description:** Find contiguous subarray with largest sum.

**Approach:** `dp[i] = max(nums[i], dp[i-1] + nums[i])`. Track global max.

```python
def maxSubArray(nums):
    cur = res = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        res = max(res, cur)
    return res
```

Time: O(n) | Space: O(1)

---

## 6. Best Time to Buy and Sell Stock

**Description:** One transaction, max profit.

**Approach:** Track min price seen, max profit = price - min_price.

```python
def maxProfit(prices):
    min_p, res = float('inf'), 0
    for p in prices:
        min_p = min(min_p, p)
        res = max(res, p - min_p)
    return res
```

Time: O(n) | Space: O(1)

---

## 7. Best Time to Buy and Sell Stock II

**Description:** Unlimited transactions.

**Approach:** Greedy - add all positive price differences.

```python
def maxProfit(prices):
    return sum(max(0, prices[i] - prices[i-1]) for i in range(1, len(prices)))
```

Time: O(n) | Space: O(1)

---

## 8. Divisor Game

**Description:** Two players, subtract divisor of n. Can first player win?

**Approach:** `dp[i]` = first wins for n=i. Try all divisors; if any move leads to loss for opponent, win.

```python
def divisorGame(n):
    dp = [False] * (n + 1)
    for i in range(2, n + 1):
        for j in range(1, int(i**0.5) + 1):
            if i % j == 0 and not dp[i - j]:
                dp[i] = True
                break
    return dp[n]
```

Time: O(n * sqrt(n)) | Space: O(n)

---

## 9. Range Sum Query - Immutable

**Description:** Answer sum queries for subarray.

**Approach:** Prefix sum array. Query [i,j] = prefix[j+1] - prefix[i].

```python
class NumArray:
    def __init__(self, nums):
        self.prefix = [0]
        for x in nums:
            self.prefix.append(self.prefix[-1] + x)
    def sumRange(self, left, right):
        return self.prefix[right + 1] - self.prefix[left]
```

Time: O(1) per query | Space: O(n)

---

## 10. Counting Bits

**Description:** For each i in [0,n], count 1-bits in binary representation.

**Approach:** `dp[i] = dp[i >> 1] + (i & 1)`.

```python
def countBits(n):
    res = [0] * (n + 1)
    for i in range(1, n + 1):
        res[i] = res[i >> 1] + (i & 1)
    return res
```

Time: O(n) | Space: O(1)

---

## 11. Decode Ways

**Description:** Count ways to decode digit string to letters (1-26).

**Approach:** `dp[i]` = ways for prefix of length i. Check 1-digit and 2-digit valid decodings.

```python
def numDecodings(s):
    dp = [1, 1] if s[0] != '0' else [1, 0]
    for i in range(1, len(s)):
        cur = 0
        if s[i] != '0':
            cur += dp[-1]
        if 10 <= int(s[i-1:i+1]) <= 26:
            cur += dp[-2]
        dp.append(cur)
    return dp[-1] if s else 0
```

Time: O(n) | Space: O(n)

---

## 12. Unique Paths

**Description:** Count paths from top-left to bottom-right in m x n grid.

**Approach:** `dp[i][j] = dp[i-1][j] + dp[i][j-1]`. First row/col = 1.

```python
def uniquePaths(m, n):
    dp = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j-1]
    return dp[-1]
```

Time: O(m * n) | Space: O(n)

---

## 13. Unique Paths II

**Description:** Same with obstacles.

**Approach:** Skip obstacle cells. Initialize first row/col accounting for obstacles.

```python
def uniquePathsWithObstacles(grid):
    m, n = len(grid), len(grid[0])
    dp = [0] * n
    dp[0] = 1
    for i in range(m):
        for j in range(n):
            if grid[i][j]:
                dp[j] = 0
            elif j:
                dp[j] += dp[j-1]
    return dp[-1]
```

Time: O(m * n) | Space: O(n)

---

## 14. Minimum Path Sum

**Description:** Min sum path in grid from top-left to bottom-right.

**Approach:** `dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])`.

```python
def minPathSum(grid):
    m, n = len(grid), len(grid[0])
    dp = [float('inf')] * n
    dp[0] = 0
    for i in range(m):
        dp[0] += grid[i][0]
        for j in range(1, n):
            dp[j] = grid[i][j] + min(dp[j], dp[j-1])
    return dp[-1]
```

Time: O(m * n) | Space: O(n)

---

## 15. Triangle (Minimum Total)

**Description:** Min path sum from top to bottom of triangle.

**Approach:** Bottom-up. `dp[j] = triangle[i][j] + min(dp[j], dp[j+1])`.

```python
def minimumTotal(triangle):
    dp = triangle[-1][:]
    for i in range(len(triangle)-2, -1, -1):
        for j in range(len(triangle[i])):
            dp[j] = triangle[i][j] + min(dp[j], dp[j+1])
    return dp[0]
```

Time: O(n^2) | Space: O(n)

---

## 16. Maximum Product Subarray

**Description:** Max product of contiguous subarray.

**Approach:** Track max and min (negative * negative = positive). `max_dp = max(nums[i], max_dp * nums[i], min_dp * nums[i])`.

```python
def maxProduct(nums):
    res = max_dp = min_dp = nums[0]
    for x in nums[1:]:
        max_dp, min_dp = max(x, max_dp*x, min_dp*x), min(x, max_dp*x, min_dp*x)
        res = max(res, max_dp)
    return res
```

Time: O(n) | Space: O(1)

---

## 17. Word Break

**Description:** Can string be segmented into dictionary words?

**Approach:** `dp[i]` = can segment prefix of length i. For each j < i, if dp[j] and s[j:i] in dict, dp[i] = True.

```python
def wordBreak(s, wordDict):
    words = set(wordDict)
    dp = [True] + [False] * len(s)
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break
    return dp[-1]
```

Time: O(n^2) | Space: O(n)

---

## 18. Coin Change

**Description:** Min coins to make amount.

**Approach:** `dp[i] = min(dp[i-c] + 1)` for each coin c. Initialize dp[0] = 0.

```python
def coinChange(coins, amount):
    dp = [0] + [float('inf')] * amount
    for i in range(1, amount + 1):
        for c in coins:
            if i >= c:
                dp[i] = min(dp[i], dp[i-c] + 1)
    return dp[-1] if dp[-1] != float('inf') else -1
```

Time: O(amount * coins) | Space: O(amount)

---

## 19. Longest Increasing Subsequence (O(n^2))

**Description:** Length of longest strictly increasing subsequence.

**Approach:** `dp[i] = 1 + max(dp[j])` for j < i where nums[j] < nums[i].

```python
def lengthOfLIS(nums):
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
```

Time: O(n^2) | Space: O(n)

---

## 20. Is Subsequence

**Description:** Is s subsequence of t?

**Approach:** Two pointers or DP. `dp[i][j]` = is s[:i] subseq of t[:j].

```python
def isSubsequence(s, t):
    i = 0
    for c in t:
        if i < len(s) and s[i] == c:
            i += 1
    return i == len(s)
```

Time: O(n) | Space: O(1)

---

## 21. Nth Tribonacci Number

**Description:** Tribonacci: T(n) = T(n-1) + T(n-2) + T(n-3).

**Approach:** Three variables, iterate.

```python
def tribonacci(n):
    a, b, c = 0, 1, 1
    for _ in range(n - 2):
        a, b, c = b, c, a + b + c
    return c if n > 0 else 0
```

Time: O(n) | Space: O(1)

---

## 22. Get Maximum in Generated Array

**Description:** Generate array with rules, return max.

**Approach:** Simulate generation, track max.

```python
def getMaximumGenerated(n):
    if n == 0:
        return 0
    nums = [0] * (n + 1)
    nums[1] = 1
    for i in range(2, n + 1):
        nums[i] = nums[i//2] + (nums[i//2 + 1] if i % 2 else 0)
    return max(nums)
```

Time: O(n) | Space: O(n)

---

## 23. Delete and Earn

**Description:** Pick nums, delete num and num-1. Maximize sum.

**Approach:** Convert to points array (index = value, value = sum of that value). House robber on points.

```python
def deleteAndEarn(nums):
    from collections import Counter
    c = Counter(nums)
    mx = max(nums)
    points = [0] * (mx + 1)
    for i in range(mx + 1):
        points[i] = i * c.get(i, 0)
    prev, cur = 0, 0
    for p in points:
        prev, cur = cur, max(cur, prev + p)
    return cur
```

Time: O(n) | Space: O(max(nums))

---

## 24. Perfect Squares

**Description:** Min number of perfect squares that sum to n.

**Approach:** `dp[i] = min(dp[i - j*j] + 1)` for j*j <= i.

```python
def numSquares(n):
    dp = [0] + [float('inf')] * n
    for i in range(1, n + 1):
        j = 1
        while j * j <= i:
            dp[i] = min(dp[i], dp[i - j*j] + 1)
            j += 1
    return dp[-1]
```

Time: O(n * sqrt(n)) | Space: O(n)

---

## 25. Maximum Sum of Non-Adjacent Elements (1D)

**Description:** Max sum picking non-adjacent elements.

**Approach:** Same as house robber.

```python
def maxSumNonAdjacent(nums):
    prev, cur = 0, 0
    for x in nums:
        prev, cur = cur, max(cur, prev + x)
    return cur
```

Time: O(n) | Space: O(1)
