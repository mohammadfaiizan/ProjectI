# Medium DP Problems

## 1. Longest Palindromic Substring

**Description:** Find longest palindromic substring.

**Approach:** Expand around center or 2D DP. `dp[i][j]` = is s[i:j+1] palindrome. Fill by length.

```python
def longestPalindrome(s):
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return s[l+1:r]
    res = ""
    for i in range(len(s)):
        res = max(expand(i, i), expand(i, i+1), res, key=len)
    return res
```

Time: O(n^2) | Space: O(1)

---

## 2. Longest Palindromic Subsequence

**Description:** Length of longest palindromic subsequence.

**Approach:** Interval DP. `dp[i][j] = 2 + dp[i+1][j-1]` if match, else `max(dp[i+1][j], dp[i][j-1])`.

```python
def longestPalindromeSubseq(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for L in range(2, n + 1):
        for i in range(n - L + 1):
            j = i + L - 1
            if s[i] == s[j]:
                dp[i][j] = 2 + (dp[i+1][j-1] if L > 2 else 0)
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    return dp[0][n-1]
```

Time: O(n^2) | Space: O(n^2)

---

## 3. Longest Common Subsequence

**Description:** Length of LCS of two strings.

**Approach:** 2D DP. `dp[i][j]` = LCS of s1[:i], s2[:j]. Match or skip.

```python
def longestCommonSubsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

Time: O(m * n) | Space: O(m * n)

---

## 4. Edit Distance (Levenshtein)

**Description:** Min insert/delete/replace to transform s1 to s2.

**Approach:** `dp[i][j]` = min edits for s1[:i], s2[:j]. Three transitions: insert, delete, replace.

```python
def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
```

Time: O(m * n) | Space: O(m * n)

---

## 5. Coin Change II

**Description:** Count ways to make amount with coins.

**Approach:** `dp[i] += dp[i-c]` for each coin. Order: iterate coins first to avoid permutation count.

```python
def change(amount, coins):
    dp = [1] + [0] * amount
    for c in coins:
        for i in range(c, amount + 1):
            dp[i] += dp[i - c]
    return dp[amount]
```

Time: O(amount * coins) | Space: O(amount)

---

## 6. Target Sum

**Description:** Assign + or - to each number to reach target.

**Approach:** Convert to subset sum. Find subsets with sum = (total + target) / 2.

```python
def findTargetSumWays(nums, target):
    total = sum(nums)
    if (total + target) % 2 or abs(target) > total:
        return 0
    t = (total + target) // 2
    dp = [1] + [0] * t
    for n in nums:
        for i in range(t, n - 1, -1):
            dp[i] += dp[i - n]
    return dp[t]
```

Time: O(n * t) | Space: O(t)

---

## 7. Partition Equal Subset Sum

**Description:** Can array be partitioned into two equal-sum subsets?

**Approach:** Subset sum with target = total/2.

```python
def canPartition(nums):
    total = sum(nums)
    if total % 2:
        return False
    t = total // 2
    dp = [True] + [False] * t
    for n in nums:
        for i in range(t, n - 1, -1):
            dp[i] |= dp[i - n]
    return dp[t]
```

Time: O(n * t) | Space: O(t)

---

## 8. Word Break II

**Description:** Return all possible sentence formations.

**Approach:** Memoized DFS. For each prefix that is a word, recurse on suffix and combine.

```python
def wordBreak(s, wordDict):
    words = set(wordDict)
    def dfs(s, memo):
        if s in memo:
            return memo[s]
        if not s:
            return [""]
        res = []
        for i in range(1, len(s) + 1):
            if s[:i] in words:
                for tail in dfs(s[i:], memo):
                    res.append((s[:i] + " " + tail).strip())
        memo[s] = res
        return res
    return dfs(s, {})
```

Time: O(n^2 * 2^n) | Space: O(2^n)

---

## 9. Unique Paths II (Obstacles)

**Description:** Unique paths with obstacles.

**Approach:** Same as unique paths, skip obstacle cells.

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

## 10. Minimum Path Sum

**Description:** Min sum path in grid.

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

## 11. House Robber II

**Description:** House robber with circular arrangement.

**Approach:** Run house robber on nums[1:] and nums[:-1], take max.

```python
def rob(nums):
    def rob_linear(arr):
        prev, cur = 0, 0
        for x in arr:
            prev, cur = cur, max(cur, prev + x)
        return cur
    return max(rob_linear(nums[1:]), rob_linear(nums[:-1])) if len(nums) > 1 else (nums[0] if nums else 0)
```

Time: O(n) | Space: O(1)

---

## 12. Maximum Product Subarray

**Description:** Max product of contiguous subarray.

**Approach:** Track max and min. Negatives can flip to max.

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

## 13. Longest Increasing Subsequence (O(n log n))

**Description:** LIS length with binary search.

**Approach:** Maintain tails array. For each x, binary search position, extend or replace.

```python
def lengthOfLIS(nums):
    import bisect
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)
```

Time: O(n log n) | Space: O(n)

---

## 14. Maximum Length of Repeated Subarray

**Description:** Max length subarray that appears in both arrays.

**Approach:** 2D DP. `dp[i][j] = 1 + dp[i-1][j-1]` if match. Track max.

```python
def findLength(nums1, nums2):
    m, n = len(nums1), len(nums2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    res = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if nums1[i-1] == nums2[j-1]:
                dp[i][j] = 1 + dp[i-1][j-1]
                res = max(res, dp[i][j])
    return res
```

Time: O(m * n) | Space: O(m * n)

---

## 15. Count Square Submatrices with All Ones

**Description:** Count all square submatrices of 1s.

**Approach:** `dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])` if grid[i][j]=1. Sum all dp.

```python
def countSquares(matrix):
    m, n = len(matrix), len(matrix[0])
    res = 0
    for i in range(m):
        for j in range(n):
            if matrix[i][j] and i and j:
                matrix[i][j] = 1 + min(matrix[i-1][j], matrix[i][j-1], matrix[i-1][j-1])
            res += matrix[i][j]
    return res
```

Time: O(m * n) | Space: O(1)

---

## 16. Minimum Cost for Tickets

**Description:** Min cost to travel on given days with 1/7/30 day passes.

**Approach:** `dp[i]` = min cost for first i days. For each day, try 1/7/30 day pass.

```python
def mincostTickets(days, costs):
    days_set = set(days)
    dp = [0] * (max(days) + 1)
    for d in range(1, len(dp)):
        if d not in days_set:
            dp[d] = dp[d-1]
        else:
            dp[d] = min(dp[max(0,d-1)]+costs[0], dp[max(0,d-7)]+costs[1], dp[max(0,d-30)]+costs[2])
    return dp[-1]
```

Time: O(max(days)) | Space: O(max(days))

---

## 17. Integer Break

**Description:** Break n into positive integers, maximize product.

**Approach:** `dp[i] = max(j * dp[i-j], j * (i-j))` for j in 1..i-1.

```python
def integerBreak(n):
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        for j in range(1, i):
            dp[i] = max(dp[i], j * dp[i-j], j * (i - j))
    return dp[n]
```

Time: O(n^2) | Space: O(n)

---

## 18. Combination Sum IV

**Description:** Count combinations that sum to target (order matters).

**Approach:** `dp[i] += dp[i-n]` for each n in nums. Unbounded, order matters.

```python
def combinationSum4(nums, target):
    dp = [1] + [0] * target
    for i in range(1, target + 1):
        for n in nums:
            if i >= n:
                dp[i] += dp[i - n]
    return dp[target]
```

Time: O(target * n) | Space: O(target)

---

## 19. Decode Ways II

**Description:** Decode with * wildcard (1-9).

**Approach:** Extend decode ways. Handle * for single and double digit.

```python
def numDecodings(s):
    MOD = 10**9 + 7
    dp = [1, 9 if s[0] == '*' else (1 if s[0] != '0' else 0)]
    for i in range(1, len(s)):
        cur = 0
        if s[i] == '*':
            cur += dp[-1] * 9
            if s[i-1] == '1':
                cur += dp[-2] * 9
            elif s[i-1] == '2':
                cur += dp[-2] * 6
            elif s[i-1] == '*':
                cur += dp[-2] * 15
        else:
            if s[i] != '0':
                cur += dp[-1]
            if s[i-1] == '1' or (s[i-1] == '2' and s[i] <= '6'):
                cur += dp[-2]
            elif s[i-1] == '*' and s[i] <= '6':
                cur += dp[-2] * 2
            elif s[i-1] == '*' and s[i] > '6':
                cur += dp[-2]
        dp.append(cur % MOD)
    return dp[-1] % MOD if s else 0
```

Time: O(n) | Space: O(n)

---

## 20. Maximum Sum Circular Subarray

**Description:** Max sum subarray in circular array.

**Approach:** Either max subarray in linear array, or total - min subarray (wrap-around case).

```python
def maxSubarraySumCircular(nums):
    total = sum(nums)
    max_cur = max_glob = min_cur = min_glob = nums[0]
    for x in nums[1:]:
        max_cur = max(x, max_cur + x)
        max_glob = max(max_glob, max_cur)
        min_cur = min(x, min_cur + x)
        min_glob = min(min_glob, min_cur)
    return max(max_glob, total - min_glob) if max_glob > 0 else max_glob
```

Time: O(n) | Space: O(1)

---

## 21. Best Time to Buy and Sell Stock with Cooldown

**Description:** One day cooldown after sell.

**Approach:** State machine: sold, held, rest. `sold = held + price`, `held = max(held, rest - price)`, `rest = max(rest, sold)`.

```python
def maxProfit(prices):
    sold, held, rest = 0, float('-inf'), 0
    for p in prices:
        sold, held, rest = held + p, max(held, rest - p), max(rest, sold)
    return max(sold, rest)
```

Time: O(n) | Space: O(1)

---

## 22. Best Time to Buy and Sell Stock with Transaction Fee

**Description:** Fee per transaction.

**Approach:** `cash = max(cash, hold + price - fee)`, `hold = max(hold, cash - price)`.

```python
def maxProfit(prices, fee):
    cash, hold = 0, -prices[0]
    for p in prices[1:]:
        cash = max(cash, hold + p - fee)
        hold = max(hold, cash - p)
    return cash
```

Time: O(n) | Space: O(1)

---

## 23. Interleaving String

**Description:** Is s3 interleaving of s1 and s2?

**Approach:** `dp[i][j]` = can form s3[:i+j] from s1[:i], s2[:j]. Match s1[i-1] or s2[j-1] with s3[i+j-1].

```python
def isInterleave(s1, s2, s3):
    if len(s1) + len(s2) != len(s3):
        return False
    m, n = len(s1), len(s2)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for i in range(m + 1):
        for j in range(n + 1):
            if i + j == 0:
                continue
            k = i + j - 1
            if i and s1[i-1] == s3[k]:
                dp[i][j] |= dp[i-1][j]
            if j and s2[j-1] == s3[k]:
                dp[i][j] |= dp[i][j-1]
    return dp[m][n]
```

Time: O(m * n) | Space: O(m * n)

---

## 24. Distinct Subsequences

**Description:** Count distinct subsequences of s equal to t.

**Approach:** `dp[i][j]` = count for s[:i], t[:j]. If s[i-1]==t[j-1], add dp[i-1][j-1]. Always add dp[i-1][j].

```python
def numDistinct(s, t):
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j]
            if s[i-1] == t[j-1]:
                dp[i][j] += dp[i-1][j-1]
    return dp[m][n]
```

Time: O(m * n) | Space: O(m * n)

---

## 25. Palindrome Partitioning II

**Description:** Min cuts to partition string into palindromes.

**Approach:** Precompute is_pal[i][j]. `dp[i] = min(dp[j] + 1)` for j where s[j:i] is palindrome.

```python
def minCut(s):
    n = len(s)
    pal = [[False] * n for _ in range(n)]
    for L in range(1, n + 1):
        for i in range(n - L + 1):
            j = i + L - 1
            pal[i][j] = (s[i] == s[j]) and (L <= 2 or pal[i+1][j-1])
    dp = [0] + [float('inf')] * n
    for i in range(1, n + 1):
        for j in range(i):
            if pal[j][i-1]:
                dp[i] = min(dp[i], dp[j] + 1)
    return dp[n] - 1
```

Time: O(n^2) | Space: O(n^2)

---

# Hard Problems

## 1. Edit Distance

**Description:** Min operations to transform one string to another.

**Approach:** Standard Levenshtein DP.

```python
def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(word1[i-1]!=word2[j-1]))
    return dp[m][n]
```

Time: O(m * n) | Space: O(m * n)

---

## 2. Longest Valid Parentheses

**Description:** Length of longest valid parentheses substring.

**Approach:** `dp[i]` = longest valid ending at i. Track matching open bracket.

```python
def longestValidParentheses(s):
    dp = [0] * (len(s) + 1)
    for i in range(1, len(s)):
        if s[i] == ')' and i - dp[i-1] - 1 >= 0 and s[i - dp[i-1] - 1] == '(':
            dp[i+1] = dp[i-1] + 2 + dp[i - dp[i-1] - 1]
    return max(dp) if dp else 0
```

Time: O(n) | Space: O(n)

---

## 3. Wildcard Matching

**Description:** Match string with * (any sequence) and ? (any char).

**Approach:** 2D DP. * can match empty or extend. Handle * carefully.

```python
def isMatch(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] and p[j-1] == '*'
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[i][j] = dp[i][j-1] or dp[i-1][j]
            elif p[j-1] in ('?', s[i-1]):
                dp[i][j] = dp[i-1][j-1]
    return dp[m][n]
```

Time: O(m * n) | Space: O(m * n)

---

## 4. Regular Expression Matching

**Description:** Match with . and * (zero or more of preceding).

**Approach:** 2D DP. Handle * by matching zero or one+ of preceding char.

```python
def isMatch(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(2, n + 1):
        dp[0][j] = dp[0][j-2] and p[j-1] == '*'
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[i][j] = dp[i][j-2] or (dp[i-1][j] and p[j-2] in (s[i-1], '.'))
            else:
                dp[i][j] = dp[i-1][j-1] and p[j-1] in (s[i-1], '.')
    return dp[m][n]
```

Time: O(m * n) | Space: O(m * n)

---

## 5. Minimum Insertion Steps to Make a String Palindrome

**Description:** Min insertions to make palindrome.

**Approach:** n - longest palindromic subsequence.

```python
def minInsertions(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for L in range(2, n + 1):
        for i in range(n - L + 1):
            j = i + L - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i+1][j], dp[i][j-1])
    return dp[0][n-1]
```

Time: O(n^2) | Space: O(n^2)

---

## 6. Burst Balloons

**Description:** Burst balloons to maximize coins. Coin = left * curr * right.

**Approach:** Interval DP. Add 1s at ends. `dp[i][j] = max(dp[i][k] + dp[k][j] + nums[i]*nums[k]*nums[j])`.

```python
def maxCoins(nums):
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    for L in range(2, n):
        for i in range(n - L):
            j = i + L
            for k in range(i + 1, j):
                dp[i][j] = max(dp[i][j], nums[i]*nums[k]*nums[j] + dp[i][k] + dp[k][j])
    return dp[0][n-1]
```

Time: O(n^3) | Space: O(n^2)

---

## 7. Cherry Pickup II

**Description:** Two robots from top to bottom, collect cherries, no double count.

**Approach:** 3D DP. State: (row, col1, col2). Both move down, try all column pairs.

```python
def cherryPickup(grid):
    m, n = len(grid), len(grid[0])
    dp = [[[float('-inf')] * n for _ in range(n)] for _ in range(m)]
    dp[0][0][n-1] = grid[0][0] + grid[0][n-1]
    for r in range(1, m):
        for c1 in range(n):
            for c2 in range(n):
                best = float('-inf')
                for d1 in [-1, 0, 1]:
                    for d2 in [-1, 0, 1]:
                        pc1, pc2 = c1 + d1, c2 + d2
                        if 0 <= pc1 < n and 0 <= pc2 < n:
                            cur = dp[r-1][pc1][pc2]
                            cur += grid[r][c1] + (grid[r][c2] if c1 != c2 else 0)
                            best = max(best, cur)
                dp[r][c1][c2] = best
    return max(dp[m-1][c1][c2] for c1 in range(n) for c2 in range(n))
```

Time: O(m * n^2) | Space: O(m * n^2)

---

## 8. Maximum Profit in Job Scheduling

**Description:** Jobs with start, end, profit. Max profit non-overlapping.

**Approach:** Sort by end. `dp[i] = max(dp[i-1], profit[i] + dp[j])` where j is latest non-overlapping.

```python
def jobScheduling(startTime, endTime, profit):
    import bisect
    jobs = sorted(zip(endTime, startTime, profit))
    dp = [[0, 0]]
    for e, s, p in jobs:
        i = bisect.bisect_right(dp, [s + 1]) - 1
        if dp[i][1] + p > dp[-1][1]:
            dp.append([e, dp[i][1] + p])
    return dp[-1][1]
```

Time: O(n log n) | Space: O(n)

---

## 9. Minimum Cost to Merge Stones

**Description:** Merge k consecutive piles, min cost.

**Approach:** Interval DP. Merge only when (length-1) % (k-1) == 0.

```python
def mergeStones(stones, k):
    n = len(stones)
    if (n - 1) % (k - 1):
        return -1
    prefix = [0]
    for x in stones:
        prefix.append(prefix[-1] + x)
    dp = [[0] * n for _ in range(n)]
    for L in range(k, n + 1):
        for i in range(n - L + 1):
            j = i + L - 1
            dp[i][j] = float('inf')
            for m in range(i, j, k - 1):
                dp[i][j] = min(dp[i][j], dp[i][m] + dp[m+1][j])
            if (L - 1) % (k - 1) == 0:
                dp[i][j] += prefix[j+1] - prefix[i]
    return dp[0][n-1]
```

Time: O(n^3) | Space: O(n^2)

---

## 10. Dungeon Game

**Description:** Min initial HP to reach princess in dungeon.

**Approach:** Reverse DP. `dp[i][j] = max(1, min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j])`.

```python
def calculateMinimumHP(dungeon):
    m, n = len(dungeon), len(dungeon[0])
    dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
    dp[m][n-1] = dp[m-1][n] = 1
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            dp[i][j] = max(1, min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j])
    return dp[0][0]
```

Time: O(m * n) | Space: O(m * n)

---

## 11. Count Different Palindromic Subsequences

**Description:** Count distinct palindromic subsequences.

**Approach:** Interval DP with character tracking. Handle duplicates carefully.

```python
def countPalindromicSubsequences(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for L in range(2, n + 1):
        for i in range(n - L + 1):
            j = i + L - 1
            if s[i] == s[j]:
                lo, hi = i + 1, j - 1
                while lo <= hi and s[lo] != s[i]:
                    lo += 1
                while hi >= lo and s[hi] != s[i]:
                    hi -= 1
                if lo > hi:
                    dp[i][j] = 2 * dp[i+1][j-1] + 2
                elif lo == hi:
                    dp[i][j] = 2 * dp[i+1][j-1] + 1
                else:
                    dp[i][j] = 2 * dp[i+1][j-1] - dp[lo+1][hi-1]
            else:
                dp[i][j] = dp[i+1][j] + dp[i][j-1] - dp[i+1][j-1]
            dp[i][j] %= 10**9 + 7
    return dp[0][n-1]
```

Time: O(n^2) | Space: O(n^2)

---

## 12. Minimum Cost to Cut a Stick

**Description:** Cut stick at given positions, cost = stick length. Min total cost.

**Approach:** Interval DP. Sort cuts. `dp[i][j] = cuts[j]-cuts[i] + min(dp[i][k] + dp[k][j])`.

```python
def minCost(n, cuts):
    cuts = sorted([0] + cuts + [n])
    m = len(cuts)
    dp = [[0] * m for _ in range(m)]
    for L in range(2, m):
        for i in range(m - L):
            j = i + L
            dp[i][j] = min(dp[i][k] + dp[k][j] for k in range(i+1, j)) + cuts[j] - cuts[i]
    return dp[0][m-1]
```

Time: O(m^3) | Space: O(m^2)

---

## 13. Maximum Score from Performing Multiplication Operations

**Description:** Pick from ends of array, multiply by multiplier[i], max score.

**Approach:** `dp[i][j]` = max score with i picks from left, j from right. Or dp[left][right] = remaining operations.

```python
def maximumScore(nums, multipliers):
    n, m = len(nums), len(multipliers)
    dp = [[0] * (m + 1) for _ in range(m + 1)]
    for op in range(m - 1, -1, -1):
        for left in range(op + 1):
            right = n - 1 - (op - left)
            dp[op][left] = max(
                nums[left] * multipliers[op] + dp[op+1][left+1],
                nums[right] * multipliers[op] + dp[op+1][left]
            )
    return dp[0][0]
```

Time: O(m^2) | Space: O(m^2)

---

## 14. Number of Ways to Stay in the Same Place

**Description:** Steps, arrLen. Ways to stay at 0 after steps.

**Approach:** `dp[step][pos]` = ways. Can move left, right, or stay. Space optimize to 1D.

```python
def numWays(steps, arrLen):
    MOD = 10**9 + 7
    n = min(arrLen, steps // 2 + 1)
    dp = [0] * n
    dp[0] = 1
    for _ in range(steps):
        ndp = [0] * n
        for i in range(n):
            ndp[i] = dp[i]
            if i > 0:
                ndp[i] = (ndp[i] + dp[i-1]) % MOD
            if i < n - 1:
                ndp[i] = (ndp[i] + dp[i+1]) % MOD
        dp = ndp
    return dp[0]
```

Time: O(steps * arrLen) | Space: O(arrLen)

---

## 15. Minimum Number of Refueling Stops

**Description:** Car with tank, stations with fuel. Min stops to reach target.

**Approach:** `dp[i]` = max distance with i stops. Greedy with heap or DP.

```python
def minRefuelStops(target, startFuel, stations):
    dp = [startFuel] + [0] * len(stations)
    for i, (pos, fuel) in enumerate(stations):
        for t in range(i, -1, -1):
            if dp[t] >= pos:
                dp[t+1] = max(dp[t+1], dp[t] + fuel)
    for t, d in enumerate(dp):
        if d >= target:
            return t
    return -1
```

Time: O(n^2) | Space: O(n)
