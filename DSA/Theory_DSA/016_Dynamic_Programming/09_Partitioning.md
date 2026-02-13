# Partitioning DP

## Partition Equal Subset Sum

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

## Minimum Subset Sum Difference

```python
def min_subset_sum_diff(nums):
    total = sum(nums)
    n = len(nums)
    dp = [False] * (total // 2 + 1)
    dp[0] = True
    for x in nums:
        for s in range(total // 2, x - 1, -1):
            dp[s] = dp[s] or dp[s - x]
    for s in range(total // 2, -1, -1):
        if dp[s]:
            return total - 2 * s
    return total
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

## Palindrome Partitioning II (Min Cuts)

```python
def min_cut(s):
    n = len(s)
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n):
        is_pal[i][i] = True
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and (length == 2 or is_pal[i + 1][j - 1]):
                is_pal[i][j] = True
    dp = [float('inf')] * (n + 1)
    dp[0] = -1
    for i in range(1, n + 1):
        for j in range(i):
            if is_pal[j][i - 1]:
                dp[i] = min(dp[i], dp[j] + 1)
    return dp[n]
```

## Palindrome Partitioning III (At Most K)

```python
def palindrome_partition(s, k):
    n = len(s)
    cost = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            cost[i][j] = cost[i + 1][j - 1] + (0 if s[i] == s[j] else 1)
    dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        for parts in range(1, min(k, i) + 1):
            for j in range(parts - 1, i):
                dp[i][parts] = min(dp[i][parts], dp[j][parts - 1] + cost[j][i - 1])
    return dp[n][k]
```

## Partition to K Equal Sum Subsets

```python
def can_partition_k_subsets(nums, k):
    total = sum(nums)
    if total % k:
        return False
    target = total // k
    nums.sort(reverse=True)
    if nums[0] > target:
        return False
    used = [False] * len(nums)
    
    def backtrack(start, curr_sum, count):
        if count == k:
            return True
        if curr_sum == target:
            return backtrack(0, 0, count + 1)
        for i in range(start, len(nums)):
            if used[i] or curr_sum + nums[i] > target:
                continue
            used[i] = True
            if backtrack(i + 1, curr_sum + nums[i], count):
                return True
            used[i] = False
        return False
    
    return backtrack(0, 0, 0)
```

## Partition Array for Max Sum

```python
def max_sum_after_partitioning(arr, k):
    n = len(arr)
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        curr_max = 0
        for j in range(1, min(k, i) + 1):
            curr_max = max(curr_max, arr[i - j])
            dp[i] = max(dp[i], dp[i - j] + curr_max * j)
    return dp[n]
```

## Partition Labels

```python
def partition_labels(s):
    last = {c: i for i, c in enumerate(s)}
    result = []
    start = end = 0
    for i, c in enumerate(s):
        end = max(end, last[c])
        if i == end:
            result.append(end - start + 1)
            start = i + 1
    return result
```

## Matchsticks to Square

```python
def makesquare(matchsticks):
    total = sum(matchsticks)
    if total % 4:
        return False
    side = total // 4
    matchsticks.sort(reverse=True)
    if matchsticks[0] > side:
        return False
    sides = [0] * 4
    
    def backtrack(i):
        if i == len(matchsticks):
            return all(s == side for s in sides)
        for j in range(4):
            if sides[j] + matchsticks[i] <= side:
                sides[j] += matchsticks[i]
                if backtrack(i + 1):
                    return True
                sides[j] -= matchsticks[i]
        return False
    
    return backtrack(0)
```

## Can I Win

```python
def can_i_win(max_choosable, desired_total):
    if desired_total <= 0:
        return True
    if max_choosable * (max_choosable + 1) // 2 < desired_total:
        return False
    memo = {}
    
    def dp(used, remaining):
        if remaining <= 0:
            return False
        if used in memo:
            return memo[used]
        for i in range(1, max_choosable + 1):
            mask = 1 << i
            if not (used & mask):
                if i >= remaining or not dp(used | mask, remaining - i):
                    memo[used] = True
                    return True
        memo[used] = False
        return False
    
    return dp(0, desired_total)
```

## Minimum Cost to Split Array

```python
def min_cost_to_split_array(nums, k):
    n = len(nums)
    prefix = [0]
    for x in nums:
        prefix.append(prefix[-1] + x)
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = float('inf')
        for j in range(i):
            cost = (prefix[i] - prefix[j]) % k
            if cost == 0:
                cost = k
            dp[i] = min(dp[i], dp[j] + cost)
    return dp[n]
```

## Splitting String into Max Unique Substrings

```python
def max_unique_split(s):
    n = len(s)
    memo = {}
    
    def dp(start):
        if start == n:
            return 0
        if start in memo:
            return memo[start]
        seen = set()
        best = 0
        for end in range(start + 1, n + 1):
            sub = s[start:end]
            if sub in seen:
                continue
            seen.add(sub)
            best = max(best, 1 + dp(end))
            seen.discard(sub)
        memo[start] = best
        return best
    
    return dp(0)
```
