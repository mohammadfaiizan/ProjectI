# Subsequence DP

## LIS O(n^2) DP

```python
def length_of_lis_n2(nums):
    if not nums:
        return 0
    n = len(nums)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
```

## LIS O(n log n) Binary Search

```python
def length_of_lis_nlogn(nums):
    if not nums:
        return 0
    tails = []
    for x in nums:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(x)
        else:
            tails[lo] = x
    return len(tails)
```

## Longest Non-Decreasing

```python
def longest_nondecreasing(nums):
    if not nums:
        return 0
    tails = []
    for x in nums:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(x)
        else:
            tails[lo] = x
    return len(tails)
```

## Longest Decreasing

```python
def longest_decreasing(nums):
    return longest_nondecreasing([-x for x in nums])
```

## Number of LIS

```python
def find_number_of_lis(nums):
    n = len(nums)
    if n == 0:
        return 0
    lengths = [1] * n
    counts = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                if lengths[j] + 1 > lengths[i]:
                    lengths[i] = lengths[j] + 1
                    counts[i] = counts[j]
                elif lengths[j] + 1 == lengths[i]:
                    counts[i] += counts[j]
    max_len = max(lengths)
    return sum(c for l, c in zip(lengths, counts) if l == max_len)
```

## Longest Bitonic

```python
def longest_bitonic(nums):
    n = len(nums)
    inc = [1] * n
    dec = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                inc[i] = max(inc[i], inc[j] + 1)
    for i in range(n - 2, -1, -1):
        for j in range(n - 1, i, -1):
            if nums[j] < nums[i]:
                dec[i] = max(dec[i], dec[j] + 1)
    return max(inc[i] + dec[i] - 1 for i in range(n))
```

## Longest Alternating

```python
def wiggle_max_length(nums):
    if len(nums) < 2:
        return len(nums)
    up, down = 1, 1
    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            up = down + 1
        elif nums[i] < nums[i - 1]:
            down = up + 1
    return max(up, down)
```

## Max Sum Increasing Subsequence

```python
def max_sum_increasing_subsequence(nums):
    if not nums:
        return 0
    n = len(nums)
    dp = nums[:]
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + nums[i])
    return max(dp)
```

## Longest Chain of Pairs

```python
def find_longest_chain(pairs):
    pairs.sort(key=lambda x: x[1])
    dp = [1] * len(pairs)
    for i in range(1, len(pairs)):
        for j in range(i):
            if pairs[j][1] < pairs[i][0]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
```

## Russian Doll Envelopes (2D LIS)

```python
def max_envelopes(envelopes):
    envelopes.sort(key=lambda x: (x[0], -x[1]))
    heights = [e[1] for e in envelopes]
    tails = []
    for h in heights:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < h:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(h)
        else:
            tails[lo] = h
    return len(tails)
```

## Max Length Pair Chain

```python
def find_longest_chain_greedy(pairs):
    pairs.sort(key=lambda x: x[1])
    count, end = 0, float('-inf')
    for a, b in pairs:
        if a > end:
            count += 1
            end = b
    return count
```

## Wiggle Subsequence

```python
def wiggle_max_length(nums):
    if len(nums) < 2:
        return len(nums)
    up, down = 1, 1
    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            up = down + 1
        elif nums[i] < nums[i - 1]:
            down = up + 1
    return max(up, down)
```

## Arithmetic Slices II

```python
def numberOfArithmeticSlices(nums):
    n = len(nums)
    dp = [{} for _ in range(n)]
    total = 0
    for i in range(n):
        for j in range(i):
            diff = nums[i] - nums[j]
            count = dp[j].get(diff, 0)
            dp[i][diff] = dp[i].get(diff, 0) + count + 1
            total += count
    return total
```

## Longest Arithmetic Subsequence

```python
def longest_arith_seq_length(nums):
    n = len(nums)
    dp = [{} for _ in range(n)]
    max_len = 2
    for i in range(n):
        for j in range(i):
            diff = nums[i] - nums[j]
            dp[i][diff] = dp[j].get(diff, 1) + 1
            max_len = max(max_len, dp[i][diff])
    return max_len
```

## Longest Arithmetic of Given Difference

```python
def longest_subsequence(arr, difference):
    dp = {}
    for x in arr:
        dp[x] = dp.get(x - difference, 0) + 1
    return max(dp.values())
```

## Largest Divisible Subset

```python
def largest_divisible_subset(nums):
    if not nums:
        return []
    nums.sort()
    n = len(nums)
    dp = [1] * n
    parent = [-1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[i] % nums[j] == 0 and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    idx = max(range(n), key=lambda i: dp[i])
    result = []
    while idx >= 0:
        result.append(nums[idx])
        idx = parent[idx]
    return result[::-1]
```
