# Bitmasking

## Represent Subset as Bitmask

Each element's presence is indicated by a bit. Bit i = 1 means element i is in subset.

```python
def subset_to_mask(subset: list[int], n: int) -> int:
    mask = 0
    for i in subset:
        mask |= 1 << i
    return mask

def mask_to_subset(mask: int, n: int) -> list[int]:
    return [i for i in range(n) if (mask >> i) & 1]
```

## Iterate All 2^n Subsets

```python
def iterate_all_subsets(n: int):
    for mask in range(1 << n):
        subset = [i for i in range(n) if (mask >> i) & 1]
        yield mask, subset
```

## Iterate All Subsets of Given Mask (Submask Enumeration s=(s-1)&mask)

```python
def iterate_subsets_of_mask(mask: int):
    s = mask
    while s:
        yield s
        s = (s - 1) & mask
    yield 0
```

## Check Element in Subset

```python
def in_subset(mask: int, i: int) -> bool:
    return bool((mask >> i) & 1)
```

## Add/Remove Element

```python
def add_element(mask: int, i: int) -> int:
    return mask | (1 << i)

def remove_element(mask: int, i: int) -> int:
    return mask & ~(1 << i)
```

## Union/Intersection/Difference

```python
def union_mask(a: int, b: int) -> int:
    return a | b

def intersection_mask(a: int, b: int) -> int:
    return a & b

def difference_mask(a: int, b: int) -> int:
    return a & ~b
```

## Count Elements (Popcount)

```python
def popcount(mask: int) -> int:
    return bin(mask).count('1')

def popcount_v2(mask: int) -> int:
    count = 0
    while mask:
        count += 1
        mask &= mask - 1
    return count
```

## Generate Power Set Using Bitmask

```python
def power_set(arr: list) -> list[list]:
    n = len(arr)
    result = []
    for mask in range(1 << n):
        subset = [arr[i] for i in range(n) if (mask >> i) & 1]
        result.append(subset)
    return result
```

## Subset Sum via Bitmask

```python
def subset_sum_bitmask(arr: list[int], target: int) -> bool:
    n = len(arr)
    for mask in range(1 << n):
        total = sum(arr[i] for i in range(n) if (mask >> i) & 1)
        if total == target:
            return True
    return False
```

## Maximum AND/OR/XOR of Pair

```python
def max_and_of_pair(arr: list[int]) -> int:
    result = 0
    for i in range(31, -1, -1):
        count = sum(1 for x in arr if (x >> i) & 1)
        if count >= 2:
            result |= 1 << i
            arr = [x for x in arr if (x >> i) & 1]
    return result

def max_or_of_pair(arr: list[int]) -> int:
    return max(a | b for i, a in enumerate(arr) for b in arr[i+1:]) if len(arr) >= 2 else 0

def max_xor_of_pair(arr: list[int]) -> int:
    root = {}
    for num in arr:
        node = root
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]
    max_xor = 0
    for num in arr:
        node = root
        curr = 0
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            toggled = 1 - bit
            if toggled in node:
                curr |= 1 << i
                node = node[toggled]
            else:
                node = node[bit]
        max_xor = max(max_xor, curr)
    return max_xor
```

## Find All Subsets with Given Sum

```python
def subsets_with_sum(arr: list[int], target: int) -> list[list[int]]:
    n = len(arr)
    result = []
    for mask in range(1 << n):
        total = sum(arr[i] for i in range(n) if (mask >> i) & 1)
        if total == target:
            result.append([arr[i] for i in range(n) if (mask >> i) & 1])
    return result
```

## DP + Bitmask Overview (TSP, Assignment)

```python
def tsp_bitmask(dist: list[list[int]], n: int) -> int:
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0
    for mask in range(1 << n):
        for last in range(n):
            if not (mask >> last) & 1:
                continue
            for next_city in range(n):
                if (mask >> next_city) & 1:
                    continue
                new_mask = mask | (1 << next_city)
                dp[new_mask][next_city] = min(
                    dp[new_mask][next_city],
                    dp[mask][last] + dist[last][next_city]
                )
    result = float('inf')
    for last in range(1, n):
        result = min(result, dp[(1 << n) - 1][last] + dist[last][0])
    return result
```

## Can I Win (Game with Bitmask)

```python
def can_i_win(max_choosable: int, desired_total: int) -> bool:
    if desired_total <= 0:
        return True
    if max_choosable * (max_choosable + 1) // 2 < desired_total:
        return False

    def dfs(mask: int, remaining: int, memo: dict) -> bool:
        if remaining <= 0:
            return False
        if mask in memo:
            return memo[mask]
        for i in range(1, max_choosable + 1):
            if (mask >> (i - 1)) & 1:
                continue
            if i >= remaining:
                memo[mask] = True
                return True
            if not dfs(mask | (1 << (i - 1)), remaining - i, memo):
                memo[mask] = True
                return True
        memo[mask] = False
        return False

    return dfs(0, desired_total, {})
```

## Partition into K Subsets

```python
def can_partition_k_subsets(nums: list[int], k: int) -> bool:
    total = sum(nums)
    if total % k:
        return False
    target = total // k
    n = len(nums)
    dp = [-1] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        if dp[mask] == -1:
            continue
        for i in range(n):
            if (mask >> i) & 1:
                continue
            new_mask = mask | (1 << i)
            new_sum = (dp[mask] + nums[i]) % target
            dp[new_mask] = new_sum

    return dp[(1 << n) - 1] == 0
```

## Number of Valid Words for Each Puzzle

```python
def find_num_of_valid_words(words: list[str], puzzles: list[str]) -> list[int]:
    def to_mask(s: str) -> int:
        mask = 0
        for c in s:
            mask |= 1 << (ord(c) - ord('a'))
        return mask

    count = {}
    for word in words:
        mask = to_mask(word)
        if bin(mask).count('1') <= 7:
            count[mask] = count.get(mask, 0) + 1

    result = []
    for puzzle in puzzles:
        first = 1 << (ord(puzzle[0]) - ord('a'))
        mask = to_mask(puzzle)
        total = 0
        submask = mask
        while submask:
            if submask & first:
                total += count.get(submask, 0)
            submask = (submask - 1) & mask
        result.append(total)
    return result
```
