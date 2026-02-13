# Combinatorial Problems

## Generate All Subsets (Power Set)

**Theory**: For each element, either include or exclude. 2^n subsets.

**Iterative**:
```python
def subsets_iterative(nums):
    result = [[]]
    for num in nums:
        result += [curr + [num] for curr in result]
    return result
```

**Recursive**:
```python
def subsets_recursive(nums):
    result = []

    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result
```

**Bitmask**:
```python
def subsets_bitmask(nums):
    n = len(nums)
    result = []
    for mask in range(1 << n):
        subset = [nums[i] for i in range(n) if (mask >> i) & 1]
        result.append(subset)
    return result
```

## Subsets with Duplicates (Subsets II)

**Theory**: Sort first. Skip duplicate elements at same level (when not first at that level).

```python
def subsets_with_dup(nums):
    nums.sort()
    result = []

    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result
```

## Generate All Permutations

**Theory**: Fix one position at a time. Swap or use used array.

```python
def permute(nums):
    result = []

    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False

    backtrack([], [False] * len(nums))
    return result
```

## Permutations with Duplicates (Permutations II)

**Theory**: Use frequency map. For each unique element, use one instance and recurse.

```python
def permute_unique(nums):
    from collections import Counter
    count = Counter(nums)
    result = []

    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for c in count:
            if count[c] > 0:
                count[c] -= 1
                path.append(c)
                backtrack(path)
                path.pop()
                count[c] += 1

    backtrack([])
    return result
```

## Combinations (nCr)

**Theory**: Choose k from n. Backtrack with start index to avoid duplicates.

```python
def combine(n, k):
    result = []

    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()

    backtrack(1, [])
    return result
```

## Combination Sum I (Unlimited Use)

**Theory**: Same element can be used multiple times. Recurse with same index.

```python
def combination_sum(candidates, target):
    result = []

    def backtrack(start, path, remain):
        if remain == 0:
            result.append(path[:])
            return
        if remain < 0:
            return
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            backtrack(i, path, remain - candidates[i])
            path.pop()

    backtrack(0, [], target)
    return result
```

## Combination Sum II (Each Once)

**Theory**: Each candidate used once. Sort and skip duplicates at same level.

```python
def combination_sum2(candidates, target):
    candidates.sort()
    result = []

    def backtrack(start, path, remain):
        if remain == 0:
            result.append(path[:])
            return
        if remain < 0:
            return
        for i in range(start, len(candidates)):
            if i > start and candidates[i] == candidates[i - 1]:
                continue
            path.append(candidates[i])
            backtrack(i + 1, path, remain - candidates[i])
            path.pop()

    backtrack(0, [], target)
    return result
```

## Combination Sum III (K Numbers 1-9)

**Theory**: Use exactly k numbers from 1-9, each at most once, sum to n.

```python
def combination_sum3(k, n):
    result = []

    def backtrack(start, path, remain):
        if len(path) == k and remain == 0:
            result.append(path[:])
            return
        if len(path) >= k or remain <= 0:
            return
        for i in range(start, 10):
            path.append(i)
            backtrack(i + 1, path, remain - i)
            path.pop()

    backtrack(1, [], n)
    return result
```

## Combination Sum IV (Count Ways)

**Theory**: Count permutations that sum to target. Note: This is DP, not backtracking. Backtracking would enumerate all (exponential).

```python
def combination_sum4(nums, target):
    dp = [0] * (target + 1)
    dp[0] = 1
    for i in range(1, target + 1):
        for num in nums:
            if i >= num:
                dp[i] += dp[i - num]
    return dp[target]
```

## Letter Combinations of Phone Number

**Theory**: Map digits to letters. Build string one digit at a time.

```python
def letter_combinations(digits):
    if not digits:
        return []
    mapping = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }
    result = []

    def backtrack(index, path):
        if index == len(digits):
            result.append(path)
            return
        for c in mapping[digits[index]]:
            backtrack(index + 1, path + c)

    backtrack(0, '')
    return result
```

## Generate Parentheses (N Pairs)

**Theory**: Add '(' when open < n. Add ')' when close < open. Valid when open == close == n.

```python
def generate_parenthesis(n):
    result = []

    def backtrack(open_count, close_count, path):
        if len(path) == 2 * n:
            result.append(path)
            return
        if open_count < n:
            backtrack(open_count + 1, close_count, path + '(')
        if close_count < open_count:
            backtrack(open_count, close_count + 1, path + ')')

    backtrack(0, 0, '')
    return result
```

## Palindrome Partitioning (All Valid)

**Theory**: At each position, try cutting. If prefix is palindrome, recurse on rest.

```python
def partition_palindrome(s):
    result = []

    def is_palindrome(s):
        return s == s[::-1]

    def backtrack(start, path):
        if start == len(s):
            result.append(path[:])
            return
        for i in range(start + 1, len(s) + 1):
            prefix = s[start:i]
            if is_palindrome(prefix):
                path.append(prefix)
                backtrack(i, path)
                path.pop()

    backtrack(0, [])
    return result
```

## Restore IP Addresses

**Theory**: Place 3 dots. Each segment 0-255, no leading zeros except "0".

```python
def restore_ip_addresses(s):
    result = []

    def valid(segment):
        if len(segment) > 1 and segment[0] == '0':
            return False
        return 0 <= int(segment) <= 255

    def backtrack(start, path):
        if len(path) == 4 and start == len(s):
            result.append('.'.join(path))
            return
        if len(path) >= 4:
            return
        for i in range(1, 4):
            if start + i <= len(s):
                segment = s[start:start + i]
                if valid(segment):
                    path.append(segment)
                    backtrack(start + i, path)
                    path.pop()

    backtrack(0, [])
    return result
```

## Partition to K Equal Sum Subsets

**Theory**: Backtrack to fill k buckets. Each bucket gets sum/k. Prune when remaining cannot fill current bucket.

```python
def can_partition_k_subsets(nums, k):
    total = sum(nums)
    if total % k != 0:
        return False
    target = total // k
    nums.sort(reverse=True)
    if nums[0] > target:
        return False
    buckets = [0] * k

    def backtrack(index):
        if index == len(nums):
            return True
        for i in range(k):
            if buckets[i] + nums[index] <= target:
                buckets[i] += nums[index]
                if backtrack(index + 1):
                    return True
                buckets[i] -= nums[index]
            if buckets[i] == 0:
                break
        return False

    return backtrack(0)
```

## Fair Distribution of Cookies

**Theory**: Distribute n cookies to k children. Minimize max cookies any child gets. Backtrack: assign each cookie to a child.

```python
def distribute_cookies(cookies, k):
    children = [0] * k
    result = [float('inf')]

    def backtrack(index):
        if index == len(cookies):
            result[0] = min(result[0], max(children))
            return
        for i in range(k):
            children[i] += cookies[index]
            if max(children) < result[0]:
                backtrack(index + 1)
            children[i] -= cookies[index]

    backtrack(0)
    return result[0]
```

## Matchsticks to Square

**Theory**: Use all matchsticks to form square. Four sides each of length sum/4. Backtrack: assign each stick to a side.

```python
def makesquare(matchsticks):
    total = sum(matchsticks)
    if total % 4 != 0:
        return False
    side = total // 4
    matchsticks.sort(reverse=True)
    sides = [0] * 4

    def backtrack(index):
        if index == len(matchsticks):
            return True
        for i in range(4):
            if sides[i] + matchsticks[index] <= side:
                sides[i] += matchsticks[index]
                if backtrack(index + 1):
                    return True
                sides[i] -= matchsticks[index]
            if sides[i] == 0:
                break
        return False

    return backtrack(0)
```

## Splitting String into Descending Consecutive Values

**Theory**: Split string so each part is one less than previous. First part can be any valid number.

```python
def split_string(s):
    def backtrack(start, prev):
        if start == len(s):
            return True
        for i in range(start + 1, len(s) + 1):
            curr = int(s[start:i])
            if prev is None:
                if backtrack(i, curr):
                    return True
            elif curr == prev - 1:
                if backtrack(i, curr):
                    return True
            elif curr >= prev:
                break
        return False

    return backtrack(0, None)
```

## Beautiful Arrangement

**Theory**: Permutation where position i divides arr[i] or arr[i] divides i. Count such permutations.

```python
def count_arrangement(n):
    used = [False] * (n + 1)
    count = [0]

    def backtrack(pos):
        if pos > n:
            count[0] += 1
            return
        for i in range(1, n + 1):
            if not used[i] and (pos % i == 0 or i % pos == 0):
                used[i] = True
                backtrack(pos + 1)
                used[i] = False

    backtrack(1)
    return count[0]
```

## Count Numbers with Unique Digits

**Theory**: Count numbers in [0, 10^n) with all unique digits. Use backtracking or combinatorial counting.

```python
def count_numbers_with_unique_digits(n):
    if n == 0:
        return 1
    total = 10
    for i in range(2, n + 1):
        choices = 9
        for j in range(1, i):
            choices *= (10 - j)
        total += choices
    return total
```

## All Paths from Source to Target

**Theory**: DAG. Find all paths from node 0 to node n-1. DFS/backtrack.

```python
def all_paths_source_target(graph):
    n = len(graph)
    result = []

    def backtrack(node, path):
        if node == n - 1:
            result.append(path[:])
            return
        for neighbor in graph[node]:
            path.append(neighbor)
            backtrack(neighbor, path)
            path.pop()

    backtrack(0, [0])
    return result
```
