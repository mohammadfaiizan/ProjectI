# Prefix and Suffix Arrays

Prefix sums allow O(1) range sum queries after O(n) preprocessing. Prefix[i] = sum of arr[0..i-1]. Range sum [l, r] = prefix[r+1] - prefix[l]. Suffix arrays store cumulative values from the right.

## Build Prefix Sum

```python
def build_prefix_sum(arr):
    prefix = [0]
    for x in arr:
        prefix.append(prefix[-1] + x)
    return prefix
```

## Range Sum Query Immutable

Precompute prefix. Query [left, right] = prefix[right+1] - prefix[left]. Time O(1) per query, O(n) preprocess.

```python
class NumArray:
    def __init__(self, nums):
        self.prefix = [0]
        for x in nums:
            self.prefix.append(self.prefix[-1] + x)

    def sumRange(self, left, right):
        return self.prefix[right + 1] - self.prefix[left]
```

## Range Sum 2D Immutable

Prefix sum for 2D: prefix[i][j] = sum of rectangle from (0,0) to (i-1,j-1). Query = inclusion-exclusion. Time O(1) per query, O(mn) preprocess.

```python
class NumMatrix:
    def __init__(self, matrix):
        if not matrix or not matrix[0]:
            self.prefix = [[0]]
            return
        m, n = len(matrix), len(matrix[0])
        self.prefix = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                self.prefix[i][j] = (matrix[i-1][j-1] + self.prefix[i-1][j] +
                                     self.prefix[i][j-1] - self.prefix[i-1][j-1])

    def sumRegion(self, row1, col1, row2, col2):
        return (self.prefix[row2+1][col2+1] - self.prefix[row1][col2+1] -
                self.prefix[row2+1][col1] + self.prefix[row1][col1])
```

## Prefix XOR Array

XOR prefix for range XOR queries. XOR is its own inverse. prefix[i] = arr[0] ^ arr[1] ^ ... ^ arr[i-1]. Range XOR [l,r] = prefix[r+1] ^ prefix[l].

```python
def build_prefix_xor(arr):
    prefix = [0]
    for x in arr:
        prefix.append(prefix[-1] ^ x)
    return prefix

def range_xor(prefix, left, right):
    return prefix[right + 1] ^ prefix[left]
```

## XOR Queries of Subarray

Given queries [l,r], return XOR of subarray. Build prefix XOR, each query O(1).

```python
def xor_queries(arr, queries):
    prefix = [0]
    for x in arr:
        prefix.append(prefix[-1] ^ x)
    return [prefix[r + 1] ^ prefix[l] for l, r in queries]
```

## Subarray Sum Equals k (Prefix + Hashmap)

prefix[j] - prefix[i] = k means prefix[i] = prefix[j] - k. For each j, count how many i < j have prefix[i] = prefix[j] - k. Time O(n), Space O(n).

```python
def subarray_sum_equals_k(arr, k):
    prefix_count = {0: 1}
    total = 0
    count = 0
    for x in arr:
        total += x
        count += prefix_count.get(total - k, 0)
        prefix_count[total] = prefix_count.get(total, 0) + 1
    return count
```

## Contiguous Array Trick

Binary array. Treat 0 as -1. Longest subarray with equal 0s and 1s = longest subarray with sum 0. Prefix + hashmap storing first occurrence. Time O(n), Space O(n).

```python
def contiguous_array(arr):
    prefix_idx = {0: -1}
    total = 0
    max_len = 0
    for i, x in enumerate(arr):
        total += 1 if x == 1 else -1
        if total in prefix_idx:
            max_len = max(max_len, i - prefix_idx[total])
        else:
            prefix_idx[total] = i
    return max_len
```

## Count Subarrays Divisible by k (Prefix Mod)

(prefix[j] - prefix[i]) % k == 0 means prefix[j] % k == prefix[i] % k. Handle negative: ((x % k) + k) % k. Count pairs with same mod. Time O(n), Space O(k).

```python
def count_subarrays_divisible_k(arr, k):
    mod_count = {0: 1}
    total = 0
    count = 0
    for x in arr:
        total = (total + x % k + k) % k
        count += mod_count.get(total, 0)
        mod_count[total] = mod_count.get(total, 0) + 1
    return count
```

## Product Except Self

Output[i] = product of all elements except arr[i]. Prefix product from left, suffix product from right. Or: one pass for left products, one pass for right. Time O(n), Space O(1) for output.

```python
def product_except_self(arr):
    n = len(arr)
    result = [1] * n
    left = 1
    for i in range(n):
        result[i] = left
        left *= arr[i]
    right = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right
        right *= arr[i]
    return result
```

## Equilibrium Index

Index where sum of left elements equals sum of right. total - prefix[i] - arr[i] == prefix[i]. So 2*prefix[i] + arr[i] == total. Or: left_sum, right_sum = total - left_sum - arr[i]. Time O(n), Space O(1).

```python
def equilibrium_index(arr):
    total = sum(arr)
    left_sum = 0
    for i in range(len(arr)):
        if left_sum == total - left_sum - arr[i]:
            return i
        left_sum += arr[i]
    return -1
```

## Max Subarray Using Prefix

Kadane can be seen as: for each j, find min prefix[i] for i < j. max_so_far = max(prefix[j] - min_prefix). Time O(n), Space O(1).

```python
def max_subarray_prefix(arr):
    min_prefix = 0
    total = 0
    max_sum = arr[0] if arr else 0
    for x in arr:
        total += x
        max_sum = max(max_sum, total - min_prefix)
        min_prefix = min(min_prefix, total)
    return max_sum
```

## Pivot Index

Same as equilibrium index. Index where sum of left equals sum of right. Time O(n), Space O(1).

```python
def pivot_index(arr):
    total = sum(arr)
    left_sum = 0
    for i in range(len(arr)):
        if left_sum == total - left_sum - arr[i]:
            return i
        left_sum += arr[i]
    return -1
```

## Running Sum

Cumulative sum: result[i] = arr[0] + arr[1] + ... + arr[i]. Time O(n), Space O(1) if in-place.

```python
def running_sum(arr):
    for i in range(1, len(arr)):
        arr[i] += arr[i-1]
    return arr
```

## Sum of Absolute Differences in Sorted Array

For sorted arr, for each i: sum(|arr[i]-arr[j]|) = arr[i]*left_count - left_sum + right_sum - arr[i]*right_count. Precompute prefix sums. Time O(n), Space O(n).

```python
def get_sum_absolute_differences(arr):
    n = len(arr)
    prefix = [0]
    for x in arr:
        prefix.append(prefix[-1] + x)
    result = []
    for i in range(n):
        left_sum = prefix[i]
        right_sum = prefix[n] - prefix[i + 1]
        left_count = i
        right_count = n - 1 - i
        val = arr[i] * left_count - left_sum + right_sum - arr[i] * right_count
        result.append(val)
    return result
```

## Minimum Penalty for Shop

Binary string, visit at each index. Penalty = count of 0s before visit + 1s after. Prefix 0s, suffix 1s. Find index minimizing penalty. Time O(n), Space O(n).

```python
def best_closing_time(customers):
    n = len(customers)
    zeros = [0]
    for c in customers:
        zeros.append(zeros[-1] + (1 if c == 'N' else 0))
    ones_after = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        ones_after[i] = ones_after[i + 1] + (1 if customers[i] == 'Y' else 0)
    min_penalty = float('inf')
    best = 0
    for i in range(n + 1):
        penalty = zeros[i] + ones_after[i]
        if penalty < min_penalty:
            min_penalty = penalty
            best = i
    return best
```

## Count Vowel Strings in Ranges

Prefix count of vowel strings. Query [l,r] = prefix[r+1] - prefix[l]. Time O(n + q), Space O(n).

```python
def vowel_strings(words, queries):
    vowels = set('aeiou')
    def is_vowel_word(w):
        return w[0] in vowels and w[-1] in vowels
    prefix = [0]
    for w in words:
        prefix.append(prefix[-1] + (1 if is_vowel_word(w) else 0))
    return [prefix[r + 1] - prefix[l] for l, r in queries]
```

## Ways to Split Into Three Parts

Binary array. Split into three non-empty parts with equal number of 1s. Count 1s, must be divisible by 3. For each valid split point, count ways. Use prefix sums of 1s. Time O(n), Space O(n).

```python
def num_ways_to_split(arr):
    total = sum(arr)
    if total % 3 != 0:
        return 0
    target = total // 3
    if target == 0:
        n = len(arr)
        return (n - 1) * (n - 2) // 2
    prefix = [0]
    for x in arr:
        prefix.append(prefix[-1] + x)
    count_first = 0
    count_second = 0
    for i in range(1, len(arr)):
        if prefix[i] == target:
            count_first += 1
        elif prefix[i] == 2 * target:
            count_second += count_first
    return count_second
```

## Make Sum Divisible by p

Remove smallest subarray such that remaining sum divisible by p. total % p = r. Need subarray with sum % p == r. Prefix mod + hashmap. Time O(n), Space O(p).

```python
def min_subarray_divisible(arr, p):
    r = sum(arr) % p
    if r == 0:
        return 0
    prefix_mod = {0: -1}
    total = 0
    min_len = len(arr) + 1
    for i, x in enumerate(arr):
        total = (total + x) % p
        target = (total - r + p) % p
        if target in prefix_mod:
            min_len = min(min_len, i - prefix_mod[target])
        prefix_mod[total] = i
    return min_len if min_len <= len(arr) else -1
```
