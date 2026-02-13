# Subarray Problems

A subarray is a contiguous slice of an array. Subarray from index i to j (inclusive) has elements arr[i], arr[i+1], ..., arr[j].

## Max Subarray Sum (Kadane's Algorithm)

Track max ending here and max so far. If current element alone is better than extending, start new subarray. Time O(n), Space O(1).

```python
def max_subarray_sum_kadane(arr):
    if not arr:
        return 0
    max_ending = max_so_far = arr[0]
    for i in range(1, len(arr)):
        max_ending = max(arr[i], max_ending + arr[i])
        max_so_far = max(max_so_far, max_ending)
    return max_so_far
```

## Max Subarray with At Least k Elements

Compute prefix sums. For each ending index j, we need best starting index i where j - i + 1 >= k. Use sliding window of min prefix in valid range.

```python
def max_subarray_at_least_k(arr, k):
    n = len(arr)
    if n < k:
        return 0
    prefix = [0]
    for x in arr:
        prefix.append(prefix[-1] + x)
    max_sum = float('-inf')
    min_prefix = 0
    for i in range(k, n + 1):
        max_sum = max(max_sum, prefix[i] - min_prefix)
        min_prefix = min(min_prefix, prefix[i - k + 1])
    return max_sum
```

## Min Subarray Sum

Similar to Kadane but minimize. Or negate and use Kadane for max.

```python
def min_subarray_sum(arr):
    if not arr:
        return 0
    min_ending = min_so_far = arr[0]
    for i in range(1, len(arr)):
        min_ending = min(arr[i], min_ending + arr[i])
        min_so_far = min(min_so_far, min_ending)
    return min_so_far
```

## Max Product Subarray

Track both max and min product (negative * negative = positive). Time O(n), Space O(1).

```python
def max_product_subarray(arr):
    if not arr:
        return 0
    max_prod = min_prod = result = arr[0]
    for i in range(1, len(arr)):
        if arr[i] < 0:
            max_prod, min_prod = min_prod, max_prod
        max_prod = max(arr[i], max_prod * arr[i])
        min_prod = min(arr[i], min_prod * arr[i])
        result = max(result, max_prod)
    return result
```

## Max Sum Circular Subarray

Either max subarray is within array (Kadane) or wraps around (total - min subarray). Take max of both. Handle all negative.

```python
def max_sum_circular(arr):
    if not arr:
        return 0
    total = sum(arr)
    max_kadane = max_subarray_sum_kadane(arr)
    neg_arr = [-x for x in arr]
    max_wrap = total + max_subarray_sum_kadane(neg_arr)
    if max_wrap == 0:
        return max_kadane
    return max(max_kadane, max_wrap)
```

## Max Sum Two Non-Overlapping Subarrays

For each split point, compute max subarray sum in left and right. Use prefix of max subarray sums from left and right.

```python
def max_sum_two_non_overlapping(arr, first_len, second_len):
    n = len(arr)
    prefix = [0]
    for x in arr:
        prefix.append(prefix[-1] + x)

    def max_sum_subarray_len(length):
        res = [0] * (n + 1)
        for i in range(length, n + 1):
            res[i] = max(res[i - 1], prefix[i] - prefix[i - length])
        return res

    left_first = max_sum_subarray_len(first_len)
    right_second = [0] * (n + 2)
    for i in range(n - second_len, -1, -1):
        s = prefix[i + second_len] - prefix[i]
        right_second[i] = max(right_second[i + 1], s)

    result = 0
    for i in range(first_len, n - second_len + 1):
        result = max(result, left_first[i] + right_second[i])
    return result
```

## Max Sum Three Non-Overlapping Subarrays

DP: for each possible middle subarray position, combine with best left and right. Time O(n), Space O(n).

```python
def max_sum_three_non_overlapping(arr, k):
    n = len(arr)
    if n < 3 * k:
        return []
    prefix = [0]
    for x in arr:
        prefix.append(prefix[-1] + x)
    windows = [prefix[i + k] - prefix[i] for i in range(n - k + 1)]
    left = [0] * len(windows)
    best = 0
    for i in range(len(windows)):
        if windows[i] > windows[best]:
            best = i
        left[i] = best
    right = [0] * len(windows)
    best = len(windows) - 1
    for i in range(len(windows) - 1, -1, -1):
        if windows[i] >= windows[best]:
            best = i
        right[i] = best
    max_sum = 0
    result = []
    for mid in range(k, len(windows) - k):
        l, r = left[mid - k], right[mid + k]
        total = windows[l] + windows[mid] + windows[r]
        if total > max_sum:
            max_sum = total
            result = [l, mid, r]
    return result
```

## Subarray Sum Equals k (Count)

Prefix sum + hashmap. Count prefix sums that differ by k. Time O(n), Space O(n).

```python
def subarray_sum_count(arr, k):
    prefix_count = {0: 1}
    total = 0
    count = 0
    for x in arr:
        total += x
        count += prefix_count.get(total - k, 0)
        prefix_count[total] = prefix_count.get(total, 0) + 1
    return count
```

## Subarray Sum Equals k (Find)

Return indices of first/last such subarray. Track first occurrence of prefix in hashmap.

```python
def subarray_sum_find(arr, k):
    prefix_idx = {0: -1}
    total = 0
    for i, x in enumerate(arr):
        total += x
        if total - k in prefix_idx:
            return [prefix_idx[total - k] + 1, i]
        prefix_idx[total] = i
    return None
```

## Subarray with 0 Sum

Check if any prefix sum repeats (or equals 0). Time O(n), Space O(n).

```python
def subarray_with_zero_sum(arr):
    seen = {0}
    total = 0
    for x in arr:
        total += x
        if total in seen:
            return True
        seen.add(total)
    return False
```

## Longest Subarray with 0 Sum

Track first occurrence of each prefix sum. Max length = max(j - first_occurrence[prefix]). Time O(n), Space O(n).

```python
def longest_subarray_zero_sum(arr):
    prefix_idx = {0: -1}
    total = 0
    max_len = 0
    for i, x in enumerate(arr):
        total += x
        if total in prefix_idx:
            max_len = max(max_len, i - prefix_idx[total])
        else:
            prefix_idx[total] = i
    return max_len
```

## Longest Subarray with Sum k

For positive integers: sliding window. For general: prefix + hashmap (store first occurrence of prefix).

```python
def longest_subarray_sum_k_positive(arr, k):
    left = total = 0
    max_len = 0
    for right in range(len(arr)):
        total += arr[right]
        while total > k and left <= right:
            total -= arr[left]
            left += 1
        if total == k:
            max_len = max(max_len, right - left + 1)
    return max_len

def longest_subarray_sum_k_general(arr, k):
    prefix_idx = {0: -1}
    total = 0
    max_len = 0
    for i, x in enumerate(arr):
        total += x
        if total - k in prefix_idx:
            max_len = max(max_len, i - prefix_idx[total - k])
        if total not in prefix_idx:
            prefix_idx[total] = i
    return max_len
```

## Longest with Sum At Most k

Sliding window: expand right, shrink left when sum > k. Time O(n), Space O(1).

```python
def longest_subarray_sum_at_most_k(arr, k):
    left = total = 0
    max_len = 0
    for right in range(len(arr)):
        total += arr[right]
        while total > k and left <= right:
            total -= arr[left]
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len
```

## Shortest with Sum At Least k

Monotonic deque of prefix sums. For each j, find largest i with prefix[j] - prefix[i] >= k. Time O(n), Space O(n).

```python
def shortest_subarray_sum_at_least_k(arr, k):
    n = len(arr)
    prefix = [0]
    for x in arr:
        prefix.append(prefix[-1] + x)
    from collections import deque
    dq = deque()
    result = n + 1
    for j in range(n + 1):
        while dq and prefix[j] - prefix[dq[0]] >= k:
            result = min(result, j - dq.popleft())
        while dq and prefix[dq[-1]] >= prefix[j]:
            dq.pop()
        dq.append(j)
    return result if result <= n else -1
```

## Smallest with Sum Greater Than Value

Sliding window for positive array. Time O(n), Space O(1).

```python
def smallest_subarray_sum_greater(arr, target):
    left = total = 0
    min_len = float('inf')
    for right in range(len(arr)):
        total += arr[right]
        while total > target and left <= right:
            min_len = min(min_len, right - left + 1)
            total -= arr[left]
            left += 1
    return min_len if min_len != float('inf') else 0
```

## Count Subarrays Sum Divisible by k

Prefix mod k. If (prefix[j] - prefix[i]) % k == 0 then prefix[j] % k == prefix[i] % k. Count pairs with same mod. Time O(n), Space O(k).

```python
def count_subarrays_divisible_by_k(arr, k):
    mod_count = {0: 1}
    total = 0
    count = 0
    for x in arr:
        total = (total + x % k + k) % k
        count += mod_count.get(total, 0)
        mod_count[total] = mod_count.get(total, 0) + 1
    return count
```

## Count with Product Less Than k

Sliding window: for each right, count subarrays ending at right with product < k. Shrink left when product >= k. Time O(n), Space O(1).

```python
def count_subarrays_product_less_k(arr, k):
    if k <= 1:
        return 0
    left = count = 0
    prod = 1
    for right in range(len(arr)):
        prod *= arr[right]
        while prod >= k and left <= right:
            prod //= arr[left]
            left += 1
        count += right - left + 1
    return count
```

## Count with Bounded Maximum

Count subarrays where max element is in [left, right]. Use contribution of each element as max. Monotonic stack. Time O(n), Space O(n).

```python
def count_subarrays_bounded_max(arr, left_val, right_val):
    def count_less_equal(bound):
        count = 0
        curr = 0
        for x in arr:
            if x <= bound:
                curr += 1
                count += curr
            else:
                curr = 0
        return count
    return count_less_equal(right_val) - count_less_equal(left_val - 1)
```

## Subarrays with All Same Elements

Count contiguous groups of same element. For group of length L, it contributes L*(L+1)//2 subarrays.

```python
def count_subarrays_all_same(arr):
    if not arr:
        return 0
    count = 0
    i = 0
    while i < len(arr):
        j = i
        while j < len(arr) and arr[j] == arr[i]:
            j += 1
        n = j - i
        count += n * (n + 1) // 2
        i = j
    return count
```

## Count with Exactly k Distinct

Sliding window: count subarrays with at most k distinct minus at most (k-1) distinct. Time O(n), Space O(n).

```python
def count_at_most_k_distinct(arr, k):
    from collections import defaultdict
    if k == 0:
        return 0
    left = count = 0
    freq = defaultdict(int)
    distinct = 0
    for right in range(len(arr)):
        if freq[arr[right]] == 0:
            distinct += 1
        freq[arr[right]] += 1
        while distinct > k:
            freq[arr[left]] -= 1
            if freq[arr[left]] == 0:
                distinct -= 1
            left += 1
        count += right - left + 1
    return count

def count_exactly_k_distinct(arr, k):
    return count_at_most_k_distinct(arr, k) - count_at_most_k_distinct(arr, k - 1)
```

## Count with At Most k Distinct

See implementation above in count_at_most_k_distinct.

## Count with Odd Sum

Subarray sum is odd if (prefix[j] - prefix[i]) is odd, i.e. prefix[j] and prefix[i] have different parity. Count even and odd prefix sums. Time O(n), Space O(1).

```python
def count_subarrays_odd_sum(arr):
    even = 1
    odd = 0
    total = 0
    count = 0
    for x in arr:
        total += x
        if total % 2 == 0:
            count += odd
            even += 1
        else:
            count += even
            odd += 1
    return count
```

## Max Length Equal 0s and 1s

Treat 0 as -1. Longest subarray with sum 0. Prefix + hashmap. Time O(n), Space O(n).

```python
def max_length_equal_01(arr):
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

## Contiguous Array

Same as max length equal 0s and 1s. Binary array, find longest contiguous subarray with equal 0s and 1s.

```python
def contiguous_array(arr):
    return max_length_equal_01(arr)
```

## Min Ops to Make Equal in Subarray

For subarray to have all same elements with min ops: change all to median (for odd length) or either median value. Count ops = sum of distances from chosen value.

```python
def min_ops_equal_subarray(arr, left, right):
    sub = sorted(arr[left:right + 1])
    n = len(sub)
    median = sub[n // 2]
    return sum(abs(x - median) for x in sub)
```

## Max Sum After One Deletion

For each position, max subarray sum that deletes one element = max(left_max_ending[i-1], right_max_ending[i+1]) or kadane without that element. Precompute left and right max ending. Time O(n), Space O(n).

```python
def max_sum_after_one_deletion(arr):
    n = len(arr)
    if n <= 1:
        return sum(arr)
    left = [arr[0]] * n
    for i in range(1, n):
        left[i] = max(arr[i], left[i - 1] + arr[i])
    right = [arr[-1]] * n
    for i in range(n - 2, -1, -1):
        right[i] = max(arr[i], right[i + 1] + arr[i])
    result = max(left)
    for i in range(1, n - 1):
        result = max(result, left[i - 1] + right[i + 1])
    return result
```

## K-Concatenation Max Sum

Concatenate array k times. Max subarray either in single copy (Kadane) or spans concatenation boundary. If total sum > 0, add (k-2)*total + max prefix + max suffix. Time O(n), Space O(1).

```python
def k_concat_max_sum(arr, k):
    mod = 10**9 + 7
    def kadane(a):
        max_ending = max_so_far = a[0]
        for x in a[1:]:
            max_ending = max(x, max_ending + x)
            max_so_far = max(max_so_far, max_ending)
        return max_so_far
    total = sum(arr)
    if k == 1:
        return max(0, kadane(arr)) % mod
    kadane_one = kadane(arr)
    max_prefix = 0
    s = 0
    for x in arr:
        s += x
        max_prefix = max(max_prefix, s)
    max_suffix = 0
    s = 0
    for x in reversed(arr):
        s += x
        max_suffix = max(max_suffix, s)
    return max(0, kadane_one, total * k, max_prefix + total * (k - 2) + max_suffix) % mod
```

## Subarray Ranges

Sum of (max - min) over all subarrays. For each element, count subarrays where it is max and where it is min. Use monotonic stack. Time O(n), Space O(n).

```python
def subarray_ranges(arr):
    n = len(arr)
    def sum_subarray_maxs(a):
        result = 0
        stack = []
        for i in range(n + 1):
            while stack and (i == n or a[stack[-1]] <= a[i]):
                j = stack.pop()
                left = stack[-1] if stack else -1
                result += a[j] * (j - left) * (i - j)
            stack.append(i)
        return result
    def sum_subarray_mins(a):
        result = 0
        stack = []
        for i in range(n + 1):
            while stack and (i == n or a[stack[-1]] >= a[i]):
                j = stack.pop()
                left = stack[-1] if stack else -1
                result += a[j] * (j - left) * (i - j)
            stack.append(i)
        return result
    return sum_subarray_maxs(arr) - sum_subarray_mins(arr)
```

## Sum of Subarray Minimums

Each element contributes as min in some subarrays. Monotonic stack to find left and right boundaries. Time O(n), Space O(n).

```python
def sum_subarray_minimums(arr):
    n = len(arr)
    mod = 10**9 + 7
    left = [-1] * n
    stack = []
    for i in range(n):
        while stack and arr[stack[-1]] >= arr[i]:
            stack.pop()
        left[i] = stack[-1] if stack else -1
        stack.append(i)
    right = [n] * n
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and arr[stack[-1]] > arr[i]:
            stack.pop()
        right[i] = stack[-1] if stack else n
        stack.append(i)
    result = 0
    for i in range(n):
        result = (result + arr[i] * (i - left[i]) * (right[i] - i)) % mod
    return result
```

## Sum of Subarray Maximums

Same as minimums but use >= for right boundary to avoid double counting. Time O(n), Space O(n).

```python
def sum_subarray_maximums(arr):
    n = len(arr)
    mod = 10**9 + 7
    left = [-1] * n
    stack = []
    for i in range(n):
        while stack and arr[stack[-1]] <= arr[i]:
            stack.pop()
        left[i] = stack[-1] if stack else -1
        stack.append(i)
    right = [n] * n
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and arr[stack[-1]] < arr[i]:
            stack.pop()
        right[i] = stack[-1] if stack else n
        stack.append(i)
    result = 0
    for i in range(n):
        result = (result + arr[i] * (i - left[i]) * (right[i] - i)) % mod
    return result
```
