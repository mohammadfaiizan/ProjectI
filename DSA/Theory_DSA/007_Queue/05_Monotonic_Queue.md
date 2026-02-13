# Monotonic Queue

## Theory

A monotonic queue (or deque) maintains elements in strictly increasing or strictly decreasing order. It is used to efficiently find the maximum or minimum in a sliding window, or to solve problems involving "next greater/smaller" in a range.

**Decreasing monotonic deque (for sliding window maximum)**: Store indices. Before adding index i, remove from the back all indices j where arr[j] < arr[i], because those can never be the maximum for any future window containing i. The front always holds the index of the current window maximum.

**Increasing monotonic deque (for sliding window minimum)**: Same idea but remove indices j where arr[j] > arr[i].

**Key operations**:
- Push: Remove from back while the new element violates monotonicity
- Pop front: Remove when index falls outside the window
- Front: Current extremum (max or min)

## Sliding Window Maximum (Deque Maintaining Decreasing Order)

For each window of size k, return the maximum. Maintain a deque of indices in decreasing order of values.

```python
from collections import deque

def max_sliding_window(nums, k):
    dq = deque()
    result = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] < x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result
```

## Sliding Window Minimum

Maintain increasing monotonic deque; front is the minimum.

```python
def min_sliding_window(nums, k):
    dq = deque()
    result = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] > x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result
```

## Shortest Subarray with Sum at Least K (Deque + Prefix Sum)

Array with possible negatives. Find shortest subarray with sum >= k. Use prefix sum; for each prefix[i], we want smallest j < i with prefix[i] - prefix[j] >= k, i.e. prefix[j] <= prefix[i] - k. Maintain increasing deque of prefix indices; for each i, pop from front while prefix[front] <= prefix[i] - k (these are valid and we want the smallest j), then pop from back while prefix[back] >= prefix[i] (prefix[i] is better for future).

```python
from collections import deque

def shortest_subarray(nums, k):
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]
    dq = deque([0])
    result = float('inf')
    for i in range(1, n + 1):
        while dq and prefix[i] - prefix[dq[0]] >= k:
            result = min(result, i - dq.popleft())
        while dq and prefix[dq[-1]] >= prefix[i]:
            dq.pop()
        dq.append(i)
    return result if result != float('inf') else -1
```

## Constrained Subsequence Sum

Choose subsequence (not necessarily contiguous) with no two chosen elements adjacent. Each choice has cost. Maximize sum. At each index i, we can take nums[i] + max of valid previous. Use deque to maintain max over the last k indices (k=2 for "no adjacent").

For the general problem with constraint "at most k apart": dp[i] = nums[i] + max(dp[i-k] ... dp[i-1]) for i > 0. Use monotonic deque to get max over sliding window of dp values.

```python
from collections import deque

def constrained_subset_sum(nums, k):
    dq = deque()
    result = float('-inf')
    for i, x in enumerate(nums):
        while dq and dq[0][1] < i - k:
            dq.popleft()
        prev_max = dq[0][0] if dq else 0
        curr = max(x, x + prev_max)
        result = max(result, curr)
        while dq and dq[-1][0] <= curr:
            dq.pop()
        dq.append((curr, i))
    return result
```

## Jump Game VI

From index 0, jump at most k steps. Maximize sum of landed indices. dp[i] = nums[i] + max(dp[i-k] ... dp[i-1]). Monotonic deque for max over sliding window.

```python
from collections import deque

def max_result(nums, k):
    n = len(nums)
    dq = deque([(nums[0], 0)])
    for i in range(1, n):
        while dq and dq[0][1] < i - k:
            dq.popleft()
        curr = nums[i] + dq[0][0]
        while dq and dq[-1][0] <= curr:
            dq.pop()
        dq.append((curr, i))
    return dq[-1][0] if dq else nums[0]
```

## Longest Continuous Subarray with Absolute Diff <= Limit

Find longest subarray where max - min <= limit. Use two deques: one decreasing (for max), one increasing (for min). Expand right; when max - min > limit, shrink left.

```python
from collections import deque

def longest_subarray(nums, limit):
    max_dq = deque()
    min_dq = deque()
    left = 0
    result = 0
    for right, x in enumerate(nums):
        while max_dq and nums[max_dq[-1]] < x:
            max_dq.pop()
        max_dq.append(right)
        while min_dq and nums[min_dq[-1]] > x:
            min_dq.pop()
        min_dq.append(right)
        while nums[max_dq[0]] - nums[min_dq[0]] > limit:
            if max_dq[0] == left:
                max_dq.popleft()
            if min_dq[0] == left:
                min_dq.popleft()
            left += 1
        result = max(result, right - left + 1)
    return result
```

## Max Value of Equation

Given points (x_i, y_i) sorted by x. Find max of y_i + y_j + |x_i - x_j| for pairs with |x_i - x_j| <= k. Rewrite: y_i + y_j + x_j - x_i = (y_i - x_i) + (y_j + x_j). For each j, we want max over i in [j-k, j) of (y_i - x_i). Monotonic deque.

```python
from collections import deque

def find_max_value_of_equation(points, k):
    dq = deque()
    result = float('-inf')
    for x, y in points:
        while dq and x - dq[0][1] > k:
            dq.popleft()
        if dq:
            result = max(result, dq[0][0] + x + y)
        val = y - x
        while dq and dq[-1][0] <= val:
            dq.pop()
        dq.append((val, x))
    return result
```

## Continuous Subarrays (Count Where Max - Min <= 2)

Count subarrays where max - min <= 2. Two monotonic deques; for each right, find smallest left such that max - min <= 2. All subarrays ending at right with left in [L, right] are valid. So we add (right - L + 1) for each right.

```python
from collections import deque

def continuous_subarrays(nums):
    max_dq = deque()
    min_dq = deque()
    left = 0
    result = 0
    for right, x in enumerate(nums):
        while max_dq and nums[max_dq[-1]] < x:
            max_dq.pop()
        max_dq.append(right)
        while min_dq and nums[min_dq[-1]] > x:
            min_dq.pop()
        min_dq.append(right)
        while nums[max_dq[0]] - nums[min_dq[0]] > 2:
            if max_dq[0] == left:
                max_dq.popleft()
            if min_dq[0] == left:
                min_dq.popleft()
            left += 1
        result += right - left + 1
    return result
```
