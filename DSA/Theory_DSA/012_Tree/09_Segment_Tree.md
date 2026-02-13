# Segment Tree

## Concept

Binary tree over array. Each node represents a segment [l, r]. Root = [0, n-1]. Left child = [l, mid], right child = [mid+1, r]. Leaves = single elements. Supports range queries and range updates in O(log n).

## Build O(n)

```python
def build_segment_tree(arr):
    n = len(arr)
    size = 1
    while size < n:
        size *= 2
    tree = [0] * (2 * size)
    for i in range(n):
        tree[size + i] = arr[i]
    for i in range(size - 1, 0, -1):
        tree[i] = tree[2 * i] + tree[2 * i + 1]
    return tree, size
```

## Point Update O(log n)

```python
def point_update(tree, size, idx, val):
    idx += size
    tree[idx] = val
    idx //= 2
    while idx:
        tree[idx] = tree[2 * idx] + tree[2 * idx + 1]
        idx //= 2
```

## Range Query Sum O(log n)

```python
def range_query_sum(tree, size, l, r):
    l += size
    r += size
    total = 0
    while l <= r:
        if l % 2 == 1:
            total += tree[l]
            l += 1
        if r % 2 == 0:
            total += tree[r]
            r -= 1
        l //= 2
        r //= 2
    return total
```

## Range Query Min

```python
def build_min_tree(arr):
    n = len(arr)
    size = 1
    while size < n:
        size *= 2
    tree = [float('inf')] * (2 * size)
    for i in range(n):
        tree[size + i] = arr[i]
    for i in range(size - 1, 0, -1):
        tree[i] = min(tree[2 * i], tree[2 * i + 1])
    return tree, size

def range_query_min(tree, size, l, r):
    l += size
    r += size
    res = float('inf')
    while l <= r:
        if l % 2 == 1:
            res = min(res, tree[l])
            l += 1
        if r % 2 == 0:
            res = min(res, tree[r])
            r -= 1
        l //= 2
        r //= 2
    return res
```

## Range Query Max

```python
def build_max_tree(arr):
    n = len(arr)
    size = 1
    while size < n:
        size *= 2
    tree = [float('-inf')] * (2 * size)
    for i in range(n):
        tree[size + i] = arr[i]
    for i in range(size - 1, 0, -1):
        tree[i] = max(tree[2 * i], tree[2 * i + 1])
    return tree, size

def range_query_max(tree, size, l, r):
    l += size
    r += size
    res = float('-inf')
    while l <= r:
        if l % 2 == 1:
            res = max(res, tree[l])
            l += 1
        if r % 2 == 0:
            res = max(res, tree[r])
            r -= 1
        l //= 2
        r //= 2
    return res
```

## Range Query GCD

```python
import math

def build_gcd_tree(arr):
    n = len(arr)
    size = 1
    while size < n:
        size *= 2
    tree = [0] * (2 * size)
    for i in range(n):
        tree[size + i] = arr[i]
    for i in range(size - 1, 0, -1):
        tree[i] = math.gcd(tree[2 * i], tree[2 * i + 1])
    return tree, size

def range_query_gcd(tree, size, l, r):
    l += size
    r += size
    res = 0
    while l <= r:
        if l % 2 == 1:
            res = math.gcd(res, tree[l])
            l += 1
        if r % 2 == 0:
            res = math.gcd(res, tree[r])
            r -= 1
        l //= 2
        r //= 2
    return res
```

## Lazy Propagation for Range Update O(log n)

Defer updates to children. Store pending updates in lazy array. Apply when querying or updating.

```python
def build_lazy_tree(arr):
    n = len(arr)
    size = 1
    while size < n:
        size *= 2
    tree = [0] * (2 * size)
    lazy = [0] * (2 * size)
    for i in range(n):
        tree[size + i] = arr[i]
    for i in range(size - 1, 0, -1):
        tree[i] = tree[2 * i] + tree[2 * i + 1]
    return tree, lazy, size

def push(tree, lazy, idx, seg_len):
    if lazy[idx]:
        tree[idx] += lazy[idx] * seg_len
        if idx < len(tree) // 2:
            lazy[2 * idx] += lazy[idx]
            lazy[2 * idx + 1] += lazy[idx]
        lazy[idx] = 0

def range_update_lazy(tree, lazy, size, l, r, delta):
    def update(l, r, delta, idx, seg_l, seg_r):
        seg_len = seg_r - seg_l + 1
        push(tree, lazy, idx, seg_len)
        if r < seg_l or l > seg_r:
            return
        if l <= seg_l and seg_r <= r:
            lazy[idx] += delta
            push(tree, lazy, idx, seg_len)
            return
        mid = (seg_l + seg_r) // 2
        update(l, r, delta, 2 * idx, seg_l, mid)
        update(l, r, delta, 2 * idx + 1, mid + 1, seg_r)
        tree[idx] = tree[2 * idx] + tree[2 * idx + 1]
    update(l, r, delta, 1, 0, size - 1)

def range_query_lazy(tree, lazy, size, l, r):
    def query(l, r, idx, seg_l, seg_r):
        seg_len = seg_r - seg_l + 1
        push(tree, lazy, idx, seg_len)
        if r < seg_l or l > seg_r:
            return 0
        if l <= seg_l and seg_r <= r:
            return tree[idx]
        mid = (seg_l + seg_r) // 2
        return query(l, r, 2 * idx, seg_l, mid) + query(l, r, 2 * idx + 1, mid + 1, seg_r)
    return query(l, r, 1, 0, size - 1)
```

## Persistent Segment Tree Overview

Each update creates new nodes instead of modifying. Keeps history. O(log n) per update, O(log n) per query. Used for range queries over different versions.

## Problems

### Range Sum Query Mutable

```python
class NumArray:
    def __init__(self, nums):
        self.n = len(nums)
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        self.tree = [0] * (2 * self.size)
        for i in range(self.n):
            self.tree[self.size + i] = nums[i]
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.tree[2*i] + self.tree[2*i+1]

    def update(self, index, val):
        index += self.size
        self.tree[index] = val
        index //= 2
        while index:
            self.tree[index] = self.tree[2*index] + self.tree[2*index+1]
            index //= 2

    def sumRange(self, left, right):
        left += self.size
        right += self.size
        total = 0
        while left <= right:
            if left % 2 == 1:
                total += self.tree[left]
                left += 1
            if right % 2 == 0:
                total += self.tree[right]
                right -= 1
            left //= 2
            right //= 2
        return total
```

### Range Minimum Query

Use min segment tree. Build and query as shown in Range Query Min section.

### Count of Smaller Numbers After Self

Coordinate compression + segment tree. For each element from right to left, query count in [0, rank-1], then update at rank.

```python
def count_smaller(nums):
    sorted_nums = sorted(set(nums))
    rank = {v: i for i, v in enumerate(sorted_nums)}
    n = len(nums)
    size = 1
    while size < n:
        size *= 2
    tree = [0] * (2 * size)
    result = []
    for i in range(n - 1, -1, -1):
        r = rank[nums[i]]
        count = range_query_sum(tree, size, 0, r - 1) if r > 0 else 0
        result.append(count)
        idx = r + size
        tree[idx] += 1
        idx //= 2
        while idx:
            tree[idx] += 1
            idx //= 2
    return result[::-1]
```

### Count of Range Sum

Prefix sums + segment tree. For each prefix sum, count how many previous prefix sums in [sum - upper, sum - lower]. Coordinate compress prefix sums.
