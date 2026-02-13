# Binary Indexed Tree (Fenwick Tree)

## Concept

Array-based structure for prefix sum queries and point updates. Uses binary representation: each index i stores sum of range [i - lowbit(i) + 1, i]. Space O(n), point update O(log n), prefix sum O(log n).

## Lowbit Operation (x & -x)

Returns lowest set bit. For index i, lowbit gives the range size that i is responsible for.

```python
def lowbit(x):
    return x & (-x)
```

## Point Update O(log n)

Add delta to index and propagate to ancestors (add lowbit).

```python
def point_update(bit, idx, delta):
    n = len(bit)
    while idx < n:
        bit[idx] += delta
        idx += idx & (-idx)
```

## Prefix Sum Query O(log n)

Sum from 1 to idx by traversing indices (subtract lowbit).

```python
def prefix_sum(bit, idx):
    total = 0
    while idx > 0:
        total += bit[idx]
        idx -= idx & (-idx)
    return total
```

## Range Sum Query

Query(l, r) = prefix_sum(r) - prefix_sum(l-1)

```python
def range_sum(bit, l, r):
    return prefix_sum(bit, r) - prefix_sum(bit, l - 1)
```

## Build from Array O(n)

```python
def build_bit(arr):
    n = len(arr)
    bit = [0] * (n + 1)
    for i in range(n):
        idx = i + 1
        bit[idx] += arr[i]
        next_idx = idx + (idx & (-idx))
        if next_idx <= n:
            bit[next_idx] += bit[idx]
    return bit
```

Alternative: initialize zeros, then point_update for each element.

```python
def build_bit_simple(arr):
    n = len(arr)
    bit = [0] * (n + 1)
    for i in range(n):
        point_update(bit, i + 1, arr[i])
    return bit
```

## Range Update + Point Query (Difference BIT)

For range [l, r] add delta: update(l, delta), update(r+1, -delta). Point query = prefix sum.

```python
def range_update_point_query(arr, l, r, delta):
    n = len(arr)
    bit = [0] * (n + 2)
    point_update(bit, l + 1, delta)
    point_update(bit, r + 2, -delta)

def get_point(bit, idx):
    return prefix_sum(bit, idx + 1)
```

## Range Update + Range Query (Two BITs)

Use B1 for range update, B2 for adjustment. Update [l, r] with delta: B1[l]+=delta, B1[r+1]-=delta, B2[l]+=delta*(l-1), B2[r+1]-=delta*r. Prefix sum = idx*prefix_sum(B1, idx) - prefix_sum(B2, idx).

```python
def range_update_range_query_init(n):
    return [0] * (n + 1), [0] * (n + 1)

def range_update(bit1, bit2, l, r, delta):
    point_update(bit1, l, delta)
    point_update(bit1, r + 1, -delta)
    point_update(bit2, l, delta * (l - 1))
    point_update(bit2, r + 1, -delta * r)

def range_query_prefix(bit1, bit2, idx):
    return idx * prefix_sum(bit1, idx) - prefix_sum(bit2, idx)

def range_query(bit1, bit2, l, r):
    return range_query_prefix(bit1, bit2, r) - range_query_prefix(bit1, bit2, l - 1)
```

## 2D BIT Overview

Two nested loops for update and query. Update (i, j) affects all (i', j') where i' >= i, j' >= j. Query (i, j) sums rectangle (1,1) to (i,j).

## Comparison with Segment Tree

| Aspect | Fenwick Tree | Segment Tree |
|--------|--------------|--------------|
| Space | O(n) | O(4n) |
| Build | O(n) | O(n) |
| Point update | O(log n) | O(log n) |
| Range query | O(log n) | O(log n) |
| Range update | Two BITs | Lazy propagation |
| Flexibility | Prefix/range sum | Any associative op |
| Code | Shorter | Longer |

## Problems

### Count Inversions

```python
def count_inversions(arr):
    sorted_arr = sorted(set(arr))
    rank = {v: i + 1 for i, v in enumerate(sorted_arr)}
    n = len(arr)
    bit = [0] * (n + 1)
    inv = 0
    for i in range(n - 1, -1, -1):
        r = rank[arr[i]]
        inv += prefix_sum(bit, r - 1)
        point_update(bit, r, 1)
    return inv
```

### Range Sum Query Mutable

```python
class NumArray:
    def __init__(self, nums):
        self.n = len(nums)
        self.nums = nums
        self.bit = [0] * (self.n + 1)
        for i in range(self.n):
            self._update(i + 1, nums[i])

    def _update(self, idx, delta):
        while idx <= self.n:
            self.bit[idx] += delta
            idx += idx & (-idx)

    def update(self, index, val):
        delta = val - self.nums[index]
        self.nums[index] = val
        self._update(index + 1, delta)

    def sumRange(self, left, right):
        def prefix(idx):
            total = 0
            while idx > 0:
                total += self.bit[idx]
                idx -= idx & (-idx)
            return total
        return prefix(right + 1) - prefix(left)
```
