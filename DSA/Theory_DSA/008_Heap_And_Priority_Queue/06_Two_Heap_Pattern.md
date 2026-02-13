# Two Heap Pattern

## Theory

The two-heap pattern uses a max-heap for the lower half and a min-heap for the upper half of a stream or dataset. This allows O(log n) insertion and O(1) access to median or other order statistics. Key invariant: max(lo) <= min(hi) and |size(lo) - size(hi)| <= 1.

## Find Median from Data Stream

Max-heap (lo) stores lower half, min-heap (hi) stores upper half. Median is either top of lo (odd) or average of both tops (even).

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.lo = []
        self.hi = []

    def add_num(self, num):
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.lo) < len(self.hi):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def find_median(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2
```

## Sliding Window Median

For each window, maintain two heaps. When sliding, remove old element (lazy delete) and add new. Rebalance when needed.

```python
import heapq

def median_sliding_window(nums, k):
    lo, hi = [], []
    for i in range(k):
        heapq.heappush(lo, -nums[i])
    for _ in range(k // 2):
        heapq.heappush(hi, -heapq.heappop(lo))
    result = [-lo[0]] if k % 2 else [(-lo[0] + hi[0]) / 2]
    to_remove = {}
    for i in range(k, len(nums)):
        out_num = nums[i - k]
        in_num = nums[i]
        balance = 0
        if out_num <= -lo[0]:
            balance -= 1
        else:
            balance += 1
        to_remove[out_num] = to_remove.get(out_num, 0) + 1
        if in_num <= -lo[0]:
            heapq.heappush(lo, -in_num)
            balance += 1
        else:
            heapq.heappush(hi, in_num)
            balance -= 1
        if balance < 0:
            heapq.heappush(lo, -heapq.heappop(hi))
        elif balance > 0:
            heapq.heappush(hi, -heapq.heappop(lo))
        while lo and to_remove.get(-lo[0], 0):
            to_remove[-lo[0]] -= 1
            if to_remove[-lo[0]] == 0:
                del to_remove[-lo[0]]
            heapq.heappop(lo)
        while hi and to_remove.get(hi[0], 0):
            to_remove[hi[0]] -= 1
            if to_remove[hi[0]] == 0:
                del to_remove[hi[0]]
            heapq.heappop(hi)
        if k % 2:
            result.append(-lo[0])
        else:
            result.append((-lo[0] + hi[0]) / 2)
    return result
```

## Maximize Capital (IPO Problem)

Max-heap of profits for affordable projects. Min-heap (or sorted list) of (capital, profit) to add projects as capital grows.

```python
import heapq

def find_maximized_capital(k, w, profits, capital):
    projects = sorted(zip(capital, profits))
    heap = []
    i = 0
    for _ in range(k):
        while i < len(projects) and projects[i][0] <= w:
            heapq.heappush(heap, -projects[i][1])
            i += 1
        if not heap:
            break
        w -= heapq.heappop(heap)
    return w
```

## Find Right Interval

For each interval i, find smallest start >= end(i). Sort by start, use heap or binary search.

```python
def find_right_interval(intervals):
    n = len(intervals)
    starts = [(intervals[i][0], i) for i in range(n)]
    starts.sort(key=lambda x: x[0])
    result = [-1] * n
    for i in range(n):
        end = intervals[i][1]
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if starts[mid][0] < end:
                lo = mid + 1
            else:
                hi = mid
        if starts[lo][0] >= end:
            result[i] = starts[lo][1]
    return result
```

## Balance Two Heaps Approach

General pattern: after each insert, ensure |lo| and |hi| differ by at most 1, and max(lo) <= min(hi).

```python
import heapq

def balance_heaps(lo, hi):
    if len(lo) > len(hi) + 1:
        heapq.heappush(hi, -heapq.heappop(lo))
    elif len(hi) > len(lo):
        heapq.heappush(lo, -heapq.heappop(hi))
```

## Continuous Median

Same as find median from data stream. Process numbers one by one, report median after each.

```python
import heapq

class ContinuousMedian:
    def __init__(self):
        self.lo = []
        self.hi = []

    def add(self, num):
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.lo) < len(self.hi):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def get_median(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2

def continuous_median(nums):
    cm = ContinuousMedian()
    medians = []
    for num in nums:
        cm.add(num)
        medians.append(cm.get_median())
    return medians
```
