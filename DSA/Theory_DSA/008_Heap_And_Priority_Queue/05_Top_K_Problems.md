# Top-K Problems

## Theory

Top-K problems ask for the k largest, k smallest, k closest, or k most frequent elements. A min-heap of size k keeps the k largest (pop when size > k); a max-heap of size k keeps the k smallest. Time complexity is typically O(n log k).

## Kth Largest Element in Array

Use min-heap of size k. When heap exceeds k, pop minimum. Final root is kth largest.

```python
import heapq

def find_kth_largest(nums, k):
    heap = []
    for x in nums:
        heapq.heappush(heap, x)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap[0]
```

## Kth Smallest Element

Use max-heap of size k. Python heapq is min-heap, so negate values.

```python
import heapq

def find_kth_smallest(nums, k):
    heap = []
    for x in nums:
        heapq.heappush(heap, -x)
        if len(heap) > k:
            heapq.heappop(heap)
    return -heap[0]
```

## K Largest Elements

Same as kth largest but return all k elements.

```python
import heapq

def k_largest_elements(nums, k):
    heap = []
    for x in nums:
        heapq.heappush(heap, x)
        if len(heap) > k:
            heapq.heappop(heap)
    return sorted(heap, reverse=True)
```

## K Smallest Elements

```python
import heapq

def k_smallest_elements(nums, k):
    heap = []
    for x in nums:
        heapq.heappush(heap, -x)
        if len(heap) > k:
            heapq.heappop(heap)
    return sorted([-x for x in heap])
```

## K Closest Points to Origin

Min-heap by distance squared (avoid sqrt). Keep k smallest distances.

```python
import heapq

def k_closest_points(points, k):
    def dist_sq(p):
        return p[0]*p[0] + p[1]*p[1]
    heap = []
    for p in points:
        heapq.heappush(heap, (-dist_sq(p), p))
        if len(heap) > k:
            heapq.heappop(heap)
    return [p for _, p in heap]
```

## Top K Frequent Elements

Count frequencies, then min-heap of (freq, num) of size k.

```python
import heapq
from collections import Counter

def top_k_frequent(nums, k):
    cnt = Counter(nums)
    heap = []
    for num, freq in cnt.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)
    return [num for _, num in heap]
```

## Top K Frequent Words

Same idea with lexicographic tie-breaker: use (-freq, word) so higher freq and smaller word come first.

```python
import heapq
from collections import Counter

def top_k_frequent_words(words, k):
    cnt = Counter(words)
    heap = []
    for word, freq in cnt.items():
        heapq.heappush(heap, (-freq, word))
        if len(heap) > k:
            heapq.heappop(heap)
    heap.sort(key=lambda x: (x[0], x[1]))
    return [word for _, word in heap]
```

## Sort Characters by Frequency

Count, then max-heap by frequency, pop and build result.

```python
import heapq
from collections import Counter

def frequency_sort(s):
    cnt = Counter(s)
    heap = [(-freq, c) for c, freq in cnt.items()]
    heapq.heapify(heap)
    result = []
    while heap:
        neg_freq, c = heapq.heappop(heap)
        result.append(c * (-neg_freq))
    return ''.join(result)
```

## K Closest Numbers to Target in Sorted Array

Binary search for position, then two pointers or heap of (distance, value) from expanding window.

```python
import heapq

def k_closest_to_target(arr, k, target):
    heap = []
    for x in arr:
        heapq.heappush(heap, (abs(x - target), x))
        if len(heap) > k:
            heapq.heappop(heap)
    return sorted([x for _, x in heap])
```

## K Closest Elements

Return k elements from sorted array closest to x. Use heap of (distance, value).

```python
import heapq

def find_k_closest_elements(arr, k, x):
    heap = []
    for v in arr:
        heapq.heappush(heap, (abs(v - x), v))
        if len(heap) > k:
            heapq.heappop(heap)
    return sorted([v for _, v in heap])
```

## Find K Pairs with Smallest Sums

Given two arrays, find k pairs (a,b) with smallest a+b. Min-heap of (sum, i, j). Start with (nums1[0]+nums2[0], 0, 0). Expand by (i+1,j) and (i,j+1).

```python
import heapq

def k_smallest_pairs(nums1, nums2, k):
    if not nums1 or not nums2:
        return []
    heap = [(nums1[0] + nums2[0], 0, 0)]
    seen = {(0, 0)}
    result = []
    while heap and len(result) < k:
        _, i, j = heapq.heappop(heap)
        result.append([nums1[i], nums2[j]])
        if i + 1 < len(nums1) and (i + 1, j) not in seen:
            seen.add((i + 1, j))
            heapq.heappush(heap, (nums1[i+1] + nums2[j], i + 1, j))
        if j + 1 < len(nums2) and (i, j + 1) not in seen:
            seen.add((i, j + 1))
            heapq.heappush(heap, (nums1[i] + nums2[j+1], i, j + 1))
    return result
```

## Sort Nearly Sorted Array (K-Sorted)

Each element is at most k positions from its sorted position. Min-heap of size k+1, slide window.

```python
import heapq

def sort_k_sorted(arr, k):
    heap = arr[:k+1]
    heapq.heapify(heap)
    result = []
    for i in range(k+1, len(arr)):
        result.append(heapq.heappop(heap))
        heapq.heappush(heap, arr[i])
    while heap:
        result.append(heapq.heappop(heap))
    return result
```

## Kth Smallest in Sorted Matrix

Min-heap of (value, row, col). Start with (matrix[0][0], 0, 0). Pop k times, push (row+1,col) and (row,col+1) if not seen.

```python
import heapq

def kth_smallest_matrix(matrix, k):
    n = len(matrix)
    heap = [(matrix[0][0], 0, 0)]
    seen = {(0, 0)}
    for _ in range(k - 1):
        val, r, c = heapq.heappop(heap)
        if r + 1 < n and (r + 1, c) not in seen:
            seen.add((r + 1, c))
            heapq.heappush(heap, (matrix[r+1][c], r + 1, c))
        if c + 1 < n and (r, c + 1) not in seen:
            seen.add((r, c + 1))
            heapq.heappush(heap, (matrix[r][c+1], r, c + 1))
    return heapq.heappop(heap)[0]
```

## Kth Smallest Prime Fraction

Array of primes in ascending order. Min-heap of (p[i]/p[j], i, j) for i < j. Pop k times.

```python
import heapq

def kth_smallest_prime_fraction(arr, k):
    n = len(arr)
    heap = [(arr[0] / arr[j], 0, j) for j in range(1, n)]
    heapq.heapify(heap)
    for _ in range(k - 1):
        _, i, j = heapq.heappop(heap)
        if i + 1 < j:
            heapq.heappush(heap, (arr[i+1] / arr[j], i + 1, j))
    _, i, j = heapq.heappop(heap)
    return [arr[i], arr[j]]
```

## Find Median from Data Stream

Two heaps: max-heap for left half, min-heap for right half. Keep sizes balanced.

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

## Reorganize String

Max-heap by frequency. Pop two most frequent, append to result, decrement and push back if count > 0. Avoid same char adjacent.

```python
import heapq
from collections import Counter

def reorganize_string(s):
    cnt = Counter(s)
    heap = [(-freq, c) for c, freq in cnt.items()]
    heapq.heapify(heap)
    result = []
    prev = None
    while heap:
        neg_freq, c = heapq.heappop(heap)
        if prev == c and not heap:
            return ""
        if prev == c:
            neg_freq2, c2 = heapq.heappop(heap)
            result.append(c2)
            prev = c2
            if neg_freq2 + 1 < 0:
                heapq.heappush(heap, (neg_freq2 + 1, c2))
            heapq.heappush(heap, (neg_freq, c))
        else:
            result.append(c)
            prev = c
            if neg_freq + 1 < 0:
                heapq.heappush(heap, (neg_freq + 1, c))
    return ''.join(result)
```

## Task Scheduler

Max-heap of (count, task). Each round, pop up to n+1 tasks, execute, decrement count, push back. Idle if heap empty.

```python
import heapq
from collections import Counter

def least_interval(tasks, n):
    cnt = Counter(tasks)
    heap = [-c for c in cnt.values()]
    heapq.heapify(heap)
    time = 0
    while heap:
        batch = []
        for _ in range(n + 1):
            if heap:
                c = heapq.heappop(heap)
                if c + 1 < 0:
                    batch.append(c + 1)
            time += 1
            if not heap and not batch:
                break
        for c in batch:
            heapq.heappush(heap, c)
    return time
```

## Furthest Building You Can Reach

Greedy: use ladder for largest jumps, bricks for rest. Min-heap of ladder jumps; when heap size > ladders, use bricks for smallest.

```python
import heapq

def furthest_building(heights, bricks, ladders):
    heap = []
    for i in range(len(heights) - 1):
        d = heights[i + 1] - heights[i]
        if d <= 0:
            continue
        heapq.heappush(heap, d)
        if len(heap) > ladders:
            bricks -= heapq.heappop(heap)
            if bricks < 0:
                return i
    return len(heights) - 1
```

## Minimum Cost to Hire K Workers

For each worker as "captain" (ratio = wage/quality), we want k workers with smallest quality sum among those with ratio <= captain. Sort by ratio, min-heap of quality (negated for max), keep k largest qualities, compute cost.

```python
import heapq

def mincost_to_hire_workers(quality, wage, k):
    workers = sorted((w / q, q) for q, w in zip(quality, wage))
    heap = []
    qsum = 0
    result = float('inf')
    for ratio, q in workers:
        heapq.heappush(heap, -q)
        qsum += q
        if len(heap) > k:
            qsum += heapq.heappop(heap)
        if len(heap) == k:
            result = min(result, ratio * qsum)
    return result
```

## IPO - Maximize Capital

Max-heap of (profit, capital). Sort projects by capital. Add all affordable projects to heap. Pop k times for max profit.

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

## Last Stone Weight

Max-heap. Pop two largest, push difference. Repeat until one or zero left.

```python
import heapq

def last_stone_weight(stones):
    heap = [-s for s in stones]
    heapq.heapify(heap)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        if a != b:
            heapq.heappush(heap, a - b)
    return -heap[0] if heap else 0
```

## Minimum Refueling Stops

Max-heap of fuel at passed stations. Drive until out of fuel, then refuel from largest passed station. Repeat until target or no more fuel.

```python
import heapq

def min_refuel_stops(target, start_fuel, stations):
    heap = []
    fuel = start_fuel
    stops = 0
    i = 0
    while fuel < target:
        while i < len(stations) and stations[i][0] <= fuel:
            heapq.heappush(heap, -stations[i][1])
            i += 1
        if not heap:
            return -1
        fuel -= heapq.heappop(heap)
        stops += 1
    return stops
```

## Smallest Range Covering Elements from K Lists

Min-heap of (value, list_id, index). Track current max. Pop min, expand that list, update range.

```python
import heapq

def smallest_range(nums):
    heap = [(row[0], i, 0) for i, row in enumerate(nums)]
    heapq.heapify(heap)
    cur_max = max(row[0] for row in nums)
    best_lo, best_hi = float('-inf'), float('inf')
    while True:
        cur_min, list_id, idx = heapq.heappop(heap)
        if cur_max - cur_min < best_hi - best_lo:
            best_lo, best_hi = cur_min, cur_max
        if idx + 1 >= len(nums[list_id]):
            break
        nxt = nums[list_id][idx + 1]
        cur_max = max(cur_max, nxt)
        heapq.heappush(heap, (nxt, list_id, idx + 1))
    return [best_lo, best_hi]
```

## Meeting Rooms (Min Rooms)

Sort intervals by start. Min-heap of end times. For each interval, if start >= min(end), pop. Push new end. Answer is max heap size.

```python
import heapq

def min_meeting_rooms(intervals):
    intervals.sort(key=lambda x: x[0])
    heap = []
    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heappop(heap)
        heapq.heappush(heap, end)
    return len(heap)
```

## Single-Threaded CPU Scheduling

Min-heap of (enqueue_time, processing_time, index). Also min-heap of available tasks by processing time. Simulate time.

```python
import heapq

def get_order(tasks):
    indexed = [(t[0], t[1], i) for i, t in enumerate(tasks)]
    indexed.sort(key=lambda x: x[0])
    heap = []
    time = 0
    i = 0
    result = []
    while i < len(indexed) or heap:
        while i < len(indexed) and indexed[i][0] <= time:
            heapq.heappush(heap, (indexed[i][1], indexed[i][2]))
            i += 1
        if not heap:
            time = indexed[i][0]
            continue
        pt, idx = heapq.heappop(heap)
        time += pt
        result.append(idx)
    return result
```

## Process Tasks Using Servers

Two heaps: available (weight, index), busy (free_time, weight, index). At each task time, free servers from busy, assign to available with smallest weight.

```python
import heapq

def assign_tasks(servers, tasks):
    available = [(servers[i], i) for i in range(len(servers))]
    heapq.heapify(available)
    busy = []
    result = []
    for t in range(len(tasks)):
        while busy and busy[0][0] <= t:
            _, w, i = heapq.heappop(busy)
            heapq.heappush(available, (w, i))
        if available:
            w, i = heapq.heappop(available)
            result.append(i)
            heapq.heappush(busy, (t + tasks[t], w, i))
        else:
            free_t, w, i = heapq.heappop(busy)
            result.append(i)
            heapq.heappush(busy, (free_t + tasks[t], w, i))
    return result
```
