# Hard Heap and Priority Queue Problems

## 1. Merge K Sorted Lists (Optimal)

Merge k sorted linked lists. Optimal O(n log k) where n is total nodes. Min-heap of (val, list_id, node). O(n log k) time, O(k) space for heap.

```python
import heapq
def mergeKLists(lists):
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))
    dummy = ListNode(0)
    cur = dummy
    while heap:
        val, i, node = heapq.heappop(heap)
        cur.next = node
        cur = cur.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next
```

Time: O(n log k) | Space: O(k)

---

## 2. Find Median from Data Stream (Follow-up: Delete)

Add, find median, and optionally delete element. Two heaps with lazy deletion. Track deleted elements. On pop, skip deleted. Rebalance when too many deleted at top.

```python
import heapq
class MedianFinder:
    def __init__(self):
        self.lo = []
        self.hi = []
        self.deleted = {}

    def addNum(self, num):
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.lo) < len(self.hi):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def findMedian(self):
        self._clean()
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2

    def removeNum(self, num):
        self.deleted[num] = self.deleted.get(num, 0) + 1

    def _clean(self):
        while self.lo and self.deleted.get(-self.lo[0]):
            self.deleted[-self.lo[0]] -= 1
            heapq.heappop(self.lo)
        while self.hi and self.deleted.get(self.hi[0]):
            self.deleted[self.hi[0]] -= 1
            heapq.heappop(self.hi)
```

Time: O(log n) add/remove, O(1) amortized median | Space: O(n)

---

## 3. Sliding Window Median

Median for each sliding window of size k. Two heaps (lo, hi) with lazy deletion. When window slides, mark old element deleted. Rebalance. O(n log k).

```python
import heapq
def medianSlidingWindow(nums, k):
    lo, hi = [], []
    deleted = {}
    for i in range(k):
        heapq.heappush(lo, -nums[i])
    for _ in range(k // 2):
        heapq.heappush(hi, -heapq.heappop(lo))
    res = [-lo[0]] if k % 2 else ((-lo[0] + hi[0]) / 2,)
    for i in range(k, len(nums)):
        out, inc = nums[i - k], nums[i]
        deleted[out] = deleted.get(out, 0) + 1
        balance = 0
        if inc <= -lo[0]:
            heapq.heappush(lo, -inc)
            balance += 1
        else:
            heapq.heappush(hi, inc)
            balance -= 1
        if out <= -lo[0]:
            balance -= 1
        else:
            balance += 1
        if balance > 0:
            heapq.heappush(hi, -heapq.heappop(lo))
        elif balance < 0:
            heapq.heappush(lo, -heapq.heappop(hi))
        while lo and deleted.get(-lo[0]):
            deleted[-lo[0]] -= 1
            heapq.heappop(lo)
        while hi and deleted.get(hi[0]):
            deleted[hi[0]] -= 1
            heapq.heappop(hi)
        res.append(-lo[0] if k % 2 else (-lo[0] + hi[0]) / 2)
    return list(res)
```

Time: O(n log k) | Space: O(k)

---

## 4. Minimum Cost to Hire K Workers (Full)

Hire exactly k workers. Pay each wage proportional to quality. Minimize total. Sort by wage/quality ratio. For each as captain, take k workers with ratio <= captain. Min-heap of quality to maintain k smallest quality sum. O(n log k).

```python
import heapq
def mincostToHireWorkers(quality, wage, k):
    workers = sorted(zip(wage, quality), key=lambda x: x[0] / x[1])
    heap = []
    qsum = 0
    res = float('inf')
    for w, q in workers:
        heapq.heappush(heap, -q)
        qsum += q
        if len(heap) > k:
            qsum += heapq.heappop(heap)
        if len(heap) == k:
            res = min(res, qsum * w / q)
    return res
```

Time: O(n log k) | Space: O(k)

---

## 5. IPO (Multiple Rounds)

Same as medium but with capital constraints and project dependencies. Max-heap of profits. Sort projects by capital. Greedy select k. Handle project prerequisites with topological order if needed.

```python
import heapq
def findMaximizedCapital(k, w, profits, capital):
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

Time: O(n log n) | Space: O(n)

---

## 6. Smallest Range Covering Elements from K Lists

Pick one element from each list. Minimize range [min, max]. Min-heap (val, list_id, idx). Track current max. Pop min, expand that list. Update best range. O(n log k).

```python
import heapq
def smallestRange(nums):
    heap = [(row[0], i, 0) for i, row in enumerate(nums)]
    heapq.heapify(heap)
    max_val = max(row[0] for row in nums)
    res = [heap[0][0], max_val]
    while True:
        min_val, i, j = heapq.heappop(heap)
        if max_val - min_val < res[1] - res[0]:
            res = [min_val, max_val]
        if j + 1 >= len(nums[i]):
            break
        nxt = nums[i][j + 1]
        max_val = max(max_val, nxt)
        heapq.heappush(heap, (nxt, i, j + 1))
    return res
```

Time: O(n log k) | Space: O(k)

---

## 7. Trapping Rain Water II

2D elevation map. Water trapped after raining. Min-heap of boundary cells. Expand inward from boundary. Water level = max(heap min, cell height). O(mn log(mn)).

```python
import heapq
def trapRainWater(heightMap):
    if not heightMap or not heightMap[0]:
        return 0
    m, n = len(heightMap), len(heightMap[0])
    seen = [[False] * n for _ in range(m)]
    heap = []
    for i in range(m):
        for j in [0, n - 1]:
            heapq.heappush(heap, (heightMap[i][j], i, j))
            seen[i][j] = True
    for j in range(n):
        for i in [0, m - 1]:
            if not seen[i][j]:
                heapq.heappush(heap, (heightMap[i][j], i, j))
                seen[i][j] = True
    res = 0
    while heap:
        h, r, c = heapq.heappop(heap)
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and not seen[nr][nc]:
                seen[nr][nc] = True
                res += max(0, h - heightMap[nr][nc])
                heapq.heappush(heap, (max(h, heightMap[nr][nc]), nr, nc))
    return res
```

Time: O(m*n log(m*n)) | Space: O(m*n)

---

## 8. Kth Smallest Prime Fraction

Sorted array of primes. Kth smallest fraction p[i]/p[j] where i < j. Min-heap (p[i]/p[j], i, j). Start with (p[0]/p[j]) for all j. Pop k times, push (p[i+1]/p[j], i+1, j).

```python
import heapq
def kthSmallestPrimeFraction(arr, k):
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

Time: O(k log n) | Space: O(n)

---

## 9. Minimum Number of Refueling Stops

Drive to target. Limited fuel. Gas stations with position and fuel. Min stops. Max-heap of fuel at passed stations. When fuel runs out, refuel from largest. Greedy. O(n log n).

```python
import heapq
def minRefuelStops(target, startFuel, stations):
    heap = []
    fuel = startFuel
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

Time: O(n log n) | Space: O(n)

---

## 10. Process Tasks Using Servers (Concurrent)

Multiple tasks per second. Assign to available server with smallest weight. Available heap (weight, idx), busy heap (free_time, weight, idx). At each time, free completed, assign new. O(n log n).

```python
import heapq
def assignTasks(servers, tasks):
    available = [(w, i) for i, w in enumerate(servers)]
    heapq.heapify(available)
    busy = []
    res = []
    for t in range(len(tasks)):
        while busy and busy[0][0] <= t:
            _, w, i = heapq.heappop(busy)
            heapq.heappush(available, (w, i))
        if available:
            w, i = heapq.heappop(available)
            res.append(i)
            heapq.heappush(busy, (t + tasks[t], w, i))
        else:
            et, w, i = heapq.heappop(busy)
            res.append(i)
            heapq.heappush(busy, (et + tasks[t], w, i))
    return res
```

Time: O(n log n) | Space: O(n)

---

## 11. Maximum Performance of a Team

n engineers with speed and efficiency. Pick k. Performance = sum(speed) * min(efficiency). Maximize. Sort by efficiency descending. For each as min efficiency, take k fastest (min-heap of speed, keep k largest). O(n log k).

```python
import heapq
def maxPerformance(n, speed, efficiency, k):
    engineers = sorted(zip(efficiency, speed), reverse=True)
    heap = []
    total_speed = 0
    res = 0
    for eff, spd in engineers:
        heapq.heappush(heap, spd)
        total_speed += spd
        if len(heap) > k:
            total_speed -= heapq.heappop(heap)
        res = max(res, total_speed * eff)
    return res % (10**9 + 7)
```

Time: O(n log k) | Space: O(k)

---

## 12. Find the K-Sum of an Array

Array. Subsequence sum = sum of selected elements. Find kth largest subsequence sum. Sort, take positive and largest negative. Max-heap of (sum, last_index). Expand by including next or excluding. Complex state.

```python
import heapq
def kSum(nums, k):
    total = sum(max(0, x) for x in nums)
    nums = sorted(abs(x) for x in nums)
    heap = [(-(total - nums[0]), 0)]
    for _ in range(k - 1):
        s, i = heapq.heappop(heap)
        s = -s
        if i + 1 < len(nums):
            heapq.heappush(heap, (-(s - nums[i+1] + nums[i]), i + 1))
            heapq.heappush(heap, (-(s - nums[i+1]), i + 1))
    return -heapq.heappop(heap)[0]
```

Time: O(n log n + k log k) | Space: O(k)

---

## 13. Construct Target Array With Multiple Sums

Start with [1,1,...,1]. Operation: replace element with sum of all. Can we get target? Reverse process. Max-heap of target. Largest = sum of rest. Replace with (largest - sum_rest). Check if we reach all ones.

```python
import heapq
def isPossible(target):
    total = sum(target)
    heap = [-x for x in target]
    heapq.heapify(heap)
    while heap[0] != -1:
        largest = -heapq.heappop(heap)
        rest = total - largest
        if rest == 0 or largest <= rest:
            return False
        prev = largest % rest if rest > 0 else largest - rest
        if prev < 1:
            return False
        heapq.heappush(heap, -prev)
        total = total - largest + prev
    return True
```

Time: O(n log n * log(max)) | Space: O(n)

---

## 14. Minimum Cost to Reach Destination in Time

Graph with edges (time, cost). Reach destination within maxTime. Minimize cost. Dijkstra-like with (cost, node, time). State = (node, time). Min-heap by cost. Prune if time exceeded.

```python
import heapq
def minCost(maxTime, edges, passingFees):
    n = len(passingFees)
    graph = [[] for _ in range(n)]
    for u, v, t in edges:
        graph[u].append((v, t))
        graph[v].append((u, t))
    dist = {(0, 0): passingFees[0]}
    heap = [(passingFees[0], 0, 0)]
    while heap:
        cost, node, time = heapq.heappop(heap)
        if node == n - 1:
            return cost
        if (node, time) in dist and dist[(node, time)] < cost:
            continue
        for v, t in graph[node]:
            nt = time + t
            if nt <= maxTime:
                nc = cost + passingFees[v]
                if (v, nt) not in dist or dist[(v, nt)] > nc:
                    dist[(v, nt)] = nc
                    heapq.heappush(heap, (nc, v, nt))
    return -1
```

Time: O(E log V) | Space: O(V)

---

## 15. Number of Orders in the Backlog

Buy/sell orders. Match when buy price >= sell price. Return sum of remaining order amounts. Max-heap for buy orders, min-heap for sell orders. Match top of both. Process orders in sequence.

```python
import heapq
def getNumberOfBacklogOrders(orders):
    buy = []
    sell = []
    for price, amount, orderType in orders:
        if orderType == 0:
            while amount and sell and sell[0][0] <= price:
                p, a = heapq.heappop(sell)
                take = min(amount, a)
                amount -= take
                a -= take
                if a:
                    heapq.heappush(sell, (p, a))
            if amount:
                heapq.heappush(buy, (-price, amount))
        else:
            while amount and buy and -buy[0][0] >= price:
                p, a = heapq.heappop(buy)
                take = min(amount, a)
                amount -= take
                a -= take
                if a:
                    heapq.heappush(buy, (p, a))
            if amount:
                heapq.heappush(sell, (price, amount))
    return (sum(a for _, a in buy) + sum(a for _, a in sell)) % (10**9 + 7)
```

Time: O(n log n) | Space: O(n)

---

## 16. Maximum Number of Events That Can Be Attended

Events have [start, end]. Attend one per day. Maximize events. Sort by start. Min-heap of end times for events starting today. Each day, pop events that already ended. Attend one. Greedy.

```python
import heapq
def maxEvents(events):
    events.sort()
    heap = []
    i = 0
    res = 0
    for d in range(1, 100001):
        while i < len(events) and events[i][0] == d:
            heapq.heappush(heap, events[i][1])
            i += 1
        while heap and heap[0] < d:
            heapq.heappop(heap)
        if heap:
            heapq.heappop(heap)
            res += 1
    return res
```

Time: O(n log n) | Space: O(n)

---

## 17. Minimum Interval to Include Each Query

Intervals and queries. For each query, find smallest interval length that contains it. Sort intervals by left. For each query, add intervals with left <= query to heap (by right). Remove intervals with right < query. Min-heap by (length, right).

```python
import heapq
def minInterval(intervals, queries):
    intervals.sort()
    qs = sorted(enumerate(queries), key=lambda x: x[1])
    heap = []
    i = 0
    res = [0] * len(queries)
    for idx, q in qs:
        while i < len(intervals) and intervals[i][0] <= q:
            l, r = intervals[i]
            heapq.heappush(heap, (r - l + 1, r))
            i += 1
        while heap and heap[0][1] < q:
            heapq.heappop(heap)
        res[idx] = heap[0][0] if heap else -1
    return res
```

Time: O(n log n + q log q) | Space: O(n)

---

## 18. The Skyline Problem

Buildings [left, right, height]. Return skyline key points. Sweep line. Events (left, -height) and (right, height). Max-heap of active heights. When max changes, add point. Lazy deletion.

```python
import heapq
def getSkyline(buildings):
    events = []
    for l, r, h in buildings:
        events.append((l, -h, r))
        events.append((r, 0, 0))
    events.sort()
    heap = [(0, float('inf'))]
    res = []
    for x, neg_h, r in events:
        while heap[0][1] <= x:
            heapq.heappop(heap)
        if neg_h:
            heapq.heappush(heap, (neg_h, r))
        if not res or res[-1][1] != -heap[0][0]:
            res.append([x, -heap[0][0]])
    return res
```

Time: O(n log n) | Space: O(n)
