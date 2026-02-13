# Medium Heap and Priority Queue Problems

## 1. Find Median from Data Stream

Design structure to add numbers and return median. Two heaps - max-heap for lower half, min-heap for upper half. Keep balanced. O(log n) add, O(1) median.

```python
import heapq
class MedianFinder:
    def __init__(self):
        self.lo = []
        self.hi = []

    def addNum(self, num):
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.lo) < len(self.hi):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def findMedian(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2
```

Time: O(log n) add, O(1) median | Space: O(n)

---

## 2. Task Scheduler

Schedule tasks with cooldown n between same task. Minimize total time. Max-heap by frequency. Each round pop up to n+1 tasks. Idle if heap empty. O(time) simulation.

```python
import heapq
from collections import Counter
def leastInterval(tasks, n):
    cnt = Counter(tasks)
    heap = [-c for c in cnt.values()]
    heapq.heapify(heap)
    time = 0
    while heap:
        buf = []
        for _ in range(min(n + 1, len(heap))):
            c = heapq.heappop(heap) + 1
            if c:
                buf.append(c)
            time += 1
        for c in buf:
            heapq.heappush(heap, -c)
        if heap and len(buf) < n + 1:
            time += n + 1 - len(buf)
    return time
```

Time: O(n) | Space: O(1)

---

## 3. Top K Frequent Words

Return k most frequent words. Tie-break: lexicographically smaller first. Count, min-heap of (-freq, word) size k. Custom comparator for tie-break.

```python
import heapq
from collections import Counter
def topKFrequent(words, k):
    cnt = Counter(words)
    heap = []
    for word, freq in cnt.items():
        heapq.heappush(heap, (-freq, word))
    return [heapq.heappop(heap)[1] for _ in range(k)]
```

Time: O(n log k) | Space: O(n)

---

## 4. Reorganize String

Reorder so no two adjacent chars same. Max-heap by frequency. Pop two most frequent alternately. Fail if one char > half.

```python
import heapq
from collections import Counter
def reorganizeString(s):
    cnt = Counter(s)
    if max(cnt.values()) > (len(s) + 1) // 2:
        return ""
    heap = [(-freq, c) for c, freq in cnt.items()]
    heapq.heapify(heap)
    res = []
    prev = None
    while heap:
        freq, c = heapq.heappop(heap)
        res.append(c)
        if prev:
            heapq.heappush(heap, prev)
        prev = (freq + 1, c) if freq + 1 else None
    return ''.join(res)
```

Time: O(n log n) | Space: O(n)

---

## 5. Furthest Building You Can Reach

Climb buildings with bricks and ladders. Ladders for any height, bricks for limited. Min-heap of ladder jumps. When heap > ladders, use bricks for smallest jump. Greedy.

```python
import heapq
def furthestBuilding(heights, bricks, ladders):
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

Time: O(n log k) | Space: O(k)

---

## 6. Minimum Cost to Hire K Workers

Hire k workers. Pay = ratio * sum(quality). Ratio = wage/quality. Minimize total cost. Sort by ratio. For each as "captain", min-heap of quality for k workers with ratio <= captain. O(n log k).

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

## 7. IPO - Maximize Capital

Initial capital. Pick k projects (each has capital cost, profit). Maximize final capital. Sort projects by capital. Max-heap of profits for affordable projects. Pop k times.

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

## 8. K Closest Points to Origin (with distance)

Return k points closest to origin. Multiple solutions allowed. Max-heap of size k by distance. Or quickselect. O(n log k) or O(n) average.

```python
import heapq
def kClosest(points, k):
    heap = []
    for x, y in points:
        d = -(x*x + y*y)
        if len(heap) < k:
            heapq.heappush(heap, (d, x, y))
        elif d > heap[0][0]:
            heapq.heapreplace(heap, (d, x, y))
    return [[x, y] for _, x, y in heap]
```

Time: O(n log k) | Space: O(k)

---

## 9. Sort Characters by Frequency (with stability)

Sort by frequency, preserve order for same frequency. Count, bucket sort or heap with (freq, original_index) for stability.

```python
import heapq
from collections import Counter
def frequencySort(s):
    cnt = Counter(s)
    heap = [(-freq, c) for c, freq in cnt.items()]
    heapq.heapify(heap)
    return ''.join(c * (-freq) for freq, c in heap)
```

Time: O(n log n) | Space: O(n)

---

## 10. Find K Pairs with Smallest Sums

Two sorted arrays. Find k pairs (a,b) with smallest a+b. Min-heap (sum, i, j). Start (0,0). Expand (i+1,j) and (i,j+1). Avoid duplicates.

```python
import heapq
def kSmallestPairs(nums1, nums2, k):
    if not nums1 or not nums2:
        return []
    heap = [(nums1[0] + nums2[0], 0, 0)]
    seen = {(0, 0)}
    res = []
    while heap and len(res) < k:
        _, i, j = heapq.heappop(heap)
        res.append([nums1[i], nums2[j]])
        if i + 1 < len(nums1) and (i + 1, j) not in seen:
            seen.add((i + 1, j))
            heapq.heappush(heap, (nums1[i+1] + nums2[j], i + 1, j))
        if j + 1 < len(nums2) and (i, j + 1) not in seen:
            seen.add((i, j + 1))
            heapq.heappush(heap, (nums1[i] + nums2[j+1], i, j + 1))
    return res
```

Time: O(k log k) | Space: O(k)

---

## 11. Kth Smallest Element in Sorted Matrix

Matrix sorted row and column wise. Find kth smallest. Min-heap (value, row, col). Start (0,0). Pop k times, push (r+1,c) and (r,c+1).

```python
import heapq
def kthSmallest(matrix, k):
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

Time: O(k log k) | Space: O(k)

---

## 12. Sort Nearly Sorted Array (K-Sorted)

Each element at most k positions from sorted position. Min-heap of size k+1. Slide window. O(n log k).

```python
import heapq
def sortKSorted(arr, k):
    heap = arr[:k+1]
    heapq.heapify(heap)
    idx = 0
    for i in range(k + 1, len(arr)):
        arr[idx] = heapq.heappop(heap)
        heapq.heappush(heap, arr[i])
        idx += 1
    while heap:
        arr[idx] = heapq.heappop(heap)
        idx += 1
    return arr
```

Time: O(n log k) | Space: O(k)

---

## 13. Meeting Rooms II

Min meeting rooms needed for non-overlapping intervals. Sort by start. Min-heap of end times. If start >= min(end), reuse room. O(n log n).

```python
import heapq
def minMeetingRooms(intervals):
    intervals.sort(key=lambda x: x[0])
    heap = []
    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heappop(heap)
        heapq.heappush(heap, end)
    return len(heap)
```

Time: O(n log n) | Space: O(n)

---

## 14. Single-Threaded CPU

Tasks have enqueue time and processing time. Process in order of shortest processing when multiple available. Min-heap of (enqueue, processing, idx). Simulate time. Available tasks by processing time.

```python
import heapq
def getOrder(tasks):
    tasks = sorted([(e, p, i) for i, (e, p) in enumerate(tasks)])
    heap = []
    time = 0
    i = 0
    res = []
    while i < len(tasks) or heap:
        if not heap and i < len(tasks) and time < tasks[i][0]:
            time = tasks[i][0]
        while i < len(tasks) and tasks[i][0] <= time:
            heapq.heappush(heap, (tasks[i][1], tasks[i][2]))
            i += 1
        p, idx = heapq.heappop(heap)
        time += p
        res.append(idx)
    return res
```

Time: O(n log n) | Space: O(n)

---

## 15. Process Tasks Using Servers

Servers have weights. Assign tasks to servers. When multiple free, pick smallest weight. Two heaps: available (weight, idx), busy (free_time, weight, idx). Simulate.

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

## 16. Minimum Refueling Stops

Drive to target. Gas stations along the way. Min stops to refuel. Max-heap of fuel at passed stations. When out of fuel, refuel from largest. Greedy.

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

## 17. Smallest Range Covering Elements from K Lists

One element from each of k lists. Minimize range (max - min). Min-heap (val, list_id, idx). Track current max. Expand list with min. Update range.

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

## 18. Sliding Window Maximum

Max in each sliding window of size k. Monotonic deque (not heap). Heap alternative: max-heap with lazy deletion. O(n log n).

```python
import heapq
def maxSlidingWindow(nums, k):
    heap = [(-nums[i], i) for i in range(k)]
    heapq.heapify(heap)
    res = [-heap[0][0]]
    for i in range(k, len(nums)):
        heapq.heappush(heap, (-nums[i], i))
        while heap[0][1] <= i - k:
            heapq.heappop(heap)
        res.append(-heap[0][0])
    return res
```

Time: O(n log n) | Space: O(k)

---

## 19. Merge Intervals (with heap variant)

Merge overlapping intervals. Sort by start. Heap variant: min-heap by start, merge as we pop.

```python
def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    res = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= res[-1][1]:
            res[-1][1] = max(res[-1][1], end)
        else:
            res.append([start, end])
    return res
```

Time: O(n log n) | Space: O(n)

---

## 20. Network Delay Time

Single source shortest path in weighted graph. Dijkstra with min-heap. O((V+E) log V).

```python
import heapq
def networkDelayTime(times, n, k):
    graph = [[] for _ in range(n + 1)]
    for u, v, w in times:
        graph[u].append((v, w))
    dist = [float('inf')] * (n + 1)
    dist[k] = 0
    heap = [(0, k)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[v] > d + w:
                dist[v] = d + w
                heapq.heappush(heap, (dist[v], v))
    res = max(dist[1:])
    return res if res != float('inf') else -1
```

Time: O((V+E) log V) | Space: O(V)

---

## 21. Path With Maximum Minimum Value

Path from (0,0) to (n-1,m-1). Score = min value on path. Maximize score. Max-heap (min_so_far, r, c). Dijkstra-like expansion. Pick largest min_so_far first.

```python
import heapq
def maximumMinimumPath(grid):
    m, n = len(grid), len(grid[0])
    heap = [(-grid[0][0], 0, 0)]
    seen = {(0, 0)}
    while heap:
        score, r, c = heapq.heappop(heap)
        if r == m - 1 and c == n - 1:
            return -score
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in seen:
                seen.add((nr, nc))
                heapq.heappush(heap, (max(score, -grid[nr][nc]), nr, nc))
    return -1
```

Time: O(m*n log(m*n)) | Space: O(m*n)

---

## 22. Kth Largest Element in Array (Quickselect)

Same as easy but with O(n) average quickselect. Heap O(n log k) or quickselect O(n) average. Heap is simpler.

```python
import heapq
def findKthLargest(nums, k):
    return heapq.nlargest(k, nums)[-1]
```

Time: O(n log k) | Space: O(k)

---

## 23. Reorganize String (Optimized)

Same as above with O(n) bucket approach. Count. If max > (n+1)/2 impossible. Interleave most frequent with rest.

```python
from collections import Counter
def reorganizeString(s):
    cnt = Counter(s)
    if max(cnt.values()) > (len(s) + 1) // 2:
        return ""
    chars = sorted(cnt.items(), key=lambda x: -x[1])
    res = [''] * len(s)
    idx = 0
    for c, freq in chars:
        for _ in range(freq):
            res[idx] = c
            idx += 2
            if idx >= len(s):
                idx = 1
    return ''.join(res)
```

Time: O(n) | Space: O(n)

---

## 24. Design Twitter

Post tweet, follow, unfollow, get news feed (10 most recent from followees). K-way merge with heap. Each user's tweets. Merge k sorted lists by timestamp.

```python
import heapq
from collections import defaultdict
class Twitter:
    def __init__(self):
        self.time = 0
        self.tweets = defaultdict(list)
        self.following = defaultdict(set)

    def postTweet(self, userId, tweetId):
        self.tweets[userId].append((self.time, tweetId))
        self.time += 1

    def getNewsFeed(self, userId):
        heap = []
        self.following[userId].add(userId)
        for uid in self.following[userId]:
            if self.tweets[uid]:
                t, tid = self.tweets[uid][-1]
                heap.append((-t, tid, uid, len(self.tweets[uid]) - 1))
        heapq.heapify(heap)
        res = []
        while heap and len(res) < 10:
            _, tid, uid, idx = heapq.heappop(heap)
            res.append(tid)
            if idx > 0:
                t, tid = self.tweets[uid][idx - 1]
                heapq.heappush(heap, (-t, tid, uid, idx - 1))
        return res

    def follow(self, followerId, followeeId):
        self.following[followerId].add(followeeId)

    def unfollow(self, followerId, followeeId):
        self.following[followerId].discard(followeeId)
```

Time: O(log k) post, O(10 log k) feed | Space: O(n)

---

## 25. Ugly Number II

Nth number whose prime factors are only 2, 3, 5. Min-heap. Start 1. Pop, push 2x, 3x, 5x. Avoid duplicates with set.

```python
import heapq
def nthUglyNumber(n):
    heap = [1]
    seen = {1}
    for _ in range(n - 1):
        x = heapq.heappop(heap)
        for p in [2, 3, 5]:
            nx = x * p
            if nx not in seen:
                seen.add(nx)
                heapq.heappush(heap, nx)
    return heapq.heappop(heap)
```

Time: O(n log n) | Space: O(n)
