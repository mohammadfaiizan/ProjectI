# Hard Greedy Problems

## 1. Candy

**Description**: Children in line; each gets at least 1 candy; if rating higher than neighbor, more candies. Min total candies.

**Approach**: Two passes: left-to-right (if rating up, candy = prev+1), right-to-left (same). Take max at each position.

```python
def candy(ratings):
    n = len(ratings)
    candies = [1] * n
    for i in range(1, n):
        if ratings[i] > ratings[i-1]:
            candies[i] = candies[i-1] + 1
    for i in range(n-2, -1, -1):
        if ratings[i] > ratings[i+1]:
            candies[i] = max(candies[i], candies[i+1] + 1)
    return sum(candies)
```

Time: O(n) | Space: O(n)

---

## 2. Trapping Rain Water II

**Description**: 2D elevation map; water trapped. 3D version of trapping rain water.

**Approach**: Min-heap from boundary; expand inward; water at cell = max(0, boundary_min - height).

```python
def trapRainWater(heightMap):
    import heapq
    if not heightMap:
        return 0
    m, n = len(heightMap), len(heightMap[0])
    visited = [[False] * n for _ in range(m)]
    h = []
    for i in range(m):
        for j in range(n):
            if i in (0, m-1) or j in (0, n-1):
                heapq.heappush(h, (heightMap[i][j], i, j))
                visited[i][j] = True
    res = 0
    while h:
        level, r, c = heapq.heappop(h)
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < m and 0 <= nc < n and not visited[nr][nc]:
                visited[nr][nc] = True
                res += max(0, level - heightMap[nr][nc])
                heapq.heappush(h, (max(level, heightMap[nr][nc]), nr, nc))
    return res
```

Time: O(m*n*log(m*n)) | Space: O(m*n)

---

## 3. Minimum Number of Refueling Stops

**Description**: Car with startFuel, target distance. Stations (position, fuel). Min stops to reach target.

**Approach**: Greedy: drive as far as possible; when out of fuel, refuel at station with most fuel passed. Max-heap of fuels.

```python
def minRefuelStops(target, startFuel, stations):
    import heapq
    pq, res, i, cur = [], 0, 0, startFuel
    while cur < target:
        while i < len(stations) and stations[i][0] <= cur:
            heapq.heappush(pq, -stations[i][1])
            i += 1
        if not pq:
            return -1
        cur += -heapq.heappop(pq)
        res += 1
    return res
```

Time: O(n log n) | Space: O(n)

---

## 4. IPO

**Description**: k projects; each has capital and profit. Start with w capital. Pick project only if capital <= w. Maximize capital after k projects.

**Approach**: Sort by capital; max-heap of profits for affordable projects. Each step pick max profit, add to capital.

```python
def findMaximizedCapital(k, w, profits, capital):
    import heapq
    projects = sorted(zip(capital, profits))
    i, pq = 0, []
    for _ in range(k):
        while i < len(projects) and projects[i][0] <= w:
            heapq.heappush(pq, -projects[i][1])
            i += 1
        if not pq:
            break
        w += -heapq.heappop(pq)
    return w
```

Time: O(n log n) | Space: O(n)

---

## 5. Maximum Performance of a Team

**Description**: Engineers with speed and efficiency. Pick at most k; performance = sum(speeds) * min(efficiency). Maximize.

**Approach**: Sort by efficiency descending; for each as min efficiency, take top k speeds (min-heap of size k).

```python
def maxPerformance(n, speed, efficiency, k):
    import heapq
    eng = sorted(zip(efficiency, speed), reverse=True)
    h, total, res = [], 0, 0
    for e, s in eng:
        heapq.heappush(h, s)
        total += s
        if len(h) > k:
            total -= heapq.heappop(h)
        res = max(res, total * e)
    return res % (10**9 + 7)
```

Time: O(n log n) | Space: O(k)

---

## 6. Merge k Sorted Lists

**Description**: Merge k sorted linked lists.

**Approach**: Min-heap of (val, list_node); pop min, push next from same list.

```python
def mergeKLists(lists):
    import heapq
    h = [(l.val, i, l) for i, l in enumerate(lists) if l]
    heapq.heapify(h)
    dummy = cur = ListNode(0)
    while h:
        val, i, node = heapq.heappop(h)
        cur.next = node
        cur = cur.next
        if node.next:
            heapq.heappush(h, (node.next.val, i, node.next))
    return dummy.next
```

Time: O(n log k) | Space: O(k)

---

## 7. Minimum Cost to Hire K Workers

**Description**: Workers have quality and wage. Hire k; pay each at least their wage. Total wage = ratio * sum(quality). Minimize.

**Approach**: Sort by wage/quality; for each as "captain", take k-1 workers with smallest quality from those with ratio <= captain. Min total = sum(quality) * captain_ratio. Use heap to maintain k smallest quality.

```python
def mincostToHireWorkers(quality, wage, k):
    import heapq
    workers = sorted((w/q, q) for q, w in zip(quality, wage))
    h, total_q, res = [], 0, float('inf')
    for r, q in workers:
        heapq.heappush(h, -q)
        total_q += q
        if len(h) > k:
            total_q += heapq.heappop(h)
        if len(h) == k:
            res = min(res, total_q * r)
    return res
```

Time: O(n log n) | Space: O(k)

---

## 8. Maximum Frequency Stack

**Description**: Push, pop. Pop returns most frequent element; tie-break by most recent.

**Approach**: Map freq to stack of elements. Track max_freq. Pop from max_freq stack.

```python
class FreqStack:
    def __init__(self):
        self.freq = {}
        self.group = {}
        self.max_freq = 0
    def push(self, val):
        self.freq[val] = self.freq.get(val, 0) + 1
        f = self.freq[val]
        self.group.setdefault(f, []).append(val)
        self.max_freq = max(self.max_freq, f)
    def pop(self):
        val = self.group[self.max_freq].pop()
        self.freq[val] -= 1
        if not self.group[self.max_freq]:
            self.max_freq -= 1
        return val
```

Time: O(1) per op | Space: O(n)

---

## 9. Reconstruct Itinerary

**Description**: Tickets (from, to). Reconstruct itinerary using all tickets. Lexicographically smallest.

**Approach**: Euler path. Sort adjacency lists; DFS from JFK, backtrack and reverse path.

```python
def findItinerary(tickets):
    from collections import defaultdict
    g = defaultdict(list)
    for a, b in tickets:
        g[a].append(b)
    for k in g:
        g[k].sort(reverse=True)
    res = []
    def dfs(node):
        while g[node]:
            dfs(g[node].pop())
        res.append(node)
    dfs('JFK')
    return res[::-1]
```

Time: O(n log n) | Space: O(n)

---

## 10. Minimum Window Substring

**Description**: Smallest substring of s containing all chars of t.

**Approach**: Sliding window; expand until valid, contract from left. Track char counts.

```python
def minWindow(s, t):
    from collections import Counter
    need, have = Counter(t), 0
    required = len(need)
    l, res = 0, (0, float('inf'))
    for r, c in enumerate(s):
        if c in need:
            need[c] -= 1
            if need[c] == 0:
                have += 1
        while have == required:
            if r - l < res[1] - res[0]:
                res = (l, r)
            if s[l] in need:
                need[s[l]] += 1
                if need[s[l]] > 0:
                    have -= 1
            l += 1
    return s[res[0]:res[1]+1] if res[1] < float('inf') else ""
```

Time: O(n) | Space: O(1)

---

## 11. Maximum Events That Can Be Attended

**Description**: Events [start, end]; one event per day. Max events.

**Approach**: Sort by start; for each day, attend event with earliest end that covers that day. Min-heap of end times.

```python
def maxEvents(events):
    import heapq
    events.sort()
    h, res, i, n = [], 0, 0, len(events)
    for d in range(1, 100001):
        while i < n and events[i][0] == d:
            heapq.heappush(h, events[i][1])
            i += 1
        while h and h[0] < d:
            heapq.heappop(h)
        if h:
            heapq.heappop(h)
            res += 1
    return res
```

Time: O(n log n) | Space: O(n)

---

## 12. Employee Free Time

**Description**: Sorted intervals for each employee. Find common free time.

**Approach**: Merge all intervals; gaps between merged intervals are free time.

```python
def employeeFreeTime(schedule):
    intervals = sorted([i for emp in schedule for i in emp])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [[merged[i][1], merged[i+1][0]] for i in range(len(merged)-1)]
```

Time: O(n log n) | Space: O(n)

---

## 13. Minimum Interval to Include Each Query

**Description**: Intervals and queries. For each query, smallest interval containing it, or -1.

**Approach**: Sweep queries; maintain min-heap of (size, end) for intervals covering current query. Pop expired.

```python
def minInterval(intervals, queries):
    import heapq
    intervals.sort()
    qs = sorted(enumerate(queries), key=lambda x: x[1])
    h, res, i = [], [0]*len(queries), 0
    for idx, q in qs:
        while i < len(intervals) and intervals[i][0] <= q:
            heapq.heappush(h, (intervals[i][1]-intervals[i][0]+1, intervals[i][1]))
            i += 1
        while h and h[0][1] < q:
            heapq.heappop(h)
        res[idx] = h[0][0] if h else -1
    return res
```

Time: O(n log n + q log q) | Space: O(n)

---

## 14. Maximum Profit in Job Scheduling

**Description**: Jobs with start, end, profit. No overlap. Max profit.

**Approach**: DP + binary search. Sort by end; dp[i] = max(profit[i] + dp[j], dp[i-1]) where j = latest non-overlapping.

```python
def jobScheduling(startTime, endTime, profit):
    import bisect
    jobs = sorted(zip(endTime, startTime, profit))
    dp = [[0, 0]]
    for e, s, p in jobs:
        i = bisect.bisect_right(dp, [s+1]) - 1
        if dp[i][1] + p > dp[-1][1]:
            dp.append([e, dp[i][1] + p])
    return dp[-1][1]
```

Time: O(n log n) | Space: O(n)

---

## 15. Course Schedule III

**Description**: Courses (duration, lastDay). Take at most one at a time. Max courses.

**Approach**: Sort by lastDay; greedy take by deadline. If total time exceeds lastDay of current, remove longest duration course (max-heap).

```python
def scheduleCourse(courses):
    import heapq
    courses.sort(key=lambda x: x[1])
    h, time = [], 0
    for d, e in courses:
        if time + d <= e:
            heapq.heappush(h, -d)
            time += d
        elif h and -h[0] > d:
            time += d + heapq.heappop(h)
            heapq.heappush(h, -d)
    return len(h)
```

Time: O(n log n) | Space: O(n)

---

## 16. Patching Array

**Description**: Sorted nums and n. Add min numbers so every [1, n] can be formed as sum of (nums + added).

**Approach**: Track max formable; while max < n, if next num <= max+1, add it; else add max+1. Greedy patch.

```python
def minPatches(nums, n):
    patch, i, res = 0, 0, 0
    while patch < n:
        if i < len(nums) and nums[i] <= patch + 1:
            patch += nums[i]
            i += 1
        else:
            patch += patch + 1
            res += 1
    return res
```

Time: O(n) | Space: O(1)

---

## 17. Create Maximum Number

**Description**: Two arrays; create length k number by picking digits from both (preserving order). Maximize.

**Approach**: Try all splits (i from first, k-i from second). For each array, get max subsequence of given length (monotonic stack). Merge two max subsequences (greedy merge). Take best.

```python
def maxNumber(nums1, nums2, k):
    def max_subseq(nums, k):
        drop = len(nums) - k
        stack = []
        for x in nums:
            while drop and stack and stack[-1] < x:
                stack.pop()
                drop -= 1
            stack.append(x)
        return stack[:k]
    def merge(a, b):
        res = []
        while a or b:
            bigger = a if a > b else b
            res.append(bigger.pop(0))
        return res
    best = []
    for i in range(max(0, k-len(nums2)), min(k, len(nums1))+1):
        cur = merge(max_subseq(nums1, i), max_subseq(nums2, k-i))
        best = max(best, cur)
    return best
```

Time: O(k * (m + n)) | Space: O(k)

---

## 18. Remove Duplicate Letters

**Description**: Remove duplicates so result is lexicographically smallest.

**Approach**: Monotonic stack; pop if char has more occurrences later and top > current. Track last index per char.

```python
def removeDuplicateLetters(s):
    last = {c: i for i, c in enumerate(s)}
    stack, seen = [], set()
    for i, c in enumerate(s):
        if c in seen:
            continue
        while stack and stack[-1] > c and last[stack[-1]] > i:
            seen.discard(stack.pop())
        stack.append(c)
        seen.add(c)
    return ''.join(stack)
```

Time: O(n) | Space: O(1)
