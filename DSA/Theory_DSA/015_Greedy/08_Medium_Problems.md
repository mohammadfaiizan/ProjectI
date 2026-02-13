# Medium Greedy Problems

## 1. Jump Game II

**Description**: Minimum jumps to reach last index.

**Approach**: BFS-like: at each step, extend to furthest reachable. Track jumps and current range.

```python
def jump(nums):
    jumps = end = farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == end:
            jumps += 1
            end = farthest
    return jumps
```

Time: O(n) | Space: O(1)

---

## 2. Merge Intervals

**Description**: Merge overlapping intervals.

**Approach**: Sort by start; merge consecutive overlapping intervals.

```python
def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    res = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= res[-1][1]:
            res[-1][1] = max(res[-1][1], e)
        else:
            res.append([s, e])
    return res
```

Time: O(n log n) | Space: O(1)

---

## 3. Insert Interval

**Description**: Insert new interval into sorted non-overlapping intervals and merge.

**Approach**: Find position; merge overlapping; insert.

```python
def insert(intervals, newInterval):
    res = []
    for s, e in intervals:
        if e < newInterval[0]:
            res.append([s, e])
        elif s > newInterval[1]:
            res.append(newInterval)
            newInterval = [s, e]
        else:
            newInterval = [min(s, newInterval[0]), max(e, newInterval[1])]
    res.append(newInterval)
    return res
```

Time: O(n) | Space: O(1)

---

## 4. Non-overlapping Intervals

**Description**: Minimum intervals to remove so rest are non-overlapping.

**Approach**: Same as max non-overlapping: sort by end, count non-overlapping. Answer = n - count.

```python
def eraseOverlapIntervals(intervals):
    intervals.sort(key=lambda x: x[1])
    end, count = float('-inf'), 0
    for s, e in intervals:
        if s >= end:
            end = e
            count += 1
    return len(intervals) - count
```

Time: O(n log n) | Space: O(1)

---

## 5. Task Scheduler

**Description**: Tasks with cooldown. Same task must be n apart. Min total time.

**Approach**: Schedule most frequent first; formula: (max_count-1)*(n+1) + num_max, or len(tasks).

```python
def leastInterval(tasks, n):
    from collections import Counter
    c = Counter(tasks)
    max_count = max(c.values())
    num_max = sum(1 for v in c.values() if v == max_count)
    return max(len(tasks), (max_count - 1) * (n + 1) + num_max)
```

Time: O(n) | Space: O(1)

---

## 6. Partition Labels

**Description**: Partition string so each letter in at most one part. Minimize number of parts.

**Approach**: Last index per char; extend partition until current index equals max last of partition chars.

```python
def partitionLabels(s):
    last = {c: i for i, c in enumerate(s)}
    start = end = 0
    res = []
    for i, c in enumerate(s):
        end = max(end, last[c])
        if i == end:
            res.append(end - start + 1)
            start = i + 1
    return res
```

Time: O(n) | Space: O(1)

---

## 7. Gas Station

**Description**: Circular route; gas at each station, cost to next. Find starting index to complete circuit.

**Approach**: If total gas >= total cost, solution exists. Start from 0; when tank negative, restart from next station.

```python
def canCompleteCircuit(gas, cost):
    total = tank = start = 0
    for i in range(len(gas)):
        total += gas[i] - cost[i]
        tank += gas[i] - cost[i]
        if tank < 0:
            start = i + 1
            tank = 0
    return start if total >= 0 else -1
```

Time: O(n) | Space: O(1)

---

## 8. Boats to Save People

**Description**: People with weights; boat limit; at most 2 per boat. Min boats.

**Approach**: Sort; pair heaviest with lightest if both fit; else heaviest alone.

```python
def numRescueBoats(people, limit):
    people.sort()
    i, j, boats = 0, len(people) - 1, 0
    while i <= j:
        boats += 1
        if people[i] + people[j] <= limit:
            i += 1
        j -= 1
    return boats
```

Time: O(n log n) | Space: O(1)

---

## 9. Bag of Tokens

**Description**: Play face-up (spend power, gain score) or face-down (gain power, lose score). Maximize score.

**Approach**: Buy cheapest tokens (face-up), sell most expensive (face-down) when needed. Two pointers.

```python
def bagOfTokensScore(tokens, power):
    tokens.sort()
    i, j, score, res = 0, len(tokens) - 1, 0, 0
    while i <= j:
        if power >= tokens[i]:
            power -= tokens[i]
            score += 1
            i += 1
            res = max(res, score)
        elif score > 0:
            power += tokens[j]
            score -= 1
            j -= 1
        else:
            break
    return res
```

Time: O(n log n) | Space: O(1)

---

## 10. Reorganize String

**Description**: Reorder so no two adjacent same. Return "" if impossible.

**Approach**: Possible iff max_freq <= (n+1)//2. Max-heap; alternate with most frequent.

```python
def reorganizeString(s):
    from collections import Counter
    import heapq
    c = Counter(s)
    if max(c.values()) > (len(s) + 1) // 2:
        return ""
    h = [(-v, k) for k, v in c.items()]
    heapq.heapify(h)
    res = []
    prev = None
    while h:
        v, k = heapq.heappop(h)
        res.append(k)
        if prev:
            heapq.heappush(h, prev)
        prev = (v + 1, k) if v + 1 < 0 else None
    return ''.join(res)
```

Time: O(n log n) | Space: O(n)

---

## 11. Remove K Digits

**Description**: Remove k digits from number string to get smallest possible number.

**Approach**: Monotonic stack: remove larger digits while k > 0. Keep result smallest.

```python
def removeKdigits(num, k):
    stack = []
    for d in num:
        while k and stack and stack[-1] > d:
            stack.pop()
            k -= 1
        stack.append(d)
    stack = stack[:-k] if k else stack
    return ''.join(stack).lstrip('0') or '0'
```

Time: O(n) | Space: O(n)

---

## 12. Queue Reconstruction by Height

**Description**: People (h, k) where k = number of people in front with height >= h. Reconstruct queue.

**Approach**: Sort by h descending, k ascending. Insert each at position k (greedy: taller first, then by k).

```python
def reconstructQueue(people):
    people.sort(key=lambda x: (-x[0], x[1]))
    res = []
    for p in people:
        res.insert(p[1], p)
    return res
```

Time: O(n^2) | Space: O(n)

---

## 13. Minimum Number of Arrows to Burst Balloons

**Description**: Intervals (balloons); arrow at x bursts all containing x. Min arrows.

**Approach**: Sort by end; count non-overlapping intervals (same as activity selection).

```python
def findMinArrowShots(points):
    points.sort(key=lambda x: x[1])
    end, count = float('-inf'), 0
    for s, e in points:
        if s > end:
            end = e
            count += 1
    return count
```

Time: O(n log n) | Space: O(1)

---

## 14. Meeting Rooms II

**Description**: Min rooms for all meetings.

**Approach**: Sweep line (start +1, end -1) or min-heap of end times.

```python
def minMeetingRooms(intervals):
    starts = sorted(i[0] for i in intervals)
    ends = sorted(i[1] for i in intervals)
    i = res = cur = 0
    for s in starts:
        while ends[i] <= s:
            cur -= 1
            i += 1
        cur += 1
        res = max(res, cur)
    return res
```

Time: O(n log n) | Space: O(n)

---

## 15. Car Pooling

**Description**: Trips (num_passengers, start, end). Capacity limit. Possible?

**Approach**: Sweep line; track passenger count; reject if exceeds capacity.

```python
def carPooling(trips, capacity):
    events = []
    for n, s, e in trips:
        events.append((s, n))
        events.append((e, -n))
    events.sort()
    cur = 0
    for _, delta in events:
        cur += delta
        if cur > capacity:
            return False
    return True
```

Time: O(n log n) | Space: O(n)

---

## 16. Maximum Swap

**Description**: Swap two digits once to maximize number.

**Approach**: Find rightmost smaller digit and swap with rightmost larger digit to its right.

```python
def maximumSwap(num):
    s = list(str(num))
    last = {int(c): i for i, c in enumerate(s)}
    for i, c in enumerate(s):
        for d in range(9, int(c), -1):
            if d in last and last[d] > i:
                s[i], s[last[d]] = s[last[d]], s[i]
                return int(''.join(s))
    return num
```

Time: O(n) | Space: O(n)

---

## 17. Wiggle Sort

**Description**: Reorder so nums[0] < nums[1] > nums[2] < nums[3] ...

**Approach**: Greedy swap: at odd index, ensure larger than neighbors; at even, ensure smaller.

```python
def wiggleSort(nums):
    for i in range(1, len(nums)):
        if (i % 2 and nums[i] < nums[i-1]) or (i % 2 == 0 and nums[i] > nums[i-1]):
            nums[i], nums[i-1] = nums[i-1], nums[i]
```

Time: O(n) | Space: O(1)

---

## 18. Largest Number

**Description**: Concatenate numbers to form largest possible number.

**Approach**: Custom sort: a before b if a+b > b+a (string comparison).

```python
def largestNumber(nums):
    from functools import cmp_to_key
    s = sorted(map(str, nums), key=cmp_to_key(lambda a, b: -1 if a+b > b+a else 1))
    return ''.join(s).lstrip('0') or '0'
```

Time: O(n log n) | Space: O(n)

---

## 19. Minimum Deletions to Make Character Frequencies Unique

**Description**: Delete min chars so no two chars have same frequency.

**Approach**: Sort frequencies descending; for each duplicate, reduce until unique (or 0).

```python
def minDeletions(s):
    from collections import Counter
    c = sorted(Counter(s).values(), reverse=True)
    seen, res = set(), 0
    for f in c:
        while f in seen and f > 0:
            f -= 1
            res += 1
        seen.add(f)
    return res
```

Time: O(n) | Space: O(1)

---

## 20. Reduce Array Size to Half

**Description**: Remove min number of distinct integers so remaining count <= half.

**Approach**: Greedy: remove most frequent first. Sort by frequency descending.

```python
def minSetSize(arr):
    from collections import Counter
    c = Counter(arr)
    freq = sorted(c.values(), reverse=True)
    n, removed, count = len(arr), 0, 0
    for f in freq:
        removed += f
        count += 1
        if removed >= n // 2:
            return count
    return count
```

Time: O(n log n) | Space: O(n)

---

## 21. Maximum Length of Pair Chain

**Description**: Pairs (a, b); chain if b_i < a_{i+1}. Max chain length.

**Approach**: Sort by end; activity selection (non-overlapping intervals).

```python
def findLongestChain(pairs):
    pairs.sort(key=lambda x: x[1])
    end, count = float('-inf'), 0
    for a, b in pairs:
        if a > end:
            end = b
            count += 1
    return count
```

Time: O(n log n) | Space: O(1)

---

## 22. Video Stitching

**Description**: Clips cover [0, time]. Min clips to cover.

**Approach**: Sort by start; greedy pick clip extending furthest at each step.

```python
def videoStitching(clips, time):
    clips.sort()
    end = res = i = 0
    while end < time:
        best = end
        while i < len(clips) and clips[i][0] <= end:
            best = max(best, clips[i][1])
            i += 1
        if best == end:
            return -1
        end = best
        res += 1
    return res
```

Time: O(n log n) | Space: O(1)

---

## 23. Minimum Taps to Open to Water a Garden

**Description**: Taps at positions with ranges. Min taps to cover [0, n].

**Approach**: Convert to intervals; greedy jump to furthest covering current position.

```python
def minTaps(n, ranges):
    intervals = [(max(0, i - r), min(n, i + r)) for i, r in enumerate(ranges)]
    intervals.sort()
    end = res = i = 0
    while end < n:
        best = end
        while i < len(intervals) and intervals[i][0] <= end:
            best = max(best, intervals[i][1])
            i += 1
        if best == end:
            return -1
        end = best
        res += 1
    return res
```

Time: O(n log n) | Space: O(n)

---

## 24. Broken Calculator

**Description**: Start from X, operations: multiply by 2 or subtract 1. Reach Y. Min operations.

**Approach**: Work backwards from Y: if Y > X, Y odd then add 1 else divide by 2. Greedy reverse.

```python
def brokenCalc(startValue, target):
    res = 0
    while target > startValue:
        res += 1
        target = target + 1 if target % 2 else target // 2
    return res + startValue - target
```

Time: O(log Y) | Space: O(1)

---

## 25. Score of Parentheses

**Description**: () = 1, AB = A+B, (A) = 2*A. Compute score.

**Approach**: Stack; on ( push 0; on ) pop and add 2*top or 1 to new top.

```python
def scoreOfParentheses(s):
    stack = [0]
    for c in s:
        if c == '(':
            stack.append(0)
        else:
            v = stack.pop()
            stack[-1] += max(2 * v, 1)
    return stack[0]
```

Time: O(n) | Space: O(n)

---

## 26. Advantage Shuffle

**Description**: Permute A to maximize number of positions where A[i] > B[i].

**Approach**: Sort both; for each B[i], use smallest A[j] > B[i] (binary search or two pointers).

```python
def advantageCount(A, B):
    import bisect
    A.sort()
    res = []
    for b in B:
        i = bisect.bisect_right(A, b)
        if i < len(A):
            res.append(A.pop(i))
        else:
            res.append(A.pop(0))
    return res
```

Time: O(n log n) | Space: O(n)

---

## 27. Minimum Operations to Reduce X to Zero

**Description**: Remove from left or right; sum removed = x. Min operations.

**Approach**: Equivalent to max subarray of sum = total - x. Sliding window or prefix sum.

```python
def minOperations(nums, x):
    target = sum(nums) - x
    if target < 0:
        return -1
    if target == 0:
        return len(nums)
    cur = res = left = 0
    for right in range(len(nums)):
        cur += nums[right]
        while cur > target:
            cur -= nums[left]
            left += 1
        if cur == target:
            res = max(res, right - left + 1)
    return len(nums) - res if res else -1
```

Time: O(n) | Space: O(1)

---

## 28. Dota2 Senate

**Description**: R and D vote; each bans next opponent. Who wins?

**Approach**: Queue for each party; simulate; each bans next opponent. Party with members left wins.

```python
def predictPartyVictory(senate):
    from collections import deque
    r = deque(i for i, c in enumerate(senate) if c == 'R')
    d = deque(i for i, c in enumerate(senate) if c == 'D')
    n = len(senate)
    while r and d:
        ri, di = r.popleft(), d.popleft()
        if ri < di:
            r.append(ri + n)
        else:
            d.append(di + n)
    return "Radiant" if r else "Dire"
```

Time: O(n) | Space: O(n)
