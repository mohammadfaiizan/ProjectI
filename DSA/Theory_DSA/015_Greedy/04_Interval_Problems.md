# Greedy - Interval Problems

## Activity Selection / Max Non-Overlapping Intervals

Sort by end time; pick each interval that does not overlap with the last chosen.

```python
def max_non_overlapping_intervals(intervals):
    intervals.sort(key=lambda x: x[1])
    count = 0
    last_end = -float('inf')
    for s, e in intervals:
        if s >= last_end:
            count += 1
            last_end = e
    return count
```

## Meeting Rooms I (Can Attend All?)

Check if any two intervals overlap. Sort by start; compare consecutive intervals.

```python
def can_attend_meetings(intervals):
    intervals.sort(key=lambda x: x[0])
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i - 1][1]:
            return False
    return True
```

## Meeting Rooms II (Min Rooms - Sweep Line or Min-Heap)

Count maximum overlapping intervals at any time. Sweep line: events (start, +1), (end, -1).

```python
def min_meeting_rooms_sweep(intervals):
    events = []
    for s, e in intervals:
        events.append((s, 1))
        events.append((e, -1))
    events.sort(key=lambda x: (x[0], x[1]))
    count = 0
    max_count = 0
    for _, delta in events:
        count += delta
        max_count = max(max_count, count)
    return max_count

def min_meeting_rooms_heap(intervals):
    intervals.sort(key=lambda x: x[0])
    import heapq
    heap = []
    for s, e in intervals:
        if heap and heap[0] <= s:
            heapq.heappop(heap)
        heapq.heappush(heap, e)
    return len(heap)
```

## Merge Intervals

Sort by start; merge overlapping consecutive intervals.

```python
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0][:]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged
```

## Insert Interval

Insert new interval and merge. Binary search for position or linear scan.

```python
def insert_interval(intervals, new_interval):
    result = []
    i = 0
    while i < len(intervals) and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1
    while i < len(intervals) and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    result.append(new_interval)
    result.extend(intervals[i:])
    return result
```

## Non-Overlapping Intervals (Min Removals)

Min intervals to remove so rest are non-overlapping. Same as max non-overlapping: sort by end, count non-overlapping.

```python
def erase_overlap_intervals(intervals):
    intervals.sort(key=lambda x: x[1])
    count = 0
    last_end = -float('inf')
    for s, e in intervals:
        if s >= last_end:
            count += 1
            last_end = e
    return len(intervals) - count
```

## Minimum Arrows to Burst Balloons

Intervals are balloons; arrow at x bursts all balloons containing x. Min arrows = max non-overlapping intervals (sort by end).

```python
def find_min_arrow_shots(points):
    points.sort(key=lambda x: x[1])
    count = 0
    last_end = -float('inf')
    for s, e in points:
        if s > last_end:
            count += 1
            last_end = e
    return count
```

## Interval List Intersections

Given two lists of disjoint sorted intervals, find all intersections.

```python
def interval_intersection(first, second):
    i = j = 0
    result = []
    while i < len(first) and j < len(second):
        lo = max(first[i][0], second[j][0])
        hi = min(first[i][1], second[j][1])
        if lo <= hi:
            result.append([lo, hi])
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return result
```

## Remove Covered Intervals

Remove intervals that are completely covered by another. Sort by start ascending, end descending; count non-covered.

```python
def remove_covered_intervals(intervals):
    intervals.sort(key=lambda x: (x[0], -x[1]))
    count = 0
    max_end = -1
    for _, e in intervals:
        if e > max_end:
            count += 1
            max_end = e
    return count
```

## My Calendar I

Book event if no double booking. Maintain sorted list; check overlap before insert.

```python
def __init__(self):
    self.events = []

def book(self, start, end):
    for s, e in self.events:
        if start < e and end > s:
            return False
    self.events.append((start, end))
    self.events.sort()
    return True
```

## My Calendar II

Allow double booking but not triple. Track single and double bookings.

```python
def __init__(self):
    self.single = []
    self.double = []

def book(self, start, end):
    for s, e in self.double:
        if start < e and end > s:
            return False
    for s, e in self.single:
        if start < e and end > s:
            self.double.append((max(start, s), min(end, e)))
    self.single.append((start, end))
    return True
```

## My Calendar III

Return max number of overlapping events. Sweep line.

```python
def __init__(self):
    self.events = []

def book(self, start, end):
    self.events.append((start, 1))
    self.events.append((end, -1))
    self.events.sort(key=lambda x: (x[0], x[1]))
    count = 0
    max_count = 0
    for _, delta in self.events:
        count += delta
        max_count = max(max_count, count)
    return max_count
```

## Employee Free Time

Merge all intervals, find gaps between merged intervals.

```python
def employee_free_time(schedule):
    intervals = []
    for emp in schedule:
        intervals.extend(emp)
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0][:]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [[merged[i][1], merged[i + 1][0]] for i in range(len(merged) - 1)]
```

## Car Pooling

Track capacity at each pickup/drop. Sweep line with capacity changes.

```python
def car_pooling(trips, capacity):
    events = []
    for n, s, e in trips:
        events.append((s, n))
        events.append((e, -n))
    events.sort(key=lambda x: (x[0], x[1]))
    count = 0
    for _, delta in events:
        count += delta
        if count > capacity:
            return False
    return True
```

## Video Stitching

Cover [0, time] with minimum clips. Sort by start; greedy pick clip that extends furthest.

```python
def video_stitching(clips, time):
    clips.sort(key=lambda x: x[0])
    count = 0
    end = 0
    i = 0
    while end < time:
        max_end = end
        while i < len(clips) and clips[i][0] <= end:
            max_end = max(max_end, clips[i][1])
            i += 1
        if max_end == end:
            return -1
        end = max_end
        count += 1
    return count
```

## Minimum Taps to Water Garden

n+1 positions 0..n; each tap at i waters [i-ranges[i], i+ranges[i]]. Min taps to cover [0, n]. Greedy: at each position, pick tap that extends furthest right.

```python
def min_taps(n, ranges):
    intervals = [(max(0, i - r), min(n, i + r)) for i, r in enumerate(ranges)]
    intervals.sort(key=lambda x: x[0])
    count = 0
    end = 0
    i = 0
    while end < n:
        max_end = end
        while i < len(intervals) and intervals[i][0] <= end:
            max_end = max(max_end, intervals[i][1])
            i += 1
        if max_end == end:
            return -1
        end = max_end
        count += 1
    return count
```

## Maximum Events That Can Be Attended

Events have [start, end]; attend at most one per day. Greedy: for each day d, attend the event with earliest end that has start <= d <= end.

```python
import heapq

def max_events(events):
    events.sort(key=lambda x: x[0])
    heap = []
    i = 0
    count = 0
    for d in range(1, 100001):
        while i < len(events) and events[i][0] == d:
            heapq.heappush(heap, events[i][1])
            i += 1
        while heap and heap[0] < d:
            heapq.heappop(heap)
        if heap:
            heapq.heappop(heap)
            count += 1
    return count
```

## Maximum Profit in Job Scheduling (DP + Binary Search)

Weighted intervals; max profit with no overlap. Sort by end; DP[i] = max(DP[i-1], profit[i] + DP[j]) where j is latest non-overlapping job.

```python
import bisect

def job_scheduling(start_time, end_time, profit):
    jobs = sorted(zip(start_time, end_time, profit), key=lambda x: x[1])
    n = len(jobs)
    dp = [0] * (n + 1)
    ends = [0] + [j[1] for j in jobs]
    for i in range(1, n + 1):
        s, e, p = jobs[i - 1]
        j = bisect.bisect_right(ends, s) - 1
        dp[i] = max(dp[i - 1], dp[j] + p)
    return dp[n]
```

## Minimum Interval to Include Each Query

For each query point, find smallest interval (by size) that contains it. Sweep: sort events (interval start, end, size) and queries; use min-heap of (size, end) for active intervals.

```python
import heapq

def min_interval_to_include_queries(intervals, queries):
    intervals_sorted = sorted([(lo, hi, hi - lo + 1) for lo, hi in intervals], key=lambda x: x[0])
    q_indexed = sorted((q, i) for i, q in enumerate(queries))
    result = [-1] * len(queries)
    heap = []
    j = 0
    for q, idx in q_indexed:
        while j < len(intervals_sorted) and intervals_sorted[j][0] <= q:
            lo, hi, size = intervals_sorted[j]
            heapq.heappush(heap, (size, hi))
            j += 1
        while heap and heap[0][1] < q:
            heapq.heappop(heap)
        if heap:
            result[idx] = heap[0][0]
    return result
```
