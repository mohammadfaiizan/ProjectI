# Greedy - Advanced Operations

## Prove Greedy Correctness (Exchange Argument Walkthrough)

Exchange argument: show that any optimal solution can be modified to include our greedy choice without worsening the result.

Example: Activity selection. Suppose optimal solution O has activity A' as first choice, and our greedy picks A (earliest finish). Since A finishes no later than A', we can replace A' with A in O. The rest of O remains valid (no overlap with A). So there exists an optimal solution starting with A.

```python
def exchange_argument_activity_selection(activities):
    activities.sort(key=lambda x: x[1])
    selected = []
    last_end = -float('inf')
    for start, end in activities:
        if start >= last_end:
            selected.append((start, end))
            last_end = end
    return selected
```

## Greedy vs DP Transition Examples

Some problems can be solved by both; the choice depends on problem structure.

**Interval scheduling**: Greedy by end time works (one choice: take or skip based on overlap).

**Weighted interval scheduling**: Cannot greedily take earliest-finish; need DP or DP with binary search (consider all choices: take or skip).

```python
import bisect

def weighted_interval_scheduling_dp(intervals):
    intervals.sort(key=lambda x: x[1])
    n = len(intervals)
    ends = [0] + [x[1] for x in intervals]
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        s, e, w = intervals[i - 1]
        j = bisect.bisect_right(ends, s) - 1
        dp[i] = max(dp[i - 1], dp[j] + w)
    return dp[n]
```

## Scheduling Problems Framework

1. **Single machine, no preemption**: Often sort by deadline or by some ratio
2. **Minimize maximum lateness**: Sort by deadline (Earliest Deadline First)
3. **Minimize completion time**: Shortest Job First
4. **Minimize weighted completion time**: Sort by weight/length ratio

```python
def earliest_deadline_first(jobs):
    jobs.sort(key=lambda x: x[1])
    time = 0
    max_lateness = 0
    for duration, deadline in jobs:
        time += duration
        max_lateness = max(max_lateness, time - deadline)
    return max_lateness
```

## Sweep Line with Greedy

Process events in order; at each event make a greedy decision. Common for interval overlap, room assignment.

```python
def sweep_line_meeting_rooms(intervals):
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
```

## Greedy with Sorting + Priority Queue Combo

Sort by one dimension; use heap to track the other. Example: meeting rooms II.

```python
import heapq

def min_meeting_rooms(intervals):
    intervals.sort(key=lambda x: x[0])
    heap = []
    for s, e in intervals:
        if heap and heap[0] <= s:
            heapq.heappop(heap)
        heapq.heappush(heap, e)
    return len(heap)
```

## Greedy with Two Pointers

Two indices advancing based on comparison. Merge intervals, container with most water.

```python
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged
```

## Greedy on Trees (Example)

Example: Minimum vertex cover on a tree. For each edge, at least one endpoint must be in the cover. Greedy: pick the parent when processing leaves (or use DP for optimal).

```python
def min_vertex_cover_tree_approx(graph, root):
    def dfs(u, parent):
        take, skip = 1, 0
        for v in graph[u]:
            if v == parent:
                continue
            c_take, c_skip = dfs(v, u)
            take += min(c_take, c_skip)
            skip += c_take
        return take, skip
    take, skip = dfs(root, -1)
    return min(take, skip)
```

## Greedy on Graphs (MST, Dijkstra Are Greedy)

**Kruskal**: Sort edges by weight; add if it does not create cycle (union-find). Greedy: pick minimum weight safe edge.

**Prim**: Start from a vertex; repeatedly add minimum weight edge to tree. Greedy: pick closest vertex not in tree.

**Dijkstra**: Relax edges from closest unvisited vertex. Greedy: pick vertex with minimum tentative distance.

```python
import heapq

def dijkstra(graph, start, n):
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist
```

## Anti-Greedy Arguments

When greedy fails, construct a counterexample:

1. **Coin change [1, 3, 4], target 6**: Greedy gives 3; optimal is 2
2. **0/1 Knapsack**: Greedy by value/weight can fill with low-value items first
3. **Graph coloring**: Greedy (color by order) can use more colors than optimal
4. **Set cover**: Greedy (pick set covering most uncovered) has approximation ratio ln(n) but not optimal

```python
def greedy_coin_fails():
    coins = [1, 3, 4]
    amount = 6
    coins.sort(reverse=True)
    count = 0
    rem = amount
    for c in coins:
        count += rem // c
        rem %= c
    return count
```
