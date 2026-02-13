# Greedy - Fractional Knapsack and Variants

## Fractional Knapsack (Theory and Proof)

We have n items with weights and values; knapsack capacity W. We can take fractions of items. Goal: maximize total value.

**Greedy**: Sort by value/weight descending. Take items in that order, fully if possible, else fraction to fill capacity.

**Proof**: Suppose optimal solution O differs from greedy G. Let i be first item where they differ. G takes more of i than O. Then O must have taken more of some item j with lower ratio. Exchange some of j for i: we get more value (since ratio_i > ratio_j). So O was not optimal. Contradiction.

```python
def fractional_knapsack(weights, values, capacity):
    items = [(v / w, w, v) for v, w in zip(values, weights)]
    items.sort(key=lambda x: x[0], reverse=True)
    total = 0
    for ratio, w, v in items:
        if capacity <= 0:
            break
        take = min(w, capacity)
        total += take * ratio
        capacity -= take
    return total
```

## Job Sequencing with Deadlines (Maximize Profit)

n jobs, each with deadline and profit. One job per time slot. Schedule to maximize profit. Sort by profit descending; for each job assign to latest available slot before deadline.

```python
def job_sequencing(jobs):
    jobs.sort(key=lambda x: x[2], reverse=True)
    max_deadline = max(j[1] for j in jobs)
    slots = [False] * (max_deadline + 1)
    profit = 0
    for _, deadline, p in jobs:
        for t in range(min(deadline, max_deadline), 0, -1):
            if not slots[t]:
                slots[t] = True
                profit += p
                break
    return profit
```

## Weighted Job Scheduling Overview

Jobs with start, end, profit. Maximize profit with no overlapping. Not greedy; use DP with binary search. Sort by end; dp[i] = max(profit[i] + dp[j], dp[i-1]) where j is latest non-overlapping job.

## Minimum Platforms / Meeting Rooms (Sweep Line)

Same as meeting rooms II: min platforms needed so no train waits. Sweep line on arrival/departure events.

```python
def min_platforms(arrival, departure):
    events = [(t, 1) for t in arrival] + [(t, -1) for t in departure]
    events.sort(key=lambda x: (x[0], x[1]))
    count = 0
    max_count = 0
    for _, delta in events:
        count += delta
        max_count = max(max_count, count)
    return max_count
```

## Assign Cookies

Assign smallest cookie satisfying each child. Sort both; two pointers.

```python
def find_content_children(g, s):
    g.sort()
    s.sort()
    i = j = 0
    while i < len(g) and j < len(s):
        if s[j] >= g[i]:
            i += 1
        j += 1
    return i
```

## Lemonade Change

Customers pay 5, 10, or 20. Must give change. Greedy: always give change with largest bills first (one 5 for 10, one 10+5 or three 5s for 20).

```python
def lemonade_change(bills):
    five = ten = 0
    for b in bills:
        if b == 5:
            five += 1
        elif b == 10:
            if five == 0:
                return False
            five -= 1
            ten += 1
        else:
            if ten > 0 and five > 0:
                ten -= 1
                five -= 1
            elif five >= 3:
                five -= 3
            else:
                return False
    return True
```

## Boats to Save People

People with weights; boat limit. Each boat at most 2 people. Min boats. Sort; pair heaviest with lightest if both fit.

```python
def num_rescue_boats(people, limit):
    people.sort()
    i, j = 0, len(people) - 1
    boats = 0
    while i <= j:
        if people[i] + people[j] <= limit:
            i += 1
        j -= 1
        boats += 1
    return boats
```

## Bag of Tokens

Tokens have power. Play face-up: spend power, gain 1 score. Play face-down: gain power, lose 1 score. Maximize score. Greedy: buy lowest power tokens (face-up), sell highest power (face-down) when needed.

```python
def bag_of_tokens_score(tokens, power):
    tokens.sort()
    i, j = 0, len(tokens) - 1
    score = 0
    max_score = 0
    while i <= j:
        if power >= tokens[i]:
            power -= tokens[i]
            score += 1
            i += 1
            max_score = max(max_score, score)
        elif score > 0:
            power += tokens[j]
            score -= 1
            j -= 1
        else:
            break
    return max_score
```

## Two City Scheduling

2n people; cost to send to A or B. Send exactly n to each. Minimize total cost. Sort by (costA - costB); first n go to A, rest to B.

```python
def two_city_scheduling(costs):
    costs.sort(key=lambda x: x[0] - x[1])
    n = len(costs) // 2
    return sum(c[0] for c in costs[:n]) + sum(c[1] for c in costs[n:])
```

## Minimum Cost to Move Chips

Chips at positions. Move chip 2 positions cost 0, 1 position cost 1. Move all to same position. All chips at even can move to one even at 0 cost; same for odd. Answer is min(count_odd, count_even).

```python
def min_cost_to_move_chips(position):
    odd = sum(1 for p in position if p % 2 == 1)
    even = len(position) - odd
    return min(odd, even)
```

## Largest Values From Labels

Items have value and label. Use at most num_wanted items and at most use_limit per label. Greedy: sort by value descending; take while constraints hold.

```python
def largest_vals_from_labels(values, labels, num_wanted, use_limit):
    items = sorted(zip(values, labels), reverse=True)
    label_count = {}
    total = 0
    count = 0
    for v, l in items:
        if count >= num_wanted:
            break
        if label_count.get(l, 0) < use_limit:
            total += v
            label_count[l] = label_count.get(l, 0) + 1
            count += 1
    return total
```

## Reduce Array Size to Half

Remove min number of distinct integers so remaining count <= half. Greedy: remove most frequent first.

```python
from collections import Counter

def min_set_size(arr):
    n = len(arr)
    freq = Counter(arr)
    counts = sorted(freq.values(), reverse=True)
    removed = 0
    size = 0
    for c in counts:
        removed += c
        size += 1
        if removed >= n // 2:
            return size
    return size
```

## Task Scheduler

Tasks with cooldown n. Same task must be n apart. Min total time. Greedy: schedule most frequent task first, fill gaps with others.

```python
from collections import Counter

def least_interval(tasks, n):
    counts = Counter(tasks)
    max_count = max(counts.values())
    num_max = sum(1 for c in counts.values() if c == max_count)
    return max(len(tasks), (max_count - 1) * (n + 1) + num_max)
```

## Reorganize String (No Two Adjacent Same)

Reorder so no two adjacent chars same. Possible iff max_freq <= (n+1)//2. Greedy: place most frequent in even indices first.

```python
from collections import Counter
import heapq

def reorganize_string(s):
    counts = Counter(s)
    if max(counts.values()) > (len(s) + 1) // 2:
        return ""
    heap = [(-c, ch) for ch, c in counts.items()]
    heapq.heapify(heap)
    result = []
    prev = None
    while heap:
        neg_c, ch = heapq.heappop(heap)
        result.append(ch)
        if prev and prev[0] < -1:
            heapq.heappush(heap, (prev[0] + 1, prev[1]))
        prev = (neg_c + 1, ch) if neg_c + 1 < 0 else None
    return "".join(result)
```

## Distant Barcodes

Same as reorganize string: rearrange barcodes so no two adjacent same. Use max-heap of (count, value); alternate with most frequent.

```python
from collections import Counter
import heapq

def rearrange_barcodes(barcodes):
    counts = Counter(barcodes)
    heap = [(-c, v) for v, c in counts.items()]
    heapq.heapify(heap)
    result = []
    prev = None
    while heap:
        neg_c, v = heapq.heappop(heap)
        result.append(v)
        if prev and prev[0] < -1:
            heapq.heappush(heap, (prev[0] + 1, prev[1]))
        prev = (neg_c + 1, v) if neg_c + 1 < 0 else None
    return result
```
