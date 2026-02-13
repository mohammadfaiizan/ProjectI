# Binary Search on Answer

## Concept

**Idea:** The answer lies in a range [min, max]. Instead of searching the array, we binary search on the answer space. For each candidate value mid, check if it is feasible. If feasible, try smaller (for minimization) or larger (for maximization). The feasibility function is the key.

**Steps:**
1. Identify the answer space (min possible to max possible)
2. Define feasibility check: can we achieve the goal with value x?
3. Binary search: if feasible(mid), we have a valid answer; narrow the range
4. Return the optimal value

**When to use:**
- "Minimum x such that condition holds"
- "Maximum x such that condition holds"
- Optimization over a continuous or discrete range

---

## Koko Eating Bananas

**Problem:** Piles of bananas, h hours. Koko eats k bananas per hour from one pile. Find minimum k to finish all in h hours.

**Approach:** Answer space [1, max(piles)]. Feasible(k): sum(ceil(pile/k)) <= h. Binary search for minimum k.

```python
import math

def min_eating_speed(piles, h):
    def feasible(k):
        return sum(math.ceil(p / k) for p in piles) <= h

    left, right = 1, max(piles)
    while left < right:
        mid = (left + right) // 2
        if feasible(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

---

## Minimum Days to Make M Bouquets

**Problem:** Bloom day for each flower. Need m bouquets of k adjacent flowers. Find minimum days to wait.

**Approach:** Answer space [1, max(bloomDay)]. Feasible(d): count adjacent groups of size >= k in bloomed flowers (bloomDay <= d). Binary search for minimum d.

```python
def min_days(bloomDay, m, k):
    if m * k > len(bloomDay):
        return -1

    def feasible(d):
        bouquets = 0
        adjacent = 0
        for b in bloomDay:
            if b <= d:
                adjacent += 1
                if adjacent == k:
                    bouquets += 1
                    adjacent = 0
            else:
                adjacent = 0
        return bouquets >= m

    left, right = min(bloomDay), max(bloomDay)
    while left < right:
        mid = (left + right) // 2
        if feasible(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

---

## Capacity to Ship Packages Within D Days

**Problem:** Weights of packages, ship in order. Find minimum capacity so all shipped within d days.

**Approach:** Answer space [max(weights), sum(weights)]. Feasible(cap): greedy load, count days needed. Binary search for minimum cap.

```python
def ship_within_days(weights, days):
    def feasible(cap):
        d = 1
        curr = 0
        for w in weights:
            if curr + w > cap:
                d += 1
                curr = w
            else:
                curr += w
        return d <= days

    left, right = max(weights), sum(weights)
    while left < right:
        mid = (left + right) // 2
        if feasible(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

---

## Split Array Largest Sum

**Problem:** Split array into k subarrays. Minimize the largest sum among subarrays.

**Approach:** Answer space [max(nums), sum(nums)]. Feasible(s): greedy split, count subarrays with sum <= s. Binary search for minimum s.

```python
def split_array(nums, k):
    def feasible(s):
        subarrays = 1
        curr = 0
        for n in nums:
            if curr + n > s:
                subarrays += 1
                curr = n
            else:
                curr += n
        return subarrays <= k

    left, right = max(nums), sum(nums)
    while left < right:
        mid = (left + right) // 2
        if feasible(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

---

## Magnetic Force (Aggressive Cows)

**Problem:** Positions of baskets, place m cows. Maximize minimum distance between any two cows.

**Approach:** Answer space [1, max(pos)-min(pos)]. Feasible(d): greedy place cows with distance >= d. Binary search for maximum d.

```python
def max_distance(position, m):
    position.sort()

    def feasible(d):
        count = 1
        last = position[0]
        for p in position[1:]:
            if p - last >= d:
                count += 1
                last = p
        return count >= m

    left, right = 1, position[-1] - position[0]
    while left < right:
        mid = (left + right + 1) // 2
        if feasible(mid):
            left = mid
        else:
            right = mid - 1
    return left
```

---

## Allocate Minimum Pages (Book Allocation)

**Problem:** Pages per book, m students. Each gets contiguous books. Minimize maximum pages per student.

**Approach:** Same as split array largest sum. Answer space [max(pages), sum(pages)]. Feasible(max_pages): greedy allocation.

```python
def allocate_books(pages, m):
    if m > len(pages):
        return -1

    def feasible(max_pages):
        students = 1
        curr = 0
        for p in pages:
            if curr + p > max_pages:
                students += 1
                curr = p
            else:
                curr += p
        return students <= m

    left, right = max(pages), sum(pages)
    while left < right:
        mid = (left + right) // 2
        if feasible(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

---

## Painter's Partition

**Problem:** Boards with lengths, k painters. Each paints contiguous boards. Minimize time (sum of lengths) for slowest painter.

**Approach:** Same as book allocation. Answer space [max(boards), sum(boards)].

```python
def painter_partition(boards, k):
    def feasible(max_time):
        painters = 1
        curr = 0
        for b in boards:
            if curr + b > max_time:
                painters += 1
                curr = b
            else:
                curr += b
        return painters <= k

    left, right = max(boards), sum(boards)
    while left < right:
        mid = (left + right) // 2
        if feasible(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

---

## Minimize Max Distance to Gas Station

**Problem:** Positions of gas stations on a line, add k new stations. Minimize maximum distance between adjacent stations.

**Approach:** Answer space [0, max_gap]. Feasible(d): for each gap, stations needed = ceil(gap/d) - 1. Total <= k.

```python
import math

def minmax_gas_distance(stations, k):
    gaps = [stations[i+1] - stations[i] for i in range(len(stations)-1)]
    max_gap = max(gaps)

    def feasible(d):
        if d == 0:
            return False
        total = sum(math.ceil(g / d) - 1 for g in gaps)
        return total <= k

    left, right = 0, max_gap
    for _ in range(100):
        mid = (left + right) / 2
        if feasible(mid):
            right = mid
        else:
            left = mid
    return left
```

---

## Nth Magical Number

**Problem:** Find nth positive integer divisible by a or b.

**Approach:** Answer space [1, n*min(a,b)]. Feasible(x): count = x//a + x//b - x//lcm(a,b). Binary search for smallest x with count >= n.

```python
import math

def nth_magical_number(n, a, b):
    lcm_ab = a * b // math.gcd(a, b)
    mod = 10**9 + 7

    def count(x):
        return x // a + x // b - x // lcm_ab

    left, right = 1, n * min(a, b)
    while left < right:
        mid = (left + right) // 2
        if count(mid) < n:
            left = mid + 1
        else:
            right = mid
    return left % mod
```

---

## Smallest Divisor Given Threshold

**Problem:** Array and threshold. Divide each element by divisor (ceil), sum results. Find smallest divisor such that sum <= threshold.

**Approach:** Answer space [1, max(nums)]. Feasible(d): sum(ceil(n/d)) <= threshold.

```python
import math

def smallest_divisor(nums, threshold):
    def feasible(d):
        return sum(math.ceil(n / d) for n in nums) <= threshold

    left, right = 1, max(nums)
    while left < right:
        mid = (left + right) // 2
        if feasible(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

---

## Minimum Time to Complete Trips

**Problem:** Buses with time per trip. Total trips to complete. Find minimum time.

**Approach:** Answer space [1, totalTrips * max(time)]. Feasible(t): sum(t // time[i]) >= totalTrips.

```python
def minimum_time(time, totalTrips):
    def feasible(t):
        return sum(t // tm for tm in time) >= totalTrips

    left, right = 1, totalTrips * max(time)
    while left < right:
        mid = (left + right) // 2
        if feasible(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

---

## Maximum Candies Allocated to K Children

**Problem:** Candy piles. Each child gets piles from one pile only. Maximize minimum candies per child.

**Approach:** Answer space [1, max(candies)]. Feasible(x): sum(c // x for c in candies) >= k.

```python
def maximum_candies(candies, k):
    def feasible(x):
        return sum(c // x for c in candies) >= k

    left, right = 1, max(candies)
    while left < right:
        mid = (left + right + 1) // 2
        if feasible(mid):
            left = mid
        else:
            right = mid - 1
    return left
```

---

## Minimum Speed to Arrive on Time

**Problem:** Distances and max speeds. Must arrive by time. Find minimum constant speed (can go slower).

**Approach:** Answer space [1, 10^7]. Feasible(s): sum(ceil(dist[i]/s)) for i<n-1, plus dist[n-1]/s <= hour.

```python
import math

def min_speed_on_time(dist, hour):
    n = len(dist)
    if n >= 2 and hour <= n - 1:
        return -1

    def feasible(s):
        t = 0
        for i in range(n - 1):
            t += math.ceil(dist[i] / s)
        t += dist[-1] / s
        return t <= hour

    left, right = 1, 10**7
    while left < right:
        mid = (left + right) // 2
        if feasible(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

---

## Cutting Ribbons

**Problem:** Ribbon lengths, cut into k equal pieces. Maximize length of each piece.

**Approach:** Answer space [1, max(ribbons)]. Feasible(len): sum(r // len for r in ribbons) >= k.

```python
def max_ribbon_length(ribbons, k):
    def feasible(length):
        return sum(r // length for r in ribbons) >= k

    left, right = 1, max(ribbons)
    while left < right:
        mid = (left + right + 1) // 2
        if feasible(mid):
            left = mid
        else:
            right = mid - 1
    return left if feasible(left) else 0
```

---

## Find Kth Smallest Pair Distance

**Problem:** Array of numbers. Pair distance = |a-b|. Find kth smallest pair distance.

**Approach:** Answer space [0, max(nums)-min(nums)]. Feasible(d): count pairs with distance <= d. Binary search for smallest d with count >= k.

```python
def smallest_distance_pair(nums, k):
    nums.sort()
    n = len(nums)

    def count_pairs_at_most(d):
        count = 0
        j = 0
        for i in range(n):
            while j < n and nums[j] - nums[i] <= d:
                j += 1
            count += j - i - 1
        return count

    left, right = 0, nums[-1] - nums[0]
    while left < right:
        mid = (left + right) // 2
        if count_pairs_at_most(mid) >= k:
            right = mid
        else:
            left = mid + 1
    return left
```
