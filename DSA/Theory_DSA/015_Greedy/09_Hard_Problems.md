# Hard Greedy Problems

## 1. Candy

**Description**: Children in line; each gets at least 1 candy; if rating higher than neighbor, more candies. Min total candies.

**Approach**: Two passes: left-to-right (if rating up, candy = prev+1), right-to-left (same). Take max at each position.

---

## 2. Trapping Rain Water II

**Description**: 2D elevation map; water trapped. 3D version of trapping rain water.

**Approach**: Min-heap from boundary; expand inward; water at cell = max(0, boundary_min - height).

---

## 3. Minimum Number of Refueling Stops

**Description**: Car with startFuel, target distance. Stations (position, fuel). Min stops to reach target.

**Approach**: Greedy: drive as far as possible; when out of fuel, refuel at station with most fuel passed. Max-heap of fuels.

---

## 4. IPO

**Description**: k projects; each has capital and profit. Start with w capital. Pick project only if capital <= w. Maximize capital after k projects.

**Approach**: Sort by capital; max-heap of profits for affordable projects. Each step pick max profit, add to capital.

---

## 5. Maximum Performance of a Team

**Description**: Engineers with speed and efficiency. Pick at most k; performance = sum(speeds) * min(efficiency). Maximize.

**Approach**: Sort by efficiency descending; for each as min efficiency, take top k speeds (min-heap of size k).

---

## 6. Merge k Sorted Lists

**Description**: Merge k sorted linked lists.

**Approach**: Min-heap of (val, list_node); pop min, push next from same list.

---

## 7. Minimum Cost to Hire K Workers

**Description**: Workers have quality and wage. Hire k; pay each at least their wage. Total wage = ratio * sum(quality). Minimize.

**Approach**: Sort by wage/quality; for each as "captain", take k-1 workers with smallest quality from those with ratio <= captain. Min total = sum(quality) * captain_ratio. Use heap to maintain k smallest quality.

---

## 8. Maximum Frequency Stack

**Description**: Push, pop. Pop returns most frequent element; tie-break by most recent.

**Approach**: Map freq to stack of elements. Track max_freq. Pop from max_freq stack.

---

## 9. Reconstruct Itinerary

**Description**: Tickets (from, to). Reconstruct itinerary using all tickets. Lexicographically smallest.

**Approach**: Euler path. Sort adjacency lists; DFS from JFK, backtrack and reverse path.

---

## 10. Minimum Window Substring

**Description**: Smallest substring of s containing all chars of t.

**Approach**: Sliding window; expand until valid, contract from left. Track char counts.

---

## 11. Maximum Events That Can Be Attended

**Description**: Events [start, end]; one event per day. Max events.

**Approach**: Sort by start; for each day, attend event with earliest end that covers that day. Min-heap of end times.

---

## 12. Employee Free Time

**Description**: Sorted intervals for each employee. Find common free time.

**Approach**: Merge all intervals; gaps between merged intervals are free time.

---

## 13. Minimum Interval to Include Each Query

**Description**: Intervals and queries. For each query, smallest interval containing it, or -1.

**Approach**: Sweep queries; maintain min-heap of (size, end) for intervals covering current query. Pop expired.

---

## 14. Maximum Profit in Job Scheduling

**Description**: Jobs with start, end, profit. No overlap. Max profit.

**Approach**: DP + binary search. Sort by end; dp[i] = max(profit[i] + dp[j], dp[i-1]) where j = latest non-overlapping.

---

## 15. Course Schedule III

**Description**: Courses (duration, lastDay). Take at most one at a time. Max courses.

**Approach**: Sort by lastDay; greedy take by deadline. If total time exceeds lastDay of current, remove longest duration course (max-heap).

---

## 16. Patching Array

**Description**: Sorted nums and n. Add min numbers so every [1, n] can be formed as sum of (nums + added).

**Approach**: Track max formable; while max < n, if next num <= max+1, add it; else add max+1. Greedy patch.

---

## 17. Create Maximum Number

**Description**: Two arrays; create length k number by picking digits from both (preserving order). Maximize.

**Approach**: Try all splits (i from first, k-i from second). For each array, get max subsequence of given length (monotonic stack). Merge two max subsequences (greedy merge). Take best.

---

## 18. Remove Duplicate Letters

**Description**: Remove duplicates so result is lexicographically smallest.

**Approach**: Monotonic stack; pop if char has more occurrences later and top > current. Track last index per char.
