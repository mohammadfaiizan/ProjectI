# Hard Heap and Priority Queue Problems

## 1. Merge K Sorted Lists (Optimal)

**Description**: Merge k sorted linked lists. Optimal O(n log k) where n is total nodes.

**Approach**: Min-heap of (val, list_id, node). O(n log k) time, O(k) space for heap.

---

## 2. Find Median from Data Stream (Follow-up: Delete)

**Description**: Add, find median, and optionally delete element.

**Approach**: Two heaps with lazy deletion. Track deleted elements. On pop, skip deleted. Rebalance when too many deleted at top.

---

## 3. Sliding Window Median

**Description**: Median for each sliding window of size k.

**Approach**: Two heaps (lo, hi) with lazy deletion. When window slides, mark old element deleted. Rebalance. O(n log k).

---

## 4. Minimum Cost to Hire K Workers (Full)

**Description**: Hire exactly k workers. Pay each wage proportional to quality. Minimize total.

**Approach**: Sort by wage/quality ratio. For each as captain, take k workers with ratio <= captain. Min-heap of quality to maintain k smallest quality sum. O(n log k).

---

## 5. IPO (Multiple Rounds)

**Description**: Same as medium but with capital constraints and project dependencies.

**Approach**: Max-heap of profits. Sort projects by capital. Greedy select k. Handle project prerequisites with topological order if needed.

---

## 6. Smallest Range Covering Elements from K Lists

**Description**: Pick one element from each list. Minimize range [min, max].

**Approach**: Min-heap (val, list_id, idx). Track current max. Pop min, expand that list. Update best range. O(n log k).

---

## 7. Trapping Rain Water II

**Description**: 2D elevation map. Water trapped after raining.

**Approach**: Min-heap of boundary cells. Expand inward from boundary. Water level = max(heap min, cell height). O(mn log(mn)).

---

## 8. Kth Smallest Prime Fraction

**Description**: Sorted array of primes. Kth smallest fraction p[i]/p[j] where i < j.

**Approach**: Min-heap (p[i]/p[j], i, j). Start with (p[0]/p[j]) for all j. Pop k times, push (p[i+1]/p[j], i+1, j).

---

## 9. Minimum Number of Refueling Stops

**Description**: Drive to target. Limited fuel. Gas stations with position and fuel. Min stops.

**Approach**: Max-heap of fuel at passed stations. When fuel runs out, refuel from largest. Greedy. O(n log n).

---

## 10. Process Tasks Using Servers (Concurrent)

**Description**: Multiple tasks per second. Assign to available server with smallest weight.

**Approach**: Available heap (weight, idx), busy heap (free_time, weight, idx). At each time, free completed, assign new. O(n log n).

---

## 11. Maximum Performance of a Team

**Description**: n engineers with speed and efficiency. Pick k. Performance = sum(speed) * min(efficiency). Maximize.

**Approach**: Sort by efficiency descending. For each as min efficiency, take k fastest (min-heap of speed, keep k largest). O(n log k).

---

## 12. Find the K-Sum of an Array

**Description**: Array. Subsequence sum = sum of selected elements. Find kth largest subsequence sum.

**Approach**: Sort, take positive and largest negative. Max-heap of (sum, last_index). Expand by including next or excluding. Complex state.

---

## 13. Construct Target Array With Multiple Sums

**Description**: Start with [1,1,...,1]. Operation: replace element with sum of all. Can we get target?

**Approach**: Reverse process. Max-heap of target. Largest = sum of rest. Replace with (largest - sum_rest). Check if we reach all ones.

---

## 14. Minimum Cost to Reach Destination in Time

**Description**: Graph with edges (time, cost). Reach destination within maxTime. Minimize cost.

**Approach**: Dijkstra-like with (cost, node, time). State = (node, time). Min-heap by cost. Prune if time exceeded.

---

## 15. Number of Orders in the Backlog

**Description**: Buy/sell orders. Match when buy price >= sell price. Return sum of remaining order amounts.

**Approach**: Max-heap for buy orders, min-heap for sell orders. Match top of both. Process orders in sequence.

---

## 16. Maximum Number of Events That Can Be Attended

**Description**: Events have [start, end]. Attend one per day. Maximize events.

**Approach**: Sort by start. Min-heap of end times for events starting today. Each day, pop events that already ended. Attend one. Greedy.

---

## 17. Minimum Interval to Include Each Query

**Description**: Intervals and queries. For each query, find smallest interval length that contains it.

**Approach**: Sort intervals by left. For each query, add intervals with left <= query to heap (by right). Remove intervals with right < query. Min-heap by (length, right).

---

## 18. The Skyline Problem

**Description**: Buildings [left, right, height]. Return skyline key points.

**Approach**: Sweep line. Events (left, -height) and (right, height). Max-heap of active heights. When max changes, add point. Lazy deletion.
