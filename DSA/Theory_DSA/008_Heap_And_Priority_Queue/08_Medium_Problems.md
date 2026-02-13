# Medium Heap and Priority Queue Problems

## 1. Find Median from Data Stream

**Description**: Design structure to add numbers and return median.

**Approach**: Two heaps - max-heap for lower half, min-heap for upper half. Keep balanced. O(log n) add, O(1) median.

---

## 2. Task Scheduler

**Description**: Schedule tasks with cooldown n between same task. Minimize total time.

**Approach**: Max-heap by frequency. Each round pop up to n+1 tasks. Idle if heap empty. O(time) simulation.

---

## 3. Top K Frequent Words

**Description**: Return k most frequent words. Tie-break: lexicographically smaller first.

**Approach**: Count, min-heap of (-freq, word) size k. Custom comparator for tie-break.

---

## 4. Reorganize String

**Description**: Reorder so no two adjacent chars same.

**Approach**: Max-heap by frequency. Pop two most frequent alternately. Fail if one char > half.

---

## 5. Furthest Building You Can Reach

**Description**: Climb buildings with bricks and ladders. Ladders for any height, bricks for limited.

**Approach**: Min-heap of ladder jumps. When heap > ladders, use bricks for smallest jump. Greedy.

---

## 6. Minimum Cost to Hire K Workers

**Description**: Hire k workers. Pay = ratio * sum(quality). Ratio = wage/quality. Minimize total cost.

**Approach**: Sort by ratio. For each as "captain", min-heap of quality for k workers with ratio <= captain. O(n log k).

---

## 7. IPO - Maximize Capital

**Description**: Initial capital. Pick k projects (each has capital cost, profit). Maximize final capital.

**Approach**: Sort projects by capital. Max-heap of profits for affordable projects. Pop k times.

---

## 8. K Closest Points to Origin (with distance)

**Description**: Return k points closest to origin. Multiple solutions allowed.

**Approach**: Max-heap of size k by distance. Or quickselect. O(n log k) or O(n) average.

---

## 9. Sort Characters by Frequency (with stability)

**Description**: Sort by frequency, preserve order for same frequency.

**Approach**: Count, bucket sort or heap with (freq, original_index) for stability.

---

## 10. Find K Pairs with Smallest Sums

**Description**: Two sorted arrays. Find k pairs (a,b) with smallest a+b.

**Approach**: Min-heap (sum, i, j). Start (0,0). Expand (i+1,j) and (i,j+1). Avoid duplicates.

---

## 11. Kth Smallest Element in Sorted Matrix

**Description**: Matrix sorted row and column wise. Find kth smallest.

**Approach**: Min-heap (value, row, col). Start (0,0). Pop k times, push (r+1,c) and (r,c+1).

---

## 12. Sort Nearly Sorted Array (K-Sorted)

**Description**: Each element at most k positions from sorted position.

**Approach**: Min-heap of size k+1. Slide window. O(n log k).

---

## 13. Meeting Rooms II

**Description**: Min meeting rooms needed for non-overlapping intervals.

**Approach**: Sort by start. Min-heap of end times. If start >= min(end), reuse room. O(n log n).

---

## 14. Single-Threaded CPU

**Description**: Tasks have enqueue time and processing time. Process in order of shortest processing when multiple available.

**Approach**: Min-heap of (enqueue, processing, idx). Simulate time. Available tasks by processing time.

---

## 15. Process Tasks Using Servers

**Description**: Servers have weights. Assign tasks to servers. When multiple free, pick smallest weight.

**Approach**: Two heaps: available (weight, idx), busy (free_time, weight, idx). Simulate.

---

## 16. Minimum Refueling Stops

**Description**: Drive to target. Gas stations along the way. Min stops to refuel.

**Approach**: Max-heap of fuel at passed stations. When out of fuel, refuel from largest. Greedy.

---

## 17. Smallest Range Covering Elements from K Lists

**Description**: One element from each of k lists. Minimize range (max - min).

**Approach**: Min-heap (val, list_id, idx). Track current max. Expand list with min. Update range.

---

## 18. Sliding Window Maximum

**Description**: Max in each sliding window of size k.

**Approach**: Monotonic deque (not heap). Heap alternative: max-heap with lazy deletion. O(n log n).

---

## 19. Merge Intervals (with heap variant)

**Description**: Merge overlapping intervals.

**Approach**: Sort by start. Heap variant: min-heap by start, merge as we pop.

---

## 20. Network Delay Time

**Description**: Single source shortest path in weighted graph.

**Approach**: Dijkstra with min-heap. O((V+E) log V).

---

## 21. Path With Maximum Minimum Value

**Description**: Path from (0,0) to (n-1,m-1). Score = min value on path. Maximize score.

**Approach**: Max-heap (min_so_far, r, c). Dijkstra-like expansion. Pick largest min_so_far first.

---

## 22. Kth Largest Element in Array (Quickselect)

**Description**: Same as easy but with O(n) average quickselect.

**Approach**: Heap O(n log k) or quickselect O(n) average. Heap is simpler.

---

## 23. Reorganize String (Optimized)

**Description**: Same as above with O(n) bucket approach.

**Approach**: Count. If max > (n+1)/2 impossible. Interleave most frequent with rest.

---

## 24. Design Twitter

**Description**: Post tweet, follow, unfollow, get news feed (10 most recent from followees).

**Approach**: K-way merge with heap. Each user's tweets. Merge k sorted lists by timestamp.

---

## 25. Ugly Number II

**Description**: Nth number whose prime factors are only 2, 3, 5.

**Approach**: Min-heap. Start 1. Pop, push 2x, 3x, 5x. Avoid duplicates with set.
