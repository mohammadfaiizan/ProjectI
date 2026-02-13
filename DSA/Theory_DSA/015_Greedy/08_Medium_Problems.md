# Medium Greedy Problems

## 1. Jump Game II

**Description**: Minimum jumps to reach last index.

**Approach**: BFS-like: at each step, extend to furthest reachable. Track jumps and current range.

---

## 2. Merge Intervals

**Description**: Merge overlapping intervals.

**Approach**: Sort by start; merge consecutive overlapping intervals.

---

## 3. Insert Interval

**Description**: Insert new interval into sorted non-overlapping intervals and merge.

**Approach**: Find position; merge overlapping; insert.

---

## 4. Non-overlapping Intervals

**Description**: Minimum intervals to remove so rest are non-overlapping.

**Approach**: Same as max non-overlapping: sort by end, count non-overlapping. Answer = n - count.

---

## 5. Task Scheduler

**Description**: Tasks with cooldown. Same task must be n apart. Min total time.

**Approach**: Schedule most frequent first; formula: (max_count-1)*(n+1) + num_max, or len(tasks).

---

## 6. Partition Labels

**Description**: Partition string so each letter in at most one part. Minimize number of parts.

**Approach**: Last index per char; extend partition until current index equals max last of partition chars.

---

## 7. Gas Station

**Description**: Circular route; gas at each station, cost to next. Find starting index to complete circuit.

**Approach**: If total gas >= total cost, solution exists. Start from 0; when tank negative, restart from next station.

---

## 8. Boats to Save People

**Description**: People with weights; boat limit; at most 2 per boat. Min boats.

**Approach**: Sort; pair heaviest with lightest if both fit; else heaviest alone.

---

## 9. Bag of Tokens

**Description**: Play face-up (spend power, gain score) or face-down (gain power, lose score). Maximize score.

**Approach**: Buy cheapest tokens (face-up), sell most expensive (face-down) when needed. Two pointers.

---

## 10. Reorganize String

**Description**: Reorder so no two adjacent same. Return "" if impossible.

**Approach**: Possible iff max_freq <= (n+1)//2. Max-heap; alternate with most frequent.

---

## 11. Remove K Digits

**Description**: Remove k digits from number string to get smallest possible number.

**Approach**: Monotonic stack: remove larger digits while k > 0. Keep result smallest.

---

## 12. Queue Reconstruction by Height

**Description**: People (h, k) where k = number of people in front with height >= h. Reconstruct queue.

**Approach**: Sort by h descending, k ascending. Insert each at position k (greedy: taller first, then by k).

---

## 13. Minimum Number of Arrows to Burst Balloons

**Description**: Intervals (balloons); arrow at x bursts all containing x. Min arrows.

**Approach**: Sort by end; count non-overlapping intervals (same as activity selection).

---

## 14. Meeting Rooms II

**Description**: Min rooms for all meetings.

**Approach**: Sweep line (start +1, end -1) or min-heap of end times.

---

## 15. Car Pooling

**Description**: Trips (num_passengers, start, end). Capacity limit. Possible?

**Approach**: Sweep line; track passenger count; reject if exceeds capacity.

---

## 16. Maximum Swap

**Description**: Swap two digits once to maximize number.

**Approach**: Find rightmost smaller digit and swap with rightmost larger digit to its right.

---

## 17. Wiggle Sort

**Description**: Reorder so nums[0] < nums[1] > nums[2] < nums[3] ...

**Approach**: Greedy swap: at odd index, ensure larger than neighbors; at even, ensure smaller.

---

## 18. Largest Number

**Description**: Concatenate numbers to form largest possible number.

**Approach**: Custom sort: a before b if a+b > b+a (string comparison).

---

## 19. Minimum Deletions to Make Character Frequencies Unique

**Description**: Delete min chars so no two chars have same frequency.

**Approach**: Sort frequencies descending; for each duplicate, reduce until unique (or 0).

---

## 20. Reduce Array Size to Half

**Description**: Remove min number of distinct integers so remaining count <= half.

**Approach**: Greedy: remove most frequent first. Sort by frequency descending.

---

## 21. Maximum Length of Pair Chain

**Description**: Pairs (a, b); chain if b_i < a_{i+1}. Max chain length.

**Approach**: Sort by end; activity selection (non-overlapping intervals).

---

## 22. Video Stitching

**Description**: Clips cover [0, time]. Min clips to cover.

**Approach**: Sort by start; greedy pick clip extending furthest at each step.

---

## 23. Minimum Taps to Open to Water a Garden

**Description**: Taps at positions with ranges. Min taps to cover [0, n].

**Approach**: Convert to intervals; greedy jump to furthest covering current position.

---

## 24. Broken Calculator

**Description**: Start from X, operations: multiply by 2 or subtract 1. Reach Y. Min operations.

**Approach**: Work backwards from Y: if Y > X, Y odd then add 1 else divide by 2. Greedy reverse.

---

## 25. Score of Parentheses

**Description**: () = 1, AB = A+B, (A) = 2*A. Compute score.

**Approach**: Stack; on ( push 0; on ) pop and add 2*top or 1 to new top.

---

## 26. Advantage Shuffle

**Description**: Permute A to maximize number of positions where A[i] > B[i].

**Approach**: Sort both; for each B[i], use smallest A[j] > B[i] (binary search or two pointers).

---

## 27. Minimum Operations to Reduce X to Zero

**Description**: Remove from left or right; sum removed = x. Min operations.

**Approach**: Equivalent to max subarray of sum = total - x. Sliding window or prefix sum.

---

## 28. Dota2 Senate

**Description**: R and D vote; each bans next opponent. Who wins?

**Approach**: Queue for each party; simulate; each bans next opponent. Party with members left wins.
